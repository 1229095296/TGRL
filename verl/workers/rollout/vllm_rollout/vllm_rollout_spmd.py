# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank
  to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import logging
import os
import pickle
import socket
import threading
from contextlib import contextmanager
from copy import deepcopy
from types import MethodType
from typing import Any

import numpy as np
import ray
import torch
import torch.distributed
import zmq
from filelock import FileLock
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from vllm import LLM, SamplingParams
from vllm.distributed import parallel_state as vllm_ps
from vllm.lora.request import LoRARequest
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.worker.worker_base import WorkerWrapperBase

from verl import DataProto
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> list[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id
    # is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


class vLLMRollout(BaseRollout):
    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), (
            "tensor parallel size should be less than or equal to the world size"
        )
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)

        if kwargs.get("train_tp") is not None:
            # deployed with megatron
            import os

            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            vllm_ps.initialize_model_parallel(tensor_model_parallel_size=tensor_parallel_size)

        rope_scaling_config = getattr(model_hf_config, "rope_scaling", None)
        if not rope_scaling_config:
            max_position_embeddings = None
            if hasattr(model_hf_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.max_position_embeddings
            elif hasattr(model_hf_config, "llm_config") and hasattr(
                model_hf_config.llm_config, "max_position_embeddings"
            ):
                max_position_embeddings = model_hf_config.llm_config.max_position_embeddings
            elif hasattr(model_hf_config, "text_config") and hasattr(
                model_hf_config.text_config, "max_position_embeddings"
            ):
                max_position_embeddings = model_hf_config.text_config.max_position_embeddings
            if max_position_embeddings is None:
                raise ValueError("max_position_embeddings not found in model_hf_config")
            assert max_position_embeddings >= config.prompt_length + config.response_length, (
                "model context length should be greater than total sequence length"
            )
        else:
            # handle type where there's a length extend factor
            # see https://qwen.readthedocs.io/en/latest/deployment/vllm.html#extended-context-support
            # for using yarn as an example
            rope_scaling_factor = rope_scaling_config.get("factor", 1.0)

            assert (
                model_hf_config.max_position_embeddings * rope_scaling_factor
                >= config.prompt_length + config.response_length
            ), (
                "model context length should be greater than total sequence length, "
                + f"got rope_scaling_factor={rope_scaling_factor} and "
                + f"max_position_embeddings={model_hf_config.max_position_embeddings}"
            )

        max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )

        trust_remote_code = kwargs.get("trust_remote_code", False)
        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format

        lora_kwargs = kwargs.pop("lora_kwargs", {})
        self.lora_kwargs = lora_kwargs
        # copy it to avoid secretly modifying the engine config
        engine_kwargs = (
            {}
            if "engine_kwargs" not in config or "vllm" not in config.engine_kwargs
            else OmegaConf.to_container(deepcopy(config.engine_kwargs.vllm))
        )
        # For each vLLM engine parameter,
        # - `None` means not setting it, so we pop it, and leave it to vLLM default value
        #    (which can vary across different vLLM versions);
        # - Otherwise it's the desired value we want to explicitly set.
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        if config.get("limit_images", None):  # support for multi-image data
            engine_kwargs["limit_mm_per_prompt"] = {"image": config.get("limit_images")}

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=config.free_cache_engine,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
            **lora_kwargs,
            **engine_kwargs,
        )

        # Offload vllm model to reduce peak memory usage
        if config.free_cache_engine:
            self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        kwargs["detokenize"] = False

        # print(f"[DEBUG] Initial max_tokens from config.response_length: {config.response_length}")
        # print(f"[DEBUG] Config keys: {list(config.keys())}")

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)) and k != "seed":
                # old_value = kwargs.get(k, "NOT_SET")
                kwargs[k] = config.get(k)
                # if k == "max_tokens":
                #     print(f"[DEBUG] Config overriding max_tokens: {old_value} -> {config.get(k)}")
        kwargs["n"] = 1  # already repeat in ray_trainer
        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id
        self.tokenizer = tokenizer

        # CoT exploration mode configuration
        self.cot_exploration_mode = config.get("cot_exploration_mode", False)
        start_token_ids_cfg = config.get("cot_start_token_ids", None)
        end_token_ids_cfg = config.get("cot_end_token_ids", None)
        start_token_id_cfg = config.get("cot_start_token_id", None)
        end_token_id_cfg = config.get("cot_end_token_id", None)

        if start_token_ids_cfg is not None:
            self.cot_start_token_ids = [int(t) for t in start_token_ids_cfg]
        elif start_token_id_cfg is not None:
            self.cot_start_token_ids = [int(start_token_id_cfg)]
        else:
            encoded = self.tokenizer.encode("<think>", add_special_tokens=False)
            self.cot_start_token_ids = [int(t) for t in encoded] if len(encoded) > 0 else [151667]

        if end_token_ids_cfg is not None:
            self.cot_end_token_ids = [int(t) for t in end_token_ids_cfg]
        elif end_token_id_cfg is not None:
            self.cot_end_token_ids = [int(end_token_id_cfg)]
        else:
            encoded = self.tokenizer.encode("</think>", add_special_tokens=False)
            self.cot_end_token_ids = [int(t) for t in encoded] if len(encoded) > 0 else [151668]

        self.cot_start_token_id = self.cot_start_token_ids[0]
        self.cot_end_token_id = self.cot_end_token_ids[0]
        self.cot_top_k_per_sample = config.get("cot_top_k_per_sample", [1, 2])

        # Hybrid exploration strategy configuration
        hybrid_exploration_cfg = config.get("hybrid_exploration", {})
        self.hybrid_exploration_enabled = hybrid_exploration_cfg.get("enable", False)
        self.exploration_temperatures = hybrid_exploration_cfg.get("temperatures", [0.3, 1.2])
        self.exploration_top_k_values = hybrid_exploration_cfg.get("top_k_values", [20, 20])
        self.hybrid_baseline_count = int(hybrid_exploration_cfg.get("baseline_count", 1))
        self.hybrid_exploration_count = hybrid_exploration_cfg.get("exploration_count", None)

        # Print exploration strategy configuration
        if self.hybrid_exploration_enabled:
            print(f"[Exploration] Hybrid mode enabled: "
                  f"temperatures={self.exploration_temperatures}, "
                  f"top_k_values={self.exploration_top_k_values}")

        if self.cot_exploration_mode:
            if not self.hybrid_exploration_enabled:
                print(f"[CoT Exploration] Basic mode enabled: start_token_ids={self.cot_start_token_ids}, "
                      f"end_token_ids={self.cot_end_token_ids}, top_k_per_sample={self.cot_top_k_per_sample}")
            else:
                print(f"[CoT Exploration] CoT detection enabled with hybrid exploration")

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generate sequences for a batch of prompts.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object
            )

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(
                non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data"), strict=True
            ):
                vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
        else:
            vllm_inputs = [
                {"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
            ]

        # ensure the type of `prompt_token_ids` passed to vllm is list[int]
        # https://github.com/volcengine/verl/pull/772
        for input_data in vllm_inputs:
            if isinstance(input_data["prompt_token_ids"], np.ndarray):
                input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
            elif not isinstance(input_data["prompt_token_ids"], list):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}"
                )

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)

        kwargs = {}
        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }
        else:
            runtime_sampling_temperature = prompts.meta_info.get("runtime_sampling_temperature", None)
            if runtime_sampling_temperature is not None:
                kwargs["temperature"] = float(runtime_sampling_temperature)

        lora_requests = None
        if self.lora_kwargs:
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_requests = [
                    LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/simon-stub-path")
                ] * batch_size

        # Prepare sampling params with CoT exploration mode if enabled
        # Apply exploration strategies during training (not validation)
        # This includes: temperature exploration, CoT exploration, or both
        sampling_params_list = None
        sample_index_list = []  # Track sample_index for each request
        exploration_temp_list = []  # Track exploration_temperature for IS correction

        # Check if we're in warmup period
        in_warmup = prompts.meta_info.get("in_warmup", False)
        warmup_steps = prompts.meta_info.get("warmup_steps", 0)
        global_steps = prompts.meta_info.get("global_steps", 0)

        runtime_temperatures = prompts.meta_info.get("runtime_hybrid_temperatures", None)
        active_temperatures = list(runtime_temperatures) if runtime_temperatures is not None else list(self.exploration_temperatures)

        # During warmup, disable hybrid exploration (use standard GRPO with all samples at T1)
        hybrid_exploration_active = self.hybrid_exploration_enabled and not in_warmup

        if in_warmup and self.hybrid_exploration_enabled:
            print(
                f"[Warmup] Step {global_steps}/{warmup_steps}: "
                f"Using uniform temperature T={active_temperatures[1]} for all samples"
            )

        rollout_n = int(getattr(self.config, "n", 1))
        baseline_count_per_group = self.hybrid_baseline_count
        exploration_count_per_group = (
            int(self.hybrid_exploration_count)
            if self.hybrid_exploration_count is not None
            else max(rollout_n - baseline_count_per_group, 0)
        )
        if hybrid_exploration_active:
            if baseline_count_per_group < 0 or exploration_count_per_group < 0:
                raise ValueError(
                    f"Hybrid exploration counts must be non-negative, got "
                    f"baseline_count={baseline_count_per_group}, exploration_count={exploration_count_per_group}"
                )
            if baseline_count_per_group + exploration_count_per_group != rollout_n:
                raise ValueError(
                    f"Hybrid exploration counts must sum to rollout.n={rollout_n}, got "
                    f"baseline_count={baseline_count_per_group}, exploration_count={exploration_count_per_group}"
                )

        # Enable per-sample sampling params if any exploration strategy is enabled
        enable_per_sample_params = (self.cot_exploration_mode or hybrid_exploration_active) and not is_validate
        use_grouped_hybrid_generation = (
            hybrid_exploration_active
            and not self.cot_exploration_mode
            and len(active_temperatures) >= 2
            and rollout_n > 1
        )

        if use_grouped_hybrid_generation:
            sample_index_list = [i % rollout_n for i in range(batch_size)]
            exploration_temp_list = [
                active_temperatures[0] if sample_index < baseline_count_per_group else active_temperatures[1]
                for sample_index in sample_index_list
            ]
            baseline_count = sum(sample_index < baseline_count_per_group for sample_index in sample_index_list)
            exploration_count = batch_size - baseline_count
            print(
                f"[Hybrid Exploration] Step {global_steps}: grouped rollout "
                f"baseline={baseline_count} @ T={active_temperatures[0]}, "
                f"exploration={exploration_count} @ T={active_temperatures[1]}"
            )
        elif enable_per_sample_params:
            # Create per-request sampling params with sample_index in extra_args
            # Prompts are duplicated in ray_trainer and come in grouped order:
            # [prompt_0_sample_0, prompt_0_sample_1, ..., prompt_0_sample_n-1, prompt_1_sample_0, ...]
            sampling_params_list = []
            sample_modulus = rollout_n if hybrid_exploration_active else len(self.cot_top_k_per_sample)
            for i in range(batch_size):
                # Determine sample_index by position within each rollout group
                sample_index = i % sample_modulus
                sample_index_list.append(sample_index)

                # Create a copy of sampling params with extra_args
                # Use clone() instead of __dict__ to preserve all fields including max_tokens
                sp = self.sampling_params.clone()
                if sp.extra_args is None:
                    sp.extra_args = {}

                # Apply hybrid exploration strategy (if enabled and not in warmup)
                if hybrid_exploration_active:
                    # Samples [0, baseline_count) use T0; the rest share T1.
                    temp_index = 0 if sample_index < baseline_count_per_group else 1
                    sp.temperature = active_temperatures[temp_index]
                    top_k_index = min(temp_index, len(self.exploration_top_k_values) - 1)
                    sp.top_k = self.exploration_top_k_values[top_k_index]
                else:
                    # Fallback to default temperature exploration (backward compatible)
                    default_temps = [0.3, 1.2]
                    sp.temperature = default_temps[sample_index] if sample_index < len(default_temps) else 1.0

                sp.extra_args["sample_index"] = sample_index
                sp.extra_args["exploration_temperature"] = sp.temperature
                exploration_temp_list.append(sp.temperature)  # Collect for IS correction

                # Keep CoT token IDs for </think> detection
                sp.extra_args["cot_start_token_id"] = self.cot_start_token_id
                sp.extra_args["cot_end_token_id"] = self.cot_end_token_id
                sp.extra_args["cot_start_token_ids"] = self.cot_start_token_ids
                sp.extra_args["cot_end_token_ids"] = self.cot_end_token_ids

                # Optional: add repetition penalty for exploration samples
                if sample_index >= baseline_count_per_group and (
                    sp.repetition_penalty is None or sp.repetition_penalty == 1.0
                ):
                    sp.repetition_penalty = 1.05

                sampling_params_list.append(sp)

        # users can customize different sampling_params at different run
        # print(f"[DEBUG] kwargs passed to update_sampling_params: {kwargs}")
        with self.update_sampling_params(**kwargs):
            if use_grouped_hybrid_generation:
                response = [None] * batch_size
                rollout_log_probs = [None] * batch_size if self.config.calculate_log_probs else None

                def assign_generation_results(target_indices, outputs_subset):
                    for target_idx, output in zip(target_indices, outputs_subset, strict=True):
                        if len(output.outputs) != 1:
                            raise ValueError(
                                f"Expected one rollout output per prompt, got {len(output.outputs)} for index {target_idx}"
                            )
                        response_ids = output.outputs[0].token_ids
                        response[target_idx] = response_ids
                        if self.config.calculate_log_probs:
                            curr_log_prob = []
                            for token_pos, logprob in enumerate(output.outputs[0].logprobs):
                                curr_log_prob.append(logprob[response_ids[token_pos]].logprob)
                            rollout_log_probs[target_idx] = curr_log_prob

                baseline_indices = [
                    i for i, sample_index in enumerate(sample_index_list) if sample_index < baseline_count_per_group
                ]
                exploration_indices = [
                    i for i, sample_index in enumerate(sample_index_list) if sample_index >= baseline_count_per_group
                ]

                if baseline_indices:
                    baseline_sampling_params = self.sampling_params.clone()
                    baseline_sampling_params.temperature = active_temperatures[0]
                    baseline_sampling_params.top_k = self.exploration_top_k_values[
                        min(0, len(self.exploration_top_k_values) - 1)
                    ]
                    baseline_outputs = self.inference_engine.generate(
                        prompts=[vllm_inputs[i] for i in baseline_indices],
                        sampling_params=baseline_sampling_params,
                        lora_request=[lora_requests[i] for i in baseline_indices] if lora_requests is not None else None,
                        use_tqdm=False,
                    )
                    assign_generation_results(baseline_indices, baseline_outputs)

                if exploration_indices:
                    exploration_sampling_params = self.sampling_params.clone()
                    exploration_sampling_params.temperature = active_temperatures[1]
                    exploration_sampling_params.top_k = self.exploration_top_k_values[
                        min(1, len(self.exploration_top_k_values) - 1)
                    ]
                    if (
                        exploration_sampling_params.repetition_penalty is None
                        or exploration_sampling_params.repetition_penalty == 1.0
                    ):
                        exploration_sampling_params.repetition_penalty = 1.05
                    exploration_outputs = self.inference_engine.generate(
                        prompts=[vllm_inputs[i] for i in exploration_indices],
                        sampling_params=exploration_sampling_params,
                        lora_request=[lora_requests[i] for i in exploration_indices]
                        if lora_requests is not None
                        else None,
                        use_tqdm=False,
                    )
                    assign_generation_results(exploration_indices, exploration_outputs)

                if any(item is None for item in response):
                    raise RuntimeError("Grouped hybrid rollout did not fill all responses")
                if self.config.calculate_log_probs and any(item is None for item in rollout_log_probs):
                    raise RuntimeError("Grouped hybrid rollout did not fill all rollout log_probs")
            elif sampling_params_list is not None:
                # Use per-request sampling params
                # print(f"[DEBUG] sampling_params_list[0].max_tokens RIGHT BEFORE generate: {sampling_params_list[0].max_tokens}")
                outputs = self.inference_engine.generate(
                    prompts=vllm_inputs,
                    sampling_params=sampling_params_list,
                    lora_request=lora_requests,
                    use_tqdm=False,
                )
            else:
                # Use shared sampling params
                outputs = self.inference_engine.generate(
                    prompts=vllm_inputs,
                    sampling_params=self.sampling_params,
                    lora_request=lora_requests,
                    use_tqdm=False,
                )

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

            if not use_grouped_hybrid_generation:
                response = []
                rollout_log_probs = []
                for output_idx, output in enumerate(outputs):
                    for sample_id in range(len(output.outputs)):
                        response_ids = output.outputs[sample_id].token_ids
                        response.append(response_ids)

                        # Debug: print response info for first 2 samples
                        # if output_idx < 2:
                        #     stop_reason = output.outputs[sample_id].finish_reason
                        #     print(f"[Generation Debug] output={output_idx}, sample={sample_id}, "
                        #           f"response_len={len(response_ids)}, stop_reason={stop_reason}, "
                        #           f"token_ids={response_ids}")
                        if self.config.calculate_log_probs:
                            curr_log_prob = []
                            for i, logprob in enumerate(output.outputs[sample_id].logprobs):
                                curr_log_prob.append(logprob[response_ids[i]].logprob)
                            rollout_log_probs.append(curr_log_prob)

            response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(
                idx.device
            )
            if self.config.calculate_log_probs:
                rollout_log_probs = pad_2d_list_to_length(
                    rollout_log_probs, -1, max_length=self.config.response_length
                ).to(idx.device)
                rollout_log_probs = rollout_log_probs.to(torch.float32)

            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        if self.config.calculate_log_probs:
            # we will recompute old log prob with actor
            batch["rollout_log_probs"] = rollout_log_probs
        if len(sample_index_list) > 0:
            batch["sample_indices"] = torch.tensor(sample_index_list, dtype=torch.long, device=idx.device)
        if len(exploration_temp_list) > 0:
            batch["sample_temperatures"] = torch.tensor(exploration_temp_list, dtype=torch.float32, device=idx.device)

        # Add sample_index_list and exploration_temperature to non_tensor_batch
        # This is needed for both CoT exploration and temperature-based exploration
        if not is_validate and len(sample_index_list) > 0:
            non_tensor_batch["sample_index"] = np.array(sample_index_list, dtype=np.int32)
            # Also pass exploration temperatures for temperature-based advantage correction
            if len(exploration_temp_list) > 0:
                non_tensor_batch["exploration_temperature"] = np.array(exploration_temp_list, dtype=np.float32)

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


# https://github.com/vllm-project/vllm/issues/13175
def _monkey_patch_compute_logits(model, vocab_size: int):
    original_compute_logits = model.compute_logits

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        logits = original_compute_logits(hidden_states, sampling_metadata)
        logits[..., vocab_size:] = float("-inf")
        return logits

    model.compute_logits = MethodType(compute_logits, model)


class vLLMAsyncRollout:
    """vLLMAsyncRollout is a thin wrapper of WorkerWrapperBase,
    which is engine in single worker process.
    """

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        self.tokenizer = tokenizer

        # Engine is deferred to be initialized in init_worker
        self.config = config
        self.inference_engine: WorkerWrapperBase = None
        self.sharding_manager = None
        self.is_sleep = False
        self.address = self._init_zeromq()

    def _init_zeromq(self) -> str:
        tensor_parallel_size = self.config.tensor_model_parallel_size

        # single node: ipc, multi nodes: tcp
        local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
        socket_type = "ipc" if tensor_parallel_size <= local_world_size else "tcp"

        # File lock to prevent multiple workers listen to same port
        with FileLock("/tmp/verl_vllm_zmq.lock"):
            if socket_type == "ipc":
                pid = os.getpid()
                address = f"ipc:///tmp/verl_vllm_zmq_{pid}.ipc"
            else:
                ip, port = self._get_free_port()
                address = f"tcp://{ip}:{port}"
            context = zmq.Context()
            self.socket = context.socket(zmq.REP)
            self.socket.bind(address)

        self.loop_thread = threading.Thread(target=self._loop_forever)
        self.loop_thread.start()

        return address

    def _get_free_port(self):
        ip = ray.util.get_node_ip_address()
        with socket.socket() as sock:
            sock.bind(("", 0))
            port = sock.getsockname()[1]
        return ip, port

    def _loop_forever(self):
        while True:
            message = self.socket.recv()
            method, args, kwargs = pickle.loads(message)
            result = self.execute_method(method, *args, **kwargs)
            self.socket.send(pickle.dumps(result))

    def get_zeromq_address(self):
        return self.address

    def init_worker(self, all_kwargs: list[dict[str, Any]]):
        """Initialize worker engine."""
        all_kwargs[0]["rank"] = int(os.environ["RANK"])
        all_kwargs[0]["local_rank"] = 0

        self.vllm_config = all_kwargs[0]["vllm_config"]
        self.inference_engine = WorkerWrapperBase(vllm_config=self.vllm_config)
        self.inference_engine.init_worker(all_kwargs)

    def load_model(self, *args, **kwargs):
        self.inference_engine.load_model(*args, **kwargs)

        # inference engine is initialized now, update sharding manager
        self.sharding_manager.inference_engine = self.inference_engine
        self.sharding_manager.model_runner = self.inference_engine.worker.model_runner

        _monkey_patch_compute_logits(self.inference_engine.worker.model_runner.model, len(self.tokenizer))

    def sleep(self, *args, **kwargs):
        """Offload model weights and discard kv cache."""
        if self.is_sleep:
            return
        self.sharding_manager.__exit__(None, None, None)
        self.is_sleep = True

    def wake_up(self, *args, **kwargs):
        """Load model weights and build kv cache."""
        if not self.is_sleep:
            return
        self.sharding_manager.__enter__()  # pylint: disable=C2801
        self.is_sleep = False

    def execute_method(self, method: str | bytes, *args, **kwargs):
        if method == "init_worker":
            return self.init_worker(*args, **kwargs)
        elif method == "load_model":
            return self.load_model(*args, **kwargs)
        elif method == "sleep":
            return self.sleep(*args, **kwargs)
        elif method == "wake_up":
            return self.wake_up(*args, **kwargs)
        else:
            return self.inference_engine.execute_method(method, *args, **kwargs)
