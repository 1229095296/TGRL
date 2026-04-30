# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
Single Process Actor
"""

import logging
import os
from typing import Optional

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, get_policy_loss_fn, kl_penalty
from verl.utils.device import get_device_name, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()

    @staticmethod
    def compute_js_divergence_weights(
        logits: torch.Tensor,  # [batch_size, seq_len, vocab_size]
        T_low: float,
        T_high: float,
        response_mask: torch.Tensor,  # [batch_size, seq_len]
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Compute token-level weights based on JS divergence between two temperature distributions.

        Args:
            logits: Raw logits from model [batch_size, seq_len, vocab_size]
            T_low: Low temperature (e.g., 0.3)
            T_high: High temperature (e.g., 1.2)
            response_mask: Mask for valid tokens [batch_size, seq_len]
            eps: Small constant for numerical stability

        Returns:
            weights: Token-level weights with mean=1 per sequence [batch_size, seq_len]
        """
        import torch.nn.functional as F

        # Compute log-probs at two temperatures
        logp_low = F.log_softmax(logits / T_low, dim=-1)   # [B, L, V]
        logp_high = F.log_softmax(logits / T_high, dim=-1)

        p_low = logp_low.exp()
        p_high = logp_high.exp()

        # Mixture distribution m = 0.5*(p_low + p_high)
        m = 0.5 * (p_low + p_high)
        logm = torch.log(m + eps)

        # KL divergences
        kl_low = (p_low * (logp_low - logm)).sum(dim=-1)   # [B, L]
        kl_high = (p_high * (logp_high - logm)).sum(dim=-1)

        # JS divergence per token position
        js_div = 0.5 * (kl_low + kl_high)  # [B, L], bounded, >=0

        # Detach to prevent gradient flow through weights
        js_div = js_div.detach()

        # Normalize to mean=1 per sequence (only over valid tokens)
        # For each sequence, compute mean over masked tokens
        js_div_masked = js_div * response_mask
        num_valid_tokens = response_mask.sum(dim=-1, keepdim=True).clamp(min=1)  # [B, 1]
        mean_js = js_div_masked.sum(dim=-1, keepdim=True) / num_valid_tokens  # [B, 1]

        # Normalize: w = js_div / mean_js
        weights = (js_div + eps) / (mean_js + eps)  # [B, L]

        # Apply mask (set weights to 0 for invalid tokens)
        weights = weights * response_mask

        return weights

    @staticmethod
    def _prepare_sample_temperatures(
        temperature, batch_size: int, device: torch.device
    ) -> Optional[torch.Tensor]:
        if torch.is_tensor(temperature):
            temperature = temperature.to(device=device, dtype=torch.float32)
            if temperature.ndim == 0:
                return temperature.expand(batch_size)
            if temperature.ndim != 1 or temperature.size(0) != batch_size:
                raise ValueError(
                    f"Expected per-sample temperatures with shape [{batch_size}], got {tuple(temperature.shape)}"
                )
            return temperature
        return None

    @staticmethod
    def _scalar_temperature_for_fused_kernels(temperature, batch_size: int, device: torch.device) -> float:
        sample_temperatures = DataParallelPPOActor._prepare_sample_temperatures(temperature, batch_size, device)
        if sample_temperatures is None:
            return float(temperature)
        if not torch.allclose(sample_temperatures, sample_temperatures[0]):
            raise ValueError("Fused kernels do not support mixed per-sample temperatures in the same micro-batch")
        return float(sample_temperatures[0].item())

    def _forward_micro_batch(
        self,
        micro_batch,
        temperature,
        calculate_entropy=False,
        compute_js_weights=False,
        js_config=None,
        tampo_candidate_temperatures=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
            js_weights: # (bs, response_len) - JS weights (None if not computed)
            tampo_likelihoods: # (bs, num_temperatures) average log-likelihoods
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            if "image_bound" in micro_batch["multi_modal_inputs"][0]:  # minicpm-o logic
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = [inputs[key] for inputs in micro_batch["multi_modal_inputs"]]
            else:
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = torch.cat(
                        [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                    )

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            sample_temperatures = self._prepare_sample_temperatures(temperature, batch_size, input_ids.device)
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            tampo_likelihoods = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch.keys()
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = self._scalar_temperature_for_fused_kernels(
                        temperature, batch_size, input_ids.device
                    )
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

                    # Compute JS weights BEFORE temperature scaling if needed
                    js_weights_rmpad = None
                    if compute_js_weights and js_config is not None:
                        # Store original logits temporarily for JS computation
                        logits_original = logits_rmpad.clone()

                    if tampo_candidate_temperatures is not None:
                        response_mask = micro_batch["response_mask"].to(device=input_ids.device, dtype=torch.float32)
                        num_response_tokens = response_mask.sum(dim=-1).clamp(min=1.0)
                        tampo_likelihoods_lst = []
                        for candidate_temperature in tampo_candidate_temperatures:
                            candidate_logits = logits_rmpad.float() / float(candidate_temperature)
                            candidate_log_probs = logprobs_from_logits(
                                logits=candidate_logits,
                                labels=input_ids_rmpad_rolled,
                                inplace_backward=False,
                            )
                            if self.use_ulysses_sp:
                                candidate_log_probs = gather_outputs_and_unpad(
                                    candidate_log_probs,
                                    gather_dim=0,
                                    unpad_dim=0,
                                    padding_size=pad_size,
                                )
                            full_candidate_log_probs = pad_input(
                                hidden_states=candidate_log_probs.unsqueeze(-1),
                                indices=indices,
                                batch=batch_size,
                                seqlen=seqlen,
                            ).squeeze(-1)
                            response_candidate_log_probs = full_candidate_log_probs[
                                :, -response_length - 1 : -1
                            ]
                            avg_log_likelihood = (
                                response_candidate_log_probs * response_mask
                            ).sum(dim=-1) / num_response_tokens
                            tampo_likelihoods_lst.append(avg_log_likelihood)
                        tampo_likelihoods = torch.stack(tampo_likelihoods_lst, dim=-1).detach()

                    if sample_temperatures is None:
                        logits_rmpad.div_(float(temperature))
                    else:
                        temperature_rmpad = sample_temperatures.unsqueeze(-1).expand(batch_size, seqlen).reshape(-1, 1)
                        temperature_rmpad = index_first_axis(temperature_rmpad, indices).squeeze(-1)
                        if self.use_ulysses_sp:
                            temperature_rmpad, _, _ = ulysses_pad_and_slice_inputs(
                                temperature_rmpad.unsqueeze(0),
                                position_ids_rmpad=None,
                                sp_size=self.ulysses_sequence_parallel_size,
                            )
                            temperature_rmpad = temperature_rmpad.squeeze(0)
                            temperature_rmpad = torch.where(
                                temperature_rmpad > 0,
                                temperature_rmpad,
                                torch.ones_like(temperature_rmpad),
                            )
                        logits_rmpad = logits_rmpad / temperature_rmpad.unsqueeze(-1)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy or compute_js_weights or tampo_candidate_temperatures is not None:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                    # Compute JS weights using original logits
                    if compute_js_weights and js_config is not None:
                        # Compute JS divergence from original logits
                        logits_T0 = logits_original / js_config['T0']
                        logits_T1 = logits_original / js_config['T1']

                        # Compute log-probs at two temperatures
                        import torch.nn.functional as F
                        logp_low = F.log_softmax(logits_T0, dim=-1)   # [total_nnz, V]
                        logp_high = F.log_softmax(logits_T1, dim=-1)

                        p_low = logp_low.exp()
                        p_high = logp_high.exp()

                        # Mixture distribution m = 0.5*(p_low + p_high)
                        m = 0.5 * (p_low + p_high)
                        logm = torch.log(m + 1e-8)

                        # KL divergences
                        kl_low = (p_low * (logp_low - logm)).sum(dim=-1)   # [total_nnz]
                        kl_high = (p_high * (logp_high - logm)).sum(dim=-1)

                        # JS divergence per token position
                        js_weights_rmpad = 0.5 * (kl_low + kl_high)  # [total_nnz]
                        # Clamp to ensure non-negative (numerical precision issue)
                        js_weights_rmpad = torch.clamp(js_weights_rmpad, min=0.0)
                        js_weights_rmpad = js_weights_rmpad.detach()

                        # logits_original will be released here

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    if js_weights_rmpad is not None:
                        js_weights_rmpad = gather_outputs_and_unpad(
                            js_weights_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )
                if js_weights_rmpad is not None:
                    full_js_weights = pad_input(
                        hidden_states=js_weights_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                js_weights = None
                if js_weights_rmpad is not None:
                    js_weights = full_js_weights.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = self._scalar_temperature_for_fused_kernels(
                        temperature, batch_size, input_ids.device
                    )
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)
                    js_weights = None  # Fused kernels don't support JS weights yet

                else:
                    logits = output.logits

                    # Compute JS weights BEFORE temperature scaling if needed
                    js_weights = None
                    if compute_js_weights and js_config is not None:
                        # Store original logits temporarily for JS computation
                        logits_original = logits[:, -response_length - 1 : -1, :].clone()  # (bsz, response_length, vocab_size)

                    if tampo_candidate_temperatures is not None:
                        response_mask = micro_batch["response_mask"].to(device=logits.device, dtype=torch.float32)
                        num_response_tokens = response_mask.sum(dim=-1).clamp(min=1.0)
                        response_logits_raw = logits[:, -response_length - 1 : -1, :]
                        tampo_likelihoods_lst = []
                        for candidate_temperature in tampo_candidate_temperatures:
                            candidate_logits = response_logits_raw.float() / float(candidate_temperature)
                            candidate_log_probs = logprobs_from_logits(
                                logits=candidate_logits,
                                labels=micro_batch["responses"],
                                inplace_backward=False,
                            )
                            avg_log_likelihood = (
                                candidate_log_probs * response_mask
                            ).sum(dim=-1) / num_response_tokens
                            tampo_likelihoods_lst.append(avg_log_likelihood)
                        tampo_likelihoods = torch.stack(tampo_likelihoods_lst, dim=-1).detach()

                    if sample_temperatures is None:
                        logits.div_(float(temperature))
                    else:
                        logits = logits / sample_temperatures.view(batch_size, 1, 1)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

                    # Compute JS weights using original logits
                    if compute_js_weights and js_config is not None:
                        # Compute JS divergence from original logits
                        logits_T0 = logits_original / js_config['T0']
                        logits_T1 = logits_original / js_config['T1']

                        # Compute log-probs at two temperatures
                        import torch.nn.functional as F
                        logp_low = F.log_softmax(logits_T0, dim=-1)   # [B, L, V]
                        logp_high = F.log_softmax(logits_T1, dim=-1)

                        p_low = logp_low.exp()
                        p_high = logp_high.exp()

                        # Mixture distribution m = 0.5*(p_low + p_high)
                        m = 0.5 * (p_low + p_high)
                        logm = torch.log(m + 1e-8)

                        # KL divergences
                        kl_low = (p_low * (logp_low - logm)).sum(dim=-1)   # [B, L]
                        kl_high = (p_high * (logp_high - logm)).sum(dim=-1)

                        # JS divergence per token position
                        js_weights = 0.5 * (kl_low + kl_high)  # [B, L]
                        # Clamp to ensure non-negative (numerical precision issue)
                        js_weights = torch.clamp(js_weights, min=0.0)
                        js_weights = js_weights.detach()

                        # logits_original will be released here

            return entropy, log_probs, js_weights, tampo_likelihoods

    def compute_entropy_increments(
        self,
        logits: torch.Tensor,
        action_mask: torch.Tensor,
        T0: float = 0.01,
        T1: float = 1.2,
        max_delta_h: float = 6.0,
    ) -> torch.Tensor:
        """
        Compute entropy increment ΔH_t = H(T1) - H(T0) for each token.

        Uses numerically stable entropy computation:
        H(T) = logsumexp(logits/T) - Σ p * (logits/T)
        where p = softmax(logits/T)

        Args:
            logits: Tensor of shape [batch_size, seq_len, vocab_size]
            action_mask: Tensor of shape [batch_size, seq_len]
            T0: Reference temperature (deterministic baseline)
            T1: Exploratory temperature
            max_delta_h: Maximum entropy increment (clamp to [0, max_delta_h])

        Returns:
            delta_H: Tensor of shape [batch_size, seq_len] with entropy increments
        """
        # Compute entropy at T0
        logits_T0 = logits / T0
        H_T0 = verl_F.entropy_from_logits(logits_T0)  # [batch_size, seq_len]

        # Compute entropy at T1
        logits_T1 = logits / T1
        H_T1 = verl_F.entropy_from_logits(logits_T1)  # [batch_size, seq_len]

        # Compute increment
        delta_H = H_T1 - H_T0  # [batch_size, seq_len]

        # Apply action mask (zero out prompt/padding tokens)
        delta_H = delta_H * action_mask

        # Clamp to [0, max] for stability
        delta_H = torch.clamp(delta_H, min=0.0, max=max_delta_h)

        return delta_H

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(
        self,
        data: DataProto,
        calculate_entropy=False,
        compute_js_weights=False,
        js_config=None,
        tampo_candidate_temperatures=None,
    ) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

            calculate_entropy (bool): Whether to calculate entropy
            compute_js_weights (bool): Whether to compute JS weights for token-level weighting
            js_config (dict): Configuration for JS computation (T0, T1)

        Returns:
            tuple: (log_probs, entropys, js_weights) where js_weights is None if not computed
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info.get("temperature", 1.0)  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        if tampo_candidate_temperatures is not None:
            select_keys.append("response_mask")
        if "sample_temperatures" in data.batch.keys():
            select_keys.append("sample_temperatures")
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        js_weights_lst = []
        tampo_likelihoods_lst = []
        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            micro_batch_temperature = model_inputs.get("sample_temperatures", temperature)
            with torch.no_grad():
                entropy, log_probs, js_weights, tampo_likelihoods = self._forward_micro_batch(
                    model_inputs,
                    temperature=micro_batch_temperature,
                    calculate_entropy=calculate_entropy,
                    compute_js_weights=compute_js_weights,
                    js_config=js_config,
                    tampo_candidate_temperatures=tampo_candidate_temperatures,
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)
            if js_weights is not None:
                js_weights_lst.append(js_weights)
            if tampo_likelihoods is not None:
                tampo_likelihoods_lst.append(tampo_likelihoods)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        js_weights_all = None
        tampo_likelihoods_all = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if js_weights_lst:
            js_weights_all = torch.concat(js_weights_lst, dim=0)
        if tampo_likelihoods_lst:
            tampo_likelihoods_all = torch.concat(tampo_likelihoods_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
            if js_weights_all is not None:
                js_weights_all = restore_dynamic_batch(js_weights_all, batch_idx_list)
            if tampo_likelihoods_all is not None:
                tampo_likelihoods_all = restore_dynamic_batch(tampo_likelihoods_all, batch_idx_list)

        return log_probs, entropys, js_weights_all, tampo_likelihoods_all

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if "sample_temperatures" in data.batch.keys():
            select_keys.append("sample_temperatures")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]
                    is_weights = model_inputs.get("is_weights", None)  # IS weights for off-policy correction

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = (
                        self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    )
                    clip_ratio_high = (
                        self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    )
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    micro_batch_temperature = model_inputs.get("sample_temperatures", temperature)
                    entropy, log_prob, _, _ = self._forward_micro_batch(
                        model_inputs, temperature=micro_batch_temperature, calculate_entropy=calculate_entropy
                    )

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")

                    if self.config.policy_loss.loss_mode == "vanilla":
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            cliprange=clip_ratio,
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                            clip_ratio_c=clip_ratio_c,
                            loss_agg_mode=loss_agg_mode,
                            is_weights=is_weights,  # Pass IS weights for off-policy correction
                        )

                    else:
                        policy_loss_fn = get_policy_loss_fn(loss_mode)
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                        )

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item()
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (response_mask.shape[0] / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    micro_batch_metrics.update(
                        {
                            "actor/pg_loss": pg_loss.detach().item(),
                            "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                            "actor/ppo_kl": ppo_kl.detach().item(),
                            "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        }
                    )
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
