#!/usr/bin/env python3
"""
JS token intervention analysis with staged backends:
1. vLLM rollout generation for high-throughput exploration sampling.
2. HuggingFace replay to recover full logits for JS / entropy / margin.
3. vLLM greedy continuation for edited prefixes.
"""

import argparse
import gc
import json
import math
import os
import random
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from js_token_intervention import (
    InterventionResult,
    aggregate_outputs,
    build_intervention_job,
    build_prompt_ids,
    compute_js_entropy_margin,
    compute_score_float,
    get_torch_dtype,
    select_spans,
    summarize_results,
)


def parse_args():
    parser = argparse.ArgumentParser(description="JS token intervention analysis using vLLM + HF replay.")
    parser.add_argument(
        "--phase",
        type=str,
        default="full",
        choices=["full", "rollout", "replay", "continue", "seq_replay", "seq_continue", "seq_finalize"],
    )
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--prompt-key", type=str, default="prompt")
    parser.add_argument("--reward-model-key", type=str, default="reward_model")
    parser.add_argument("--data-source-key", type=str, default="data_source")
    parser.add_argument("--extra-info-key", type=str, default="extra_info")
    parser.add_argument("--max-examples", type=int, default=256)
    parser.add_argument("--target-successes", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--t0", type=float, default=0.3)
    parser.add_argument("--t1", type=float, default=1.2)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--rollouts-per-prompt", type=int, default=8)
    parser.add_argument("--analysis-batch-size", type=int, default=16)
    parser.add_argument("--positions-per-sample", type=int, default=1)
    parser.add_argument("--intervention-granularity", type=str, default="single", choices=["single", "block"])
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument("--intervention-mode", type=str, default="one_shot", choices=["one_shot", "sequential"])
    parser.add_argument("--sequential-rounds", type=int, default=3)
    parser.add_argument("--sequential-gap", type=int, default=2)
    parser.add_argument("--matched-random-repeats", type=int, default=5)
    parser.add_argument("--selectors", type=str, default="js,matched_random,entropy,low_margin")
    parser.add_argument("--replacement-strategy", type=str, default="t0_top1_else_t0_top2")
    parser.add_argument("--allow-eos-replacement", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--vllm-max-model-len", type=int, default=None)
    parser.add_argument("--intervention-model-device", type=str, default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard-rank", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--skip-finalize", action="store_true")
    parser.add_argument("--round-idx", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--gpu-keepalive", action="store_true")
    parser.add_argument("--gpu-keepalive-size", type=int, default=2048)
    parser.add_argument("--gpu-keepalive-interval", type=float, default=0.02)
    parser.add_argument("--gpu-keepalive-dtype", type=str, default="float16", choices=["bfloat16", "float16", "float32"])
    return parser.parse_args()


def init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="gloo")
    return rank, world_size, local_rank


def distributed_barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def get_visible_gpu_count() -> int:
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not cuda_visible:
        return torch.cuda.device_count()
    return len([item for item in cuda_visible.split(",") if item.strip()])


def init_vllm_engine(args):
    visible_gpu_count = max(get_visible_gpu_count(), 1)
    tp_size = min(args.tensor_parallel_size, visible_gpu_count)
    print(
        f"Initializing vLLM: visible_gpus={visible_gpu_count}, "
        f"tensor_parallel_size={tp_size}, gpu_mem_util={args.vllm_gpu_memory_utilization}"
    )
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=tp_size,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        seed=args.seed,
        disable_log_stats=True,
        max_model_len=args.vllm_max_model_len,
    )
    return llm


def cleanup_vllm_engine(llm):
    if llm is None:
        return
    try:
        llm.llm_engine.model_executor.shutdown()
    except Exception:
        pass
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def cleanup_torch_model(model):
    if model is None:
        return
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _gpu_keepalive_worker(
    device_idx: int,
    size: int,
    interval: float,
    dtype: torch.dtype,
    stop_event: threading.Event,
):
    try:
        device = torch.device(f"cuda:{device_idx}")
        torch.cuda.set_device(device)
        a = torch.randn((size, size), device=device, dtype=dtype)
        b = torch.randn((size, size), device=device, dtype=dtype)
        out = torch.empty((size, size), device=device, dtype=dtype)
        while not stop_event.is_set():
            torch.mm(a, b, out=out)
            torch.cuda.synchronize(device)
            if interval > 0:
                time.sleep(interval)
    except Exception:
        return


class GpuKeepalive:
    def __init__(self, args):
        self.args = args
        self.enabled = bool(args.gpu_keepalive and torch.cuda.is_available())
        self.stop_event = threading.Event()
        self.threads: list[threading.Thread] = []

    def start(self):
        if not self.enabled:
            return
        visible_gpu_count = get_visible_gpu_count()
        if visible_gpu_count <= 0:
            return
        dtype = get_torch_dtype(self.args.gpu_keepalive_dtype)
        for device_idx in range(visible_gpu_count):
            thread = threading.Thread(
                target=_gpu_keepalive_worker,
                args=(
                    device_idx,
                    self.args.gpu_keepalive_size,
                    self.args.gpu_keepalive_interval,
                    dtype,
                    self.stop_event,
                ),
                daemon=True,
            )
            thread.start()
            self.threads.append(thread)

    def stop(self):
        if not self.threads:
            return
        self.stop_event.set()
        for thread in self.threads:
            thread.join(timeout=1.0)
        self.threads.clear()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()


def to_jsonable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def write_jsonl(path: Path, rows: list[dict]):
    with path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_rank_world_size(args):
    if args.shard_rank is not None or args.num_shards is not None:
        if args.shard_rank is None or args.num_shards is None:
            raise ValueError("Both --shard-rank and --num-shards must be provided together.")
        return int(args.shard_rank), int(args.num_shards), False
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, world_size, world_size > 1


def get_successes_path(partial_dir: Path, rank: int) -> Path:
    return partial_dir / f"successful_rollouts_rank_{rank:03d}.jsonl"


def get_phase1_meta_path(partial_dir: Path, rank: int) -> Path:
    return partial_dir / f"phase1_rank_{rank:03d}.json"


def get_jobs_path(partial_dir: Path, rank: int) -> Path:
    return partial_dir / f"intervention_jobs_rank_{rank:03d}.jsonl"


def get_current_states_path(partial_dir: Path, rank: int) -> Path:
    return partial_dir / f"current_states_rank_{rank:03d}.jsonl"


def get_seq_jobs_path(partial_dir: Path, rank: int, round_idx: int) -> Path:
    return partial_dir / f"sequential_jobs_rank_{rank:03d}_round_{round_idx:02d}.jsonl"


def infer_completed_rounds(partial_dir: Path, rank: int) -> int:
    current_states_path = get_current_states_path(partial_dir, rank)
    states = read_jsonl(current_states_path)
    if not states:
        return 0
    return min(len(state.get("round_positions", [])) for state in states)


def build_phase_cmd(phase: str) -> list[str]:
    argv = list(sys.argv[1:])
    cleaned = []
    skip_next = False
    for idx, item in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if item == "--phase":
            skip_next = True
            continue
        if item.startswith("--phase="):
            continue
        if item == "--round-idx":
            skip_next = True
            continue
        if item.startswith("--round-idx="):
            continue
        cleaned.append(item)
    return [sys.executable, __file__, "--phase", phase, *cleaned]


def run_phase_subprocess(phase: str, round_idx: int | None = None):
    cmd = build_phase_cmd(phase)
    if round_idx is not None:
        cmd.extend(["--round-idx", str(round_idx)])
    proc = subprocess.run(cmd, check=False, cwd=str(Path(__file__).resolve().parent.parent))
    if proc.returncode != 0:
        raise RuntimeError(f"Phase {phase} failed with return code {proc.returncode}: {' '.join(cmd)}")


def state_key_from_fields(example_index: int, rollout_index: int, selector: str, repeat_index: int) -> str:
    return f"{example_index}:{rollout_index}:{selector}:{repeat_index}"


def state_key(state: dict) -> str:
    return state_key_from_fields(
        int(state["example_index"]),
        int(state["rollout_index"]),
        str(state["selector"]),
        int(state["repeat_index"]),
    )


def init_sequential_states(args, successful_rollouts: list[dict]) -> list[dict]:
    selectors = [selector.strip() for selector in args.selectors.split(",") if selector.strip()]
    states = []
    for item in successful_rollouts:
        for selector in selectors:
            repeats = args.matched_random_repeats if selector == "matched_random" else 1
            for repeat_idx in range(repeats):
                states.append(
                    {
                        "state_key": state_key_from_fields(
                            item["example_index"], item["rollout_index"], selector, repeat_idx
                        ),
                        "example_index": int(item["example_index"]),
                        "rollout_index": int(item["rollout_index"]),
                        "selector": selector,
                        "repeat_index": int(repeat_idx),
                        "prompt_ids": item["prompt_ids"],
                        "response_token_ids": item["response_token_ids"],
                        "original_length": len(item["response_token_ids"]),
                        "original_score": float(item["original_score"]),
                        "current_score": float(item["original_score"]),
                        "data_source": item["data_source"],
                        "ground_truth": item["ground_truth"],
                        "extra_info": item["extra_info"],
                        "round_positions": [],
                        "round_position_ratios": [],
                        "round_js_values": [],
                        "round_entropy_values": [],
                        "round_margin_values": [],
                        "round_original_tokens": [],
                        "round_replacement_tokens": [],
                        "round_original_token_texts": [],
                        "round_replacement_token_texts": [],
                        "total_replaced_tokens": 0,
                    }
                )
    return states


def build_masked_positions(round_positions: list[list[int]], total_positions: int, gap: int) -> set[int]:
    masked = set()
    for span in round_positions:
        if not span:
            continue
        start = max(0, span[0] - gap)
        end = min(total_positions - 1, span[-1] + gap)
        masked.update(range(start, end + 1))
    return masked


def build_prompt_token_ids(tokenizer, prompt_messages, enable_thinking: bool) -> list[int]:
    prompt_ids = build_prompt_ids(tokenizer, prompt_messages, enable_thinking=enable_thinking)
    return prompt_ids[0].tolist()


def generate_rollouts_vllm(
    llm: LLM,
    prompt_token_ids_list: list[list[int]],
    rollouts_per_prompt: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    seed: int,
):
    prompts = [{"prompt_token_ids": prompt_token_ids} for prompt_token_ids in prompt_token_ids_list]
    sampling_params = SamplingParams(
        n=rollouts_per_prompt,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k if top_k > 0 else -1,
        max_tokens=max_new_tokens,
        seed=seed,
        skip_special_tokens=False,
    )
    outputs = llm.generate(prompts, sampling_params)
    processed = []
    for output in outputs:
        response_texts = [item.text for item in output.outputs]
        response_token_ids_list = [item.token_ids for item in output.outputs]
        processed.append((response_texts, response_token_ids_list))
    return processed


@torch.inference_mode()
def replay_response_logits(model, prompt_ids: torch.Tensor, response_token_ids: list[int]) -> torch.Tensor:
    if not response_token_ids:
        return torch.empty((0, 0))
    device = next(model.parameters()).device
    prompt_ids = prompt_ids.to(device)
    response_tensor = torch.tensor([response_token_ids], dtype=torch.long, device=device)
    full_ids = torch.cat([prompt_ids, response_tensor], dim=1)
    attention_mask = torch.ones_like(full_ids, device=device)
    outputs = model(
        input_ids=full_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    prompt_length = prompt_ids.shape[1]
    return outputs.logits[0, prompt_length - 1 : -1, :].float().cpu()


def build_intervention_jobs_for_success(
    args,
    tokenizer,
    success_item,
    response_logits_t1: torch.Tensor,
):
    response_token_ids = success_item["response_token_ids"]
    original_length = len(response_token_ids)
    js_values, entropy_values, margin_values = compute_js_entropy_margin(list(response_logits_t1), t0=args.t0, t1=args.t1)
    candidate_positions = list(range(original_length))
    if not args.allow_eos_replacement and tokenizer.eos_token_id is not None:
        candidate_positions = [idx for idx in candidate_positions if response_token_ids[idx] != tokenizer.eos_token_id]
    if not candidate_positions:
        return []

    filtered_js = [js_values[idx] for idx in candidate_positions]
    filtered_entropy = [entropy_values[idx] for idx in candidate_positions]
    filtered_margin = [margin_values[idx] for idx in candidate_positions]

    selector_span_items = []
    per_example_rng = random.Random(args.seed + success_item["example_index"] * 9973 + success_item["rollout_index"])
    anchor_spans = select_spans(
        selector="js",
        js_values=filtered_js,
        entropy_values=filtered_entropy,
        margin_values=filtered_margin,
        valid_positions=candidate_positions,
        top_k=args.positions_per_sample,
        rng=per_example_rng,
        granularity=args.intervention_granularity,
        block_size=args.block_size,
    )
    selectors = [selector.strip() for selector in args.selectors.split(",") if selector.strip()]
    if "js" in selectors:
        selector_span_items.append(("js", 0, anchor_spans))

    for selector in selectors:
        if selector == "js":
            continue
        if selector == "matched_random":
            for repeat_idx in range(args.matched_random_repeats):
                chosen_spans = select_spans(
                    selector=selector,
                    js_values=filtered_js,
                    entropy_values=filtered_entropy,
                    margin_values=filtered_margin,
                    valid_positions=candidate_positions,
                    top_k=args.positions_per_sample,
                    rng=random.Random(
                        args.seed
                        + success_item["example_index"] * 1619
                        + success_item["rollout_index"] * 131
                        + repeat_idx
                    ),
                    granularity=args.intervention_granularity,
                    block_size=args.block_size,
                    anchor_spans=anchor_spans,
                )
                selector_span_items.append((selector, repeat_idx, chosen_spans))
        else:
            chosen_spans = select_spans(
                selector=selector,
                js_values=filtered_js,
                entropy_values=filtered_entropy,
                margin_values=filtered_margin,
                valid_positions=candidate_positions,
                top_k=args.positions_per_sample,
                rng=per_example_rng,
                granularity=args.intervention_granularity,
                block_size=args.block_size,
            )
            selector_span_items.append((selector, 0, chosen_spans))

    jobs = []
    prompt_ids = torch.tensor([success_item["prompt_ids"]], dtype=torch.long)
    for selector_name, repeat_index, spans in selector_span_items:
        for span in spans:
            job = build_intervention_job(
                example_index=success_item["example_index"],
                rollout_index=success_item["rollout_index"],
                selector_name=selector_name,
                repeat_index=repeat_index,
                positions=span,
                response_token_ids=response_token_ids,
                raw_logits=list(response_logits_t1),
                prompt_ids=prompt_ids,
                tokenizer=tokenizer,
                args=args,
                original_score=success_item["original_score"],
                data_source=success_item["data_source"],
                ground_truth=success_item["ground_truth"],
                extra_info=success_item["extra_info"],
                js_values=js_values,
                entropy_values=entropy_values,
                margin_values=margin_values,
            )
            if job is not None:
                jobs.append(job)
    return jobs


def greedy_continue_vllm_batch(llm: LLM, prefix_token_ids_list: list[list[int]], remaining_budgets: list[int]):
    if not prefix_token_ids_list:
        return []
    max_remaining_budget = max(remaining_budgets)
    if max_remaining_budget <= 0:
        return [[] for _ in prefix_token_ids_list]

    prompts = [{"prompt_token_ids": prompt_token_ids} for prompt_token_ids in prefix_token_ids_list]
    sampling_params = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=max_remaining_budget,
        skip_special_tokens=False,
    )
    outputs = llm.generate(prompts, sampling_params)
    suffix_lists = []
    for output, budget in zip(outputs, remaining_budgets):
        token_ids = output.outputs[0].token_ids[:budget]
        suffix_lists.append(token_ids)
    return suffix_lists


def sample_continue_vllm_batch(
    llm: LLM,
    prefix_token_ids_list: list[list[int]],
    remaining_budgets: list[int],
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
):
    if not prefix_token_ids_list:
        return []
    max_remaining_budget = max(remaining_budgets)
    if max_remaining_budget <= 0:
        return [[] for _ in prefix_token_ids_list]

    prompts = [{"prompt_token_ids": prompt_token_ids} for prompt_token_ids in prefix_token_ids_list]
    sampling_params = SamplingParams(
        n=1,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k if top_k > 0 else -1,
        max_tokens=max_remaining_budget,
        skip_special_tokens=False,
        seed=seed,
    )
    outputs = llm.generate(prompts, sampling_params)
    suffix_lists = []
    for output, budget in zip(outputs, remaining_budgets):
        token_ids = output.outputs[0].token_ids[:budget]
        suffix_lists.append(token_ids)
    return suffix_lists


def run_rollout_phase(args, rank: int, world_size: int, output_dir: Path, partial_dir: Path, tokenizer):
    dataset = pd.read_parquet(args.data_path)
    example_indices = list(range(rank, min(len(dataset), args.max_examples), world_size))
    scanned_count_local = 0
    successful_rollouts = []
    target_successes_local = math.ceil(args.target_successes / max(world_size, 1))

    llm = init_vllm_engine(args)
    progress = tqdm(example_indices, total=len(example_indices), disable=rank != 0, desc=f"Rank {rank} rollout")
    try:
        for example_index in progress:
            if scanned_count_local >= math.ceil(args.max_examples / max(world_size, 1)):
                break
            if len(successful_rollouts) >= target_successes_local:
                break
            scanned_count_local += 1

            row = dataset.iloc[example_index]
            prompt_messages = row[args.prompt_key]
            reward_model = row[args.reward_model_key]
            data_source = row[args.data_source_key]
            extra_info = row.get(args.extra_info_key, {}) if args.extra_info_key in row else {}
            ground_truth = reward_model["ground_truth"]

            prompt_ids = build_prompt_ids(tokenizer, prompt_messages, enable_thinking=args.enable_thinking)
            prompt_token_ids = prompt_ids[0].tolist()
            rollout_results = generate_rollouts_vllm(
                llm=llm,
                prompt_token_ids_list=[prompt_token_ids],
                rollouts_per_prompt=args.rollouts_per_prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.t1,
                top_k=args.top_k,
                top_p=args.top_p,
                seed=args.seed + example_index,
            )
            response_texts, response_token_ids_list = rollout_results[0]
            for rollout_index, (response_text, response_token_ids) in enumerate(zip(response_texts, response_token_ids_list)):
                original_score = compute_score_float(
                    data_source=data_source,
                    solution_str=response_text,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                )
                if original_score <= 0:
                    continue
                successful_rollouts.append(
                    {
                        "example_index": int(example_index),
                        "rollout_index": int(rollout_index),
                        "prompt_ids": prompt_ids[0].tolist(),
                        "prompt_token_ids": prompt_token_ids,
                        "response_token_ids": response_token_ids,
                        "original_score": float(original_score),
                        "data_source": str(data_source),
                        "ground_truth": ground_truth,
                        "extra_info": extra_info,
                    }
                )
                if len(successful_rollouts) >= target_successes_local:
                    break
    finally:
        cleanup_vllm_engine(llm)

    write_jsonl(get_successes_path(partial_dir, rank), successful_rollouts)
    get_phase1_meta_path(partial_dir, rank).write_text(
        json.dumps(
            {
                "rank": rank,
                "world_size": world_size,
                "scanned_examples_local": scanned_count_local,
                "successful_exploration_trajectories_local": len(successful_rollouts),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def run_replay_phase(args, rank: int, partial_dir: Path, tokenizer):
    successful_rollouts = read_jsonl(get_successes_path(partial_dir, rank))
    intervention_jobs = []
    if not successful_rollouts:
        write_jsonl(get_jobs_path(partial_dir, rank), intervention_jobs)
        return

    intervention_model_device = args.intervention_model_device if args.intervention_model_device != "auto" else "auto"
    intervention_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=get_torch_dtype(args.dtype),
        device_map=intervention_model_device,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
    )
    intervention_model.eval()

    replay_progress = tqdm(
        successful_rollouts,
        total=len(successful_rollouts),
        disable=rank != 0,
        desc=f"Rank {rank} replay",
    )
    try:
        with GpuKeepalive(args):
            for success_item in replay_progress:
                prompt_ids = torch.tensor([success_item["prompt_ids"]], dtype=torch.long)
                response_logits_t1 = replay_response_logits(
                    intervention_model,
                    prompt_ids=prompt_ids,
                    response_token_ids=success_item["response_token_ids"],
                )
                if response_logits_t1.numel() == 0:
                    continue
                intervention_jobs.extend(
                    build_intervention_jobs_for_success(
                        args=args,
                        tokenizer=tokenizer,
                        success_item=success_item,
                        response_logits_t1=response_logits_t1,
                    )
                )
    finally:
        cleanup_torch_model(intervention_model)

    write_jsonl(get_jobs_path(partial_dir, rank), intervention_jobs)


def build_sequential_job_for_state(
    args,
    tokenizer,
    state: dict,
    response_logits_t1: torch.Tensor,
    round_idx: int,
):
    response_token_ids = state["response_token_ids"]
    original_length = len(response_token_ids)
    if original_length == 0:
        return None

    js_values, entropy_values, margin_values = compute_js_entropy_margin(list(response_logits_t1), t0=args.t0, t1=args.t1)
    candidate_positions = list(range(original_length))
    if not args.allow_eos_replacement and tokenizer.eos_token_id is not None:
        candidate_positions = [idx for idx in candidate_positions if response_token_ids[idx] != tokenizer.eos_token_id]
    masked_positions = build_masked_positions(state.get("round_positions", []), original_length, args.sequential_gap)
    candidate_positions = [idx for idx in candidate_positions if idx not in masked_positions]
    if not candidate_positions:
        return None

    per_state_seed = (
        args.seed
        + int(state["example_index"]) * 9973
        + int(state["rollout_index"]) * 131
        + int(state["repeat_index"]) * 17
        + round_idx * 100003
    )
    per_state_rng = random.Random(per_state_seed)
    anchor_spans = select_spans(
        selector="js",
        js_values=js_values,
        entropy_values=entropy_values,
        margin_values=margin_values,
        valid_positions=candidate_positions,
        top_k=1,
        rng=per_state_rng,
        granularity=args.intervention_granularity,
        block_size=args.block_size,
    )
    if not anchor_spans:
        return None

    selector = state["selector"]
    if selector == "matched_random":
        chosen_spans = select_spans(
            selector="matched_random",
            js_values=js_values,
            entropy_values=entropy_values,
            margin_values=margin_values,
            valid_positions=candidate_positions,
            top_k=1,
            rng=random.Random(per_state_seed + 19),
            granularity=args.intervention_granularity,
            block_size=args.block_size,
            anchor_spans=anchor_spans,
        )
    elif selector == "js":
        chosen_spans = anchor_spans
    else:
        chosen_spans = select_spans(
            selector=selector,
            js_values=js_values,
            entropy_values=entropy_values,
            margin_values=margin_values,
            valid_positions=candidate_positions,
            top_k=1,
            rng=per_state_rng,
            granularity=args.intervention_granularity,
            block_size=args.block_size,
        )
    if not chosen_spans:
        return None

    prompt_ids = torch.tensor([state["prompt_ids"]], dtype=torch.long)
    job = build_intervention_job(
        example_index=state["example_index"],
        rollout_index=state["rollout_index"],
        selector_name=state["selector"],
        repeat_index=state["repeat_index"],
        positions=chosen_spans[0],
        response_token_ids=response_token_ids,
        raw_logits=list(response_logits_t1),
        prompt_ids=prompt_ids,
        tokenizer=tokenizer,
        args=args,
        original_score=state["original_score"],
        data_source=state["data_source"],
        ground_truth=state["ground_truth"],
        extra_info=state["extra_info"],
        js_values=js_values,
        entropy_values=entropy_values,
        margin_values=margin_values,
    )
    if job is None:
        return None
    job["state_key"] = state["state_key"]
    job["round_idx"] = int(round_idx)
    job["current_score_before_round"] = float(state["current_score"])
    job["round_positions_history"] = state.get("round_positions", [])
    job["round_position_ratios_history"] = state.get("round_position_ratios", [])
    job["round_js_values_history"] = state.get("round_js_values", [])
    job["round_entropy_values_history"] = state.get("round_entropy_values", [])
    job["round_margin_values_history"] = state.get("round_margin_values", [])
    job["round_original_tokens_history"] = state.get("round_original_tokens", [])
    job["round_replacement_tokens_history"] = state.get("round_replacement_tokens", [])
    job["round_original_token_texts_history"] = state.get("round_original_token_texts", [])
    job["round_replacement_token_texts_history"] = state.get("round_replacement_token_texts", [])
    job["total_replaced_tokens_before_round"] = int(state.get("total_replaced_tokens", 0))
    return job


def run_sequential_replay_phase(args, rank: int, partial_dir: Path, tokenizer, round_idx: int):
    current_states_path = get_current_states_path(partial_dir, rank)
    if current_states_path.exists():
        current_states = read_jsonl(current_states_path)
    else:
        successful_rollouts = read_jsonl(get_successes_path(partial_dir, rank))
        current_states = init_sequential_states(args, successful_rollouts)
        write_jsonl(current_states_path, current_states)

    seq_jobs = []
    if not current_states:
        write_jsonl(get_seq_jobs_path(partial_dir, rank, round_idx), seq_jobs)
        return

    intervention_model_device = args.intervention_model_device if args.intervention_model_device != "auto" else "auto"
    intervention_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=get_torch_dtype(args.dtype),
        device_map=intervention_model_device,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
    )
    intervention_model.eval()

    replay_progress = tqdm(
        current_states,
        total=len(current_states),
        disable=rank != 0,
        desc=f"Rank {rank} seq-replay r{round_idx}",
    )
    try:
        with GpuKeepalive(args):
            for state in replay_progress:
                prompt_ids = torch.tensor([state["prompt_ids"]], dtype=torch.long)
                response_logits_t1 = replay_response_logits(
                    intervention_model,
                    prompt_ids=prompt_ids,
                    response_token_ids=state["response_token_ids"],
                )
                if response_logits_t1.numel() == 0:
                    continue
                job = build_sequential_job_for_state(
                    args=args,
                    tokenizer=tokenizer,
                    state=state,
                    response_logits_t1=response_logits_t1,
                    round_idx=round_idx,
                )
                if job is not None:
                    seq_jobs.append(job)
    finally:
        cleanup_torch_model(intervention_model)

    write_jsonl(get_seq_jobs_path(partial_dir, rank, round_idx), seq_jobs)


def run_continue_phase(args, rank: int, world_size: int, output_dir: Path, partial_dir: Path, tokenizer):
    intervention_jobs = read_jsonl(get_jobs_path(partial_dir, rank))
    jsonl_path = partial_dir / f"rank_{rank:03d}.jsonl"
    phase1_meta_path = get_phase1_meta_path(partial_dir, rank)
    phase1_meta = json.loads(phase1_meta_path.read_text(encoding="utf-8")) if phase1_meta_path.exists() else {}
    scanned_count_local = int(phase1_meta.get("scanned_examples_local", 0))
    success_count_local = int(phase1_meta.get("successful_exploration_trajectories_local", 0))

    if intervention_jobs:
        llm = init_vllm_engine(args)
        continuation_progress = range(0, len(intervention_jobs), args.analysis_batch_size)
        try:
            with jsonl_path.open("w", encoding="utf-8") as fout:
                for start in continuation_progress:
                    batch_jobs = intervention_jobs[start : start + args.analysis_batch_size]
                    prefix_token_ids_list = [job["prefix_ids"] for job in batch_jobs]
                    remaining_budgets = [job["remaining_budget"] for job in batch_jobs]
                    suffix_batches = greedy_continue_vllm_batch(
                        llm=llm,
                        prefix_token_ids_list=prefix_token_ids_list,
                        remaining_budgets=remaining_budgets,
                    )
                    for job, suffix_ids in zip(batch_jobs, suffix_batches):
                        intervened_response_ids = job["prefix_response_ids"] + suffix_ids
                        intervened_text = tokenizer.decode(intervened_response_ids, skip_special_tokens=True)
                        intervened_score = compute_score_float(
                            data_source=job["data_source"],
                            solution_str=intervened_text,
                            ground_truth=job["ground_truth"],
                            extra_info=job["extra_info"],
                        )
                        result = InterventionResult(
                            example_index=job["example_index"],
                            selector=job["selector"],
                            repeat_index=job["repeat_index"],
                            rollout_index=job["rollout_index"],
                            position=job["position"],
                            original_length=job["original_length"],
                            original_score=job["original_score"],
                            intervened_score=intervened_score,
                            score_delta=intervened_score - job["original_score"],
                            reward_drop=job["original_score"] - intervened_score,
                            flipped=bool(job["original_score"] > 0 and intervened_score <= 0),
                            original_token=job["original_token"],
                            replacement_token=job["replacement_token"],
                            replacement_token_text=job["replacement_token_text"],
                            original_token_text=job["original_token_text"],
                            position_ratio=job["position_ratio"],
                            js_value=job["js_value"],
                            entropy_value=job["entropy_value"],
                            margin_value=job["margin_value"],
                            data_source=job["data_source"],
                            positions=job["positions"],
                            num_replaced_tokens=job["num_replaced_tokens"],
                            intervention_granularity=job["intervention_granularity"],
                            original_tokens=job["original_tokens"],
                            replacement_tokens=job["replacement_tokens"],
                            replacement_token_texts=job["replacement_token_texts"],
                            original_token_texts=job["original_token_texts"],
                        )
                        fout.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
        finally:
            cleanup_vllm_engine(llm)
    else:
        jsonl_path.write_text("", encoding="utf-8")

    write_rank_summary(
        partial_dir=partial_dir,
        rank=rank,
        world_size=world_size,
        scanned_count_local=scanned_count_local,
        success_count_local=success_count_local,
    )

    finalize_outputs(
        args=args,
        output_dir=output_dir,
        partial_dir=partial_dir,
        rank=rank,
        world_size=world_size,
        scanned_count_local=scanned_count_local,
    )


def run_sequential_continue_phase(args, rank: int, partial_dir: Path, tokenizer, round_idx: int):
    current_states_path = get_current_states_path(partial_dir, rank)
    current_states = read_jsonl(current_states_path)
    if not current_states:
        return

    jobs_path = get_seq_jobs_path(partial_dir, rank, round_idx)
    seq_jobs = read_jsonl(jobs_path)
    if not seq_jobs:
        return

    current_state_map = {state["state_key"]: state for state in current_states}
    llm = init_vllm_engine(args)
    continuation_progress = tqdm(
        range(0, len(seq_jobs), args.analysis_batch_size),
        total=math.ceil(len(seq_jobs) / max(args.analysis_batch_size, 1)),
        disable=rank != 0,
        desc=f"Rank {rank} seq-continue r{round_idx}",
    )
    try:
        for start in continuation_progress:
            batch_jobs = seq_jobs[start : start + args.analysis_batch_size]
            prefix_token_ids_list = [job["prefix_ids"] for job in batch_jobs]
            remaining_budgets = [job["remaining_budget"] for job in batch_jobs]
            suffix_batches = sample_continue_vllm_batch(
                llm=llm,
                prefix_token_ids_list=prefix_token_ids_list,
                remaining_budgets=remaining_budgets,
                temperature=args.t1,
                top_p=args.top_p,
                top_k=args.top_k,
                seed=args.seed + round_idx * 1000003 + start,
            )
            for job, suffix_ids in zip(batch_jobs, suffix_batches):
                updated_response_ids = job["prefix_response_ids"] + suffix_ids
                updated_text = tokenizer.decode(updated_response_ids, skip_special_tokens=True)
                updated_score = compute_score_float(
                    data_source=job["data_source"],
                    solution_str=updated_text,
                    ground_truth=job["ground_truth"],
                    extra_info=job["extra_info"],
                )
                state = current_state_map[job["state_key"]]
                state["response_token_ids"] = updated_response_ids
                state["current_score"] = float(updated_score)
                state["round_positions"] = list(job["round_positions_history"]) + [job["positions"]]
                state["round_position_ratios"] = list(job["round_position_ratios_history"]) + [job["position_ratio"]]
                state["round_js_values"] = list(job["round_js_values_history"]) + [job["js_value"]]
                state["round_entropy_values"] = list(job["round_entropy_values_history"]) + [job["entropy_value"]]
                state["round_margin_values"] = list(job["round_margin_values_history"]) + [job["margin_value"]]
                state["round_original_tokens"] = list(job["round_original_tokens_history"]) + [job["original_tokens"]]
                state["round_replacement_tokens"] = list(job["round_replacement_tokens_history"]) + [job["replacement_tokens"]]
                state["round_original_token_texts"] = list(job["round_original_token_texts_history"]) + [job["original_token_texts"]]
                state["round_replacement_token_texts"] = list(job["round_replacement_token_texts_history"]) + [job["replacement_token_texts"]]
                state["total_replaced_tokens"] = int(job["total_replaced_tokens_before_round"]) + int(job["num_replaced_tokens"])
    finally:
        cleanup_vllm_engine(llm)

    write_jsonl(current_states_path, list(current_state_map.values()))


def build_sequential_result_from_state(state: dict) -> InterventionResult:
    round_positions = state.get("round_positions", [])
    flat_positions = [int(pos) for span in round_positions for pos in span]
    round_original_tokens = state.get("round_original_tokens", [])
    round_replacement_tokens = state.get("round_replacement_tokens", [])
    flat_original_tokens = [int(token) for span in round_original_tokens for token in span]
    flat_replacement_tokens = [int(token) for span in round_replacement_tokens for token in span]
    flat_original_texts = [text for span in state.get("round_original_token_texts", []) for text in span]
    flat_replacement_texts = [text for span in state.get("round_replacement_token_texts", []) for text in span]
    num_rounds = len(round_positions)
    first_position = flat_positions[0] if flat_positions else 0
    first_original_token = flat_original_tokens[0] if flat_original_tokens else -1
    first_replacement_token = flat_replacement_tokens[0] if flat_replacement_tokens else -1
    first_original_text = flat_original_texts[0] if flat_original_texts else ""
    first_replacement_text = flat_replacement_texts[0] if flat_replacement_texts else ""
    mean_position_ratio = sum(state.get("round_position_ratios", [])) / max(num_rounds, 1) if num_rounds > 0 else 0.0
    mean_js = sum(state.get("round_js_values", [])) / max(num_rounds, 1) if num_rounds > 0 else 0.0
    mean_entropy = sum(state.get("round_entropy_values", [])) / max(num_rounds, 1) if num_rounds > 0 else 0.0
    mean_margin = sum(state.get("round_margin_values", [])) / max(num_rounds, 1) if num_rounds > 0 else 0.0
    return InterventionResult(
        example_index=int(state["example_index"]),
        selector=str(state["selector"]),
        repeat_index=int(state["repeat_index"]),
        rollout_index=int(state["rollout_index"]),
        position=int(first_position),
        original_length=int(state.get("original_length", len(state["response_token_ids"]))),
        original_score=float(state["original_score"]),
        intervened_score=float(state["current_score"]),
        score_delta=float(state["current_score"]) - float(state["original_score"]),
        reward_drop=float(state["original_score"]) - float(state["current_score"]),
        flipped=bool(float(state["original_score"]) > 0 and float(state["current_score"]) <= 0),
        original_token=int(first_original_token),
        replacement_token=int(first_replacement_token),
        replacement_token_text=str(first_replacement_text),
        original_token_text=str(first_original_text),
        position_ratio=float(mean_position_ratio),
        js_value=float(mean_js),
        entropy_value=float(mean_entropy),
        margin_value=float(mean_margin),
        data_source=str(state["data_source"]),
        positions=flat_positions,
        num_replaced_tokens=int(state.get("total_replaced_tokens", 0)),
        intervention_granularity="sequential",
        original_tokens=flat_original_tokens,
        replacement_tokens=flat_replacement_tokens,
        replacement_token_texts=flat_replacement_texts,
        original_token_texts=flat_original_texts,
        num_intervention_rounds=int(num_rounds),
        round_positions=round_positions,
    )


def run_sequential_finalize_phase(args, rank: int, world_size: int, output_dir: Path, partial_dir: Path):
    current_states_path = get_current_states_path(partial_dir, rank)
    current_states = read_jsonl(current_states_path)
    jsonl_path = partial_dir / f"rank_{rank:03d}.jsonl"
    phase1_meta_path = get_phase1_meta_path(partial_dir, rank)
    phase1_meta = json.loads(phase1_meta_path.read_text(encoding="utf-8")) if phase1_meta_path.exists() else {}
    scanned_count_local = int(phase1_meta.get("scanned_examples_local", 0))
    success_count_local = int(phase1_meta.get("successful_exploration_trajectories_local", 0))

    with jsonl_path.open("w", encoding="utf-8") as fout:
        for state in current_states:
            result = build_sequential_result_from_state(state)
            fout.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

    write_rank_summary(
        partial_dir=partial_dir,
        rank=rank,
        world_size=world_size,
        scanned_count_local=scanned_count_local,
        success_count_local=success_count_local,
    )
    finalize_outputs(
        args=args,
        output_dir=output_dir,
        partial_dir=partial_dir,
        rank=rank,
        world_size=world_size,
        scanned_count_local=scanned_count_local,
    )


def write_rank_summary(partial_dir: Path, rank: int, world_size: int, scanned_count_local: int, success_count_local: int):
    summary_local = {
        "rank": rank,
        "world_size": world_size,
        "scanned_examples_local": scanned_count_local,
        "successful_exploration_trajectories_local": success_count_local,
    }
    (partial_dir / f"summary_rank_{rank:03d}.json").write_text(
        json.dumps(summary_local, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def finalize_outputs(args, output_dir: Path, partial_dir: Path, rank: int, world_size: int, scanned_count_local: int):
    if rank != 0 or args.skip_finalize:
        return
    aggregated_results = aggregate_outputs(output_dir)
    selectors = [selector.strip() for selector in args.selectors.split(",") if selector.strip()]
    summary = {
        "model_path": args.model_path,
        "data_path": args.data_path,
        "scanned_examples_requested": args.max_examples,
        "target_successes_requested": args.target_successes,
        "world_size": world_size,
        "rollouts_per_prompt": args.rollouts_per_prompt,
        "matched_random_repeats": args.matched_random_repeats,
        "intervention_mode": args.intervention_mode,
        "intervention_granularity": args.intervention_granularity,
        "block_size": args.block_size,
        "sequential_rounds": args.sequential_rounds,
        "selectors": selectors,
        "summary": summarize_results(aggregated_results),
    }
    rank_summaries = []
    total_scanned = 0
    total_successes = 0
    for summary_path in sorted(partial_dir.glob("summary_rank_*.json")):
        rank_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rank_summaries.append(rank_summary)
        total_scanned += rank_summary["scanned_examples_local"]
        total_successes += rank_summary["successful_exploration_trajectories_local"]
    if rank_summaries:
        summary["rank_summaries"] = rank_summaries
        summary["scanned_examples"] = total_scanned
        summary["successful_exploration_trajectories"] = total_successes
    else:
        summary["scanned_examples"] = scanned_count_local
        summary["successful_exploration_trajectories"] = 0
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.intervention_mode == "sequential" and args.positions_per_sample != 1:
        raise ValueError("Sequential intervention currently requires --positions-per-sample=1.")

    rank, world_size, use_dist = get_rank_world_size(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_dir = output_dir / "partials"
    partial_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    config_path = output_dir / "config.json"
    target_successes_local = math.ceil(args.target_successes / max(world_size, 1))
    if rank == 0:
        config_payload = {
            **vars(args),
            "world_size": world_size,
            "target_successes_per_rank": target_successes_local,
        }
        config_path.write_text(json.dumps(config_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.phase == "rollout":
        run_rollout_phase(args, rank=rank, world_size=world_size, output_dir=output_dir, partial_dir=partial_dir, tokenizer=tokenizer)
        return
    if args.phase == "replay":
        run_replay_phase(args, rank=rank, partial_dir=partial_dir, tokenizer=tokenizer)
        return
    if args.phase == "continue":
        run_continue_phase(
            args,
            rank=rank,
            world_size=world_size,
            output_dir=output_dir,
            partial_dir=partial_dir,
            tokenizer=tokenizer,
        )
        return
    if args.phase == "seq_replay":
        if args.round_idx is None:
            raise ValueError("--round-idx is required for seq_replay.")
        run_sequential_replay_phase(args, rank=rank, partial_dir=partial_dir, tokenizer=tokenizer, round_idx=args.round_idx)
        return
    if args.phase == "seq_continue":
        if args.round_idx is None:
            raise ValueError("--round-idx is required for seq_continue.")
        run_sequential_continue_phase(args, rank=rank, partial_dir=partial_dir, tokenizer=tokenizer, round_idx=args.round_idx)
        return
    if args.phase == "seq_finalize":
        run_sequential_finalize_phase(
            args,
            rank=rank,
            world_size=world_size,
            output_dir=output_dir,
            partial_dir=partial_dir,
        )
        return

    if args.phase != "full":
        raise ValueError(f"Unsupported phase: {args.phase}")

    if args.intervention_mode == "one_shot":
        skip_rollout = bool(
            args.resume
            and get_successes_path(partial_dir, rank).exists()
            and get_phase1_meta_path(partial_dir, rank).exists()
        )
        skip_replay = bool(args.resume and get_jobs_path(partial_dir, rank).exists())
        if not skip_rollout:
            if rank == 0:
                print("Starting isolated phase: rollout")
            run_phase_subprocess("rollout")
        if not skip_replay:
            if rank == 0:
                print("Starting isolated phase: replay")
            run_phase_subprocess("replay")
        if rank == 0:
            print("Starting isolated phase: continue")
        run_phase_subprocess("continue")
    else:
        skip_rollout = bool(
            args.resume
            and get_successes_path(partial_dir, rank).exists()
            and get_phase1_meta_path(partial_dir, rank).exists()
        )
        start_round = infer_completed_rounds(partial_dir, rank) if args.resume else 0
        if not skip_rollout:
            if rank == 0:
                print("Starting isolated phase: rollout")
            run_phase_subprocess("rollout")
        for round_idx in range(start_round, args.sequential_rounds):
            if rank == 0:
                print(f"Starting isolated sequential phases: round={round_idx}")
            run_phase_subprocess("seq_replay", round_idx=round_idx)
            run_phase_subprocess("seq_continue", round_idx=round_idx)
        if rank == 0:
            print("Starting isolated phase: seq_finalize")
        run_phase_subprocess("seq_finalize")


if __name__ == "__main__":
    main()
