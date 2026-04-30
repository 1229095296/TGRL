#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from verl.utils.reward_score import default_compute_score


def compute_score_float(data_source, solution_str, ground_truth, extra_info=None) -> float:
    res = default_compute_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
    )
    if isinstance(res, dict):
        return float(res.get("score", res.get("reward", 0.0)))
    return float(res)


def parse_args():
    parser = argparse.ArgumentParser(description="JS token intervention analysis for math checkpoints.")
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
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard-rank", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--skip-finalize", action="store_true")
    parser.add_argument("--use-torch-compile", action="store_true", help="Enable torch.compile for additional speedup")
    return parser.parse_args()


@dataclass
class InterventionResult:
    example_index: int
    selector: str
    repeat_index: int
    rollout_index: int
    position: int
    original_length: int
    original_score: float
    intervened_score: float
    score_delta: float
    reward_drop: float
    flipped: bool
    original_token: int
    replacement_token: int
    replacement_token_text: str
    original_token_text: str
    position_ratio: float
    js_value: float
    entropy_value: float
    margin_value: float
    data_source: str
    positions: list[int] = field(default_factory=list)
    num_replaced_tokens: int = 1
    intervention_granularity: str = "single"
    original_tokens: list[int] = field(default_factory=list)
    replacement_tokens: list[int] = field(default_factory=list)
    replacement_token_texts: list[str] = field(default_factory=list)
    original_token_texts: list[str] = field(default_factory=list)
    num_intervention_rounds: int = 1
    round_positions: list[list[int]] = field(default_factory=list)


def get_torch_dtype(dtype_str: str):
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_str]


def get_input_device(model):
    hf_device_map = getattr(model, "hf_device_map", None)
    if hf_device_map:
        cuda_devices = [v for v in hf_device_map.values() if isinstance(v, str) and v.startswith("cuda")]
        if cuda_devices:
            cuda_devices = sorted(set(cuda_devices), key=lambda x: int(x.split(":")[1]))
            return torch.device(cuda_devices[0])
    return next(model.parameters()).device


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


def apply_top_k_top_p(logits: torch.Tensor, top_k: int = -1, top_p: float = 1.0) -> torch.Tensor:
    filtered = logits.clone()

    if top_k is not None and top_k > 0 and top_k < filtered.shape[-1]:
        threshold = torch.topk(filtered, top_k, dim=-1).values[..., -1, None]
        filtered = filtered.masked_fill(filtered < threshold, float("-inf"))

    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_mask = cumulative_probs > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        filtered_sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))
        filtered = torch.full_like(filtered, float("-inf"))
        filtered.scatter_(dim=-1, index=sorted_indices, src=filtered_sorted_logits)

    return filtered


def sample_tokens_batch(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    generator: torch.Generator,
) -> torch.Tensor:
    tempered = logits / max(temperature, 1e-6)
    filtered = apply_top_k_top_p(tempered, top_k=top_k, top_p=top_p)
    probs = torch.softmax(filtered, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


def build_prompt_ids(tokenizer, prompt_messages, enable_thinking: bool) -> torch.Tensor:
    apply_kwargs = {"enable_thinking": True} if enable_thinking else {}
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        **apply_kwargs,
    )
    return prompt_ids


@torch.inference_mode()
def generate_responses_for_prompt(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    rollouts_per_prompt: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    seed: int,
):
    device = get_input_device(model)
    eos_token_ids = model.generation_config.eos_token_id
    if eos_token_ids is None:
        eos_token_ids = []
    elif isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]
    eos_token_ids = set(eos_token_ids)

    input_ids = prompt_ids.to(device)
    attention_mask = torch.ones_like(input_ids, device=device)
    input_ids = input_ids.expand(rollouts_per_prompt, -1).contiguous()
    attention_mask = attention_mask.expand(rollouts_per_prompt, -1).contiguous()
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    generated_tokens = [[] for _ in range(rollouts_per_prompt)]
    raw_logits = [[] for _ in range(rollouts_per_prompt)]
    past_key_values = None
    current_input_ids = input_ids
    current_attention_mask = attention_mask
    finished = torch.zeros(rollouts_per_prompt, dtype=torch.bool, device=device)
    eos_fallback = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    for _ in range(max_new_tokens):
        outputs = model(
            input_ids=current_input_ids,
            attention_mask=current_attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        logits = outputs.logits[:, -1, :]
        for batch_idx in range(rollouts_per_prompt):
            if not finished[batch_idx]:
                raw_logits[batch_idx].append(logits[batch_idx].float().cpu())

        next_tokens = sample_tokens_batch(
            logits=logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            generator=generator,
        )
        next_tokens = torch.where(finished, torch.full_like(next_tokens, eos_fallback), next_tokens)

        for batch_idx in range(rollouts_per_prompt):
            if finished[batch_idx]:
                continue
            token_id = int(next_tokens[batch_idx].item())
            generated_tokens[batch_idx].append(token_id)
            if token_id in eos_token_ids:
                finished[batch_idx] = True

        past_key_values = outputs.past_key_values

        if torch.all(finished):
            break

        current_input_ids = next_tokens.unsqueeze(-1)
        current_attention_mask = torch.cat(
            [
                current_attention_mask,
                torch.ones((rollouts_per_prompt, 1), dtype=current_attention_mask.dtype, device=device),
            ],
            dim=-1,
        )

    return generated_tokens, raw_logits


@torch.inference_mode()
def greedy_continue(
    model,
    prefix_ids: torch.Tensor,
    max_new_tokens: int,
):
    device = get_input_device(model)
    eos_token_ids = model.generation_config.eos_token_id
    if eos_token_ids is None:
        eos_token_ids = []
    elif isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]
    eos_token_ids = set(eos_token_ids)

    input_ids = prefix_ids.to(device)
    attention_mask = torch.ones_like(input_ids, device=device)
    past_key_values = None
    current_input_ids = input_ids
    current_attention_mask = attention_mask
    generated = []

    for _ in range(max_new_tokens):
        outputs = model(
            input_ids=current_input_ids,
            attention_mask=current_attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        token_id = int(next_token.item())
        generated.append(token_id)
        past_key_values = outputs.past_key_values

        if token_id in eos_token_ids:
            break

        current_input_ids = next_token
        current_attention_mask = torch.cat(
            [current_attention_mask, torch.ones((1, 1), dtype=current_attention_mask.dtype, device=device)], dim=-1
        )

    return generated


@torch.inference_mode()
def greedy_continue_batch(
    model,
    tokenizer,
    prefix_id_list: list[torch.Tensor],
    max_new_tokens: int,
):
    if not prefix_id_list:
        return []
    if max_new_tokens <= 0:
        return [[] for _ in prefix_id_list]

    device = get_input_device(model)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    lengths = [prefix_ids.numel() for prefix_ids in prefix_id_list]
    max_len = max(lengths)
    batch_size = len(prefix_id_list)
    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)

    for idx, prefix_ids in enumerate(prefix_id_list):
        prefix_ids = prefix_ids.to(device)
        input_ids[idx, -lengths[idx] :] = prefix_ids
        attention_mask[idx, -lengths[idx] :] = 1

    sequences = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=pad_token_id,
        eos_token_id=model.generation_config.eos_token_id,
        use_cache=True,
    )
    generated_lists = []
    for idx in range(batch_size):
        tail = sequences[idx, max_len:].tolist()
        generated_lists.append(tail)
    return generated_lists


def compute_js_entropy_margin(raw_logits: list[torch.Tensor], t0: float, t1: float):
    js_values = []
    entropy_values = []
    margin_values = []
    for logits in raw_logits:
        logits = logits.float()
        probs_t0 = torch.softmax(logits / t0, dim=-1)
        probs_t1 = torch.softmax(logits / t1, dim=-1)
        mixture = 0.5 * (probs_t0 + probs_t1)
        js = 0.5 * (
            torch.sum(probs_t0 * (torch.log(probs_t0 + 1e-8) - torch.log(mixture + 1e-8)))
            + torch.sum(probs_t1 * (torch.log(probs_t1 + 1e-8) - torch.log(mixture + 1e-8)))
        )
        entropy = float(-(probs_t1 * torch.log(probs_t1 + 1e-8)).sum().item())
        top2_probs = torch.topk(probs_t1, k=2).values
        margin = float((top2_probs[0] - top2_probs[1]).item())
        js_values.append(float(js.item()))
        entropy_values.append(entropy)
        margin_values.append(margin)
    return js_values, entropy_values, margin_values


def choose_replacement_token(
    logits: torch.Tensor,
    original_token_id: int,
    t0: float,
    strategy: str,
    tokenizer,
    allow_eos_replacement: bool,
) -> int | None:
    disallowed = {original_token_id}
    if tokenizer.pad_token_id is not None:
        disallowed.add(tokenizer.pad_token_id)
    if not allow_eos_replacement and tokenizer.eos_token_id is not None:
        disallowed.add(tokenizer.eos_token_id)

    if strategy == "t0_top1_else_t0_top2":
        scores = logits / t0
    elif strategy == "second_best_current":
        scores = logits
    else:
        raise ValueError(f"Unknown replacement strategy: {strategy}")

    candidate_ids = torch.argsort(scores, descending=True).tolist()
    for token_id in candidate_ids:
        if token_id not in disallowed:
            return int(token_id)
    return None


def compute_bucket(position: int, total_positions: int, num_buckets: int = 4) -> int:
    if total_positions <= 1:
        return 0
    return min(num_buckets - 1, int(num_buckets * position / total_positions))


def spans_overlap(span_a: list[int], span_b: list[int]) -> bool:
    return not (span_a[-1] < span_b[0] or span_b[-1] < span_a[0])


def build_candidate_spans(valid_positions: list[int], total_positions: int, granularity: str, block_size: int) -> list[list[int]]:
    if not valid_positions:
        return []
    if granularity == "single":
        return [[pos] for pos in valid_positions]
    if granularity != "block":
        raise ValueError(f"Unknown intervention granularity: {granularity}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    valid_set = set(valid_positions)
    spans = []
    for start in range(max(total_positions - block_size + 1, 0)):
        span = list(range(start, start + block_size))
        if all(pos in valid_set for pos in span):
            spans.append(span)
    return spans


def select_spans(
    selector: str,
    js_values: list[float],
    entropy_values: list[float],
    margin_values: list[float],
    valid_positions: list[int],
    top_k: int,
    rng: random.Random,
    granularity: str = "single",
    block_size: int = 5,
    anchor_spans: list[list[int]] | None = None,
):
    total = len(js_values)
    candidate_spans = build_candidate_spans(valid_positions, total_positions=total, granularity=granularity, block_size=block_size)
    if not candidate_spans:
        return []

    def span_score(span: list[int], values: list[float]) -> float:
        return sum(values[pos] for pos in span) / len(span)

    def choose_non_overlapping(ranked_spans: list[list[int]]) -> list[list[int]]:
        chosen = []
        for span in ranked_spans:
            if any(spans_overlap(span, existing) for existing in chosen):
                continue
            chosen.append(span)
            if len(chosen) >= top_k:
                break
        return chosen

    if selector == "js":
        ranked = sorted(candidate_spans, key=lambda span: span_score(span, js_values), reverse=True)
        return choose_non_overlapping(ranked)
    if selector == "entropy":
        ranked = sorted(candidate_spans, key=lambda span: span_score(span, entropy_values), reverse=True)
        return choose_non_overlapping(ranked)
    if selector == "low_margin":
        ranked = sorted(candidate_spans, key=lambda span: span_score(span, margin_values))
        return choose_non_overlapping(ranked)
    if selector == "matched_random":
        anchor_spans = anchor_spans or []
        span_to_start = {tuple(span): span[0] for span in candidate_spans}
        bucket_to_candidates = defaultdict(list)
        for span in candidate_spans:
            bucket = compute_bucket(span[0], total)
            bucket_to_candidates[bucket].append(span)
        chosen = []
        for anchor_span in anchor_spans[:top_k]:
            bucket = compute_bucket(anchor_span[0], total)
            pool = [
                span
                for span in bucket_to_candidates[bucket]
                if not any(spans_overlap(span, existing) for existing in chosen)
            ]
            if not pool:
                pool = [span for span in candidate_spans if not any(spans_overlap(span, existing) for existing in chosen)]
            if not pool:
                break
            picked = rng.choice(pool)
            chosen.append(picked)
        return chosen

    raise ValueError(f"Unknown selector: {selector}")


def select_positions(
    selector: str,
    js_values: list[float],
    entropy_values: list[float],
    margin_values: list[float],
    top_k: int,
    rng: random.Random,
    anchor_positions: list[int] | None = None,
):
    total = len(js_values)
    candidate_positions = list(range(total))
    if total == 0:
        return []

    if selector == "js":
        ranked = sorted(candidate_positions, key=lambda idx: js_values[idx], reverse=True)
        return ranked[:top_k]
    if selector == "entropy":
        ranked = sorted(candidate_positions, key=lambda idx: entropy_values[idx], reverse=True)
        return ranked[:top_k]
    if selector == "low_margin":
        ranked = sorted(candidate_positions, key=lambda idx: margin_values[idx])
        return ranked[:top_k]
    if selector == "matched_random":
        if not anchor_positions:
            anchor_positions = []
        bucket_to_candidates = defaultdict(list)
        for pos in candidate_positions:
            bucket_to_candidates[compute_bucket(pos, total)].append(pos)
        chosen = []
        used = set()
        for anchor in anchor_positions[:top_k]:
            bucket = compute_bucket(anchor, total)
            pool = [pos for pos in bucket_to_candidates[bucket] if pos not in used]
            if not pool:
                pool = [pos for pos in candidate_positions if pos not in used]
            if not pool:
                break
            picked = rng.choice(pool)
            chosen.append(picked)
            used.add(picked)
        return chosen

    raise ValueError(f"Unknown selector: {selector}")


def build_intervention_job(
    *,
    example_index: int,
    rollout_index: int,
    selector_name: str,
    repeat_index: int,
    positions: list[int],
    response_token_ids: list[int],
    raw_logits: list[torch.Tensor],
    prompt_ids: torch.Tensor,
    tokenizer,
    args,
    original_score: float,
    data_source: str,
    ground_truth,
    extra_info,
    js_values: list[float],
    entropy_values: list[float],
    margin_values: list[float],
):
    replacement_tokens = []
    for pos in positions:
        replacement_token = choose_replacement_token(
            logits=raw_logits[pos],
            original_token_id=response_token_ids[pos],
            t0=args.t0,
            strategy=args.replacement_strategy,
            tokenizer=tokenizer,
            allow_eos_replacement=args.allow_eos_replacement,
        )
        if replacement_token is None:
            return None
        replacement_tokens.append(int(replacement_token))

    start_pos = positions[0]
    prefix_response_ids = response_token_ids[:start_pos] + replacement_tokens
    full_prefix_ids = torch.cat(
        [prompt_ids[0], torch.tensor(prefix_response_ids, dtype=prompt_ids.dtype)], dim=0
    )
    original_tokens = [int(response_token_ids[pos]) for pos in positions]
    replacement_token_texts = [
        tokenizer.decode([token_id], skip_special_tokens=False) for token_id in replacement_tokens
    ]
    original_token_texts = [
        tokenizer.decode([token_id], skip_special_tokens=False) for token_id in original_tokens
    ]
    return {
        "example_index": int(example_index),
        "rollout_index": int(rollout_index),
        "selector": selector_name,
        "repeat_index": int(repeat_index),
        "position": int(start_pos),
        "positions": [int(pos) for pos in positions],
        "num_replaced_tokens": len(positions),
        "intervention_granularity": args.intervention_granularity,
        "original_length": len(response_token_ids),
        "original_score": float(original_score),
        "original_token": int(original_tokens[0]),
        "replacement_token": int(replacement_tokens[0]),
        "replacement_token_text": replacement_token_texts[0],
        "original_token_text": original_token_texts[0],
        "original_tokens": original_tokens,
        "replacement_tokens": replacement_tokens,
        "replacement_token_texts": replacement_token_texts,
        "original_token_texts": original_token_texts,
        "position_ratio": float(start_pos / max(len(response_token_ids) - 1, 1)),
        "js_value": float(sum(js_values[pos] for pos in positions) / len(positions)),
        "entropy_value": float(sum(entropy_values[pos] for pos in positions) / len(positions)),
        "margin_value": float(sum(margin_values[pos] for pos in positions) / len(positions)),
        "data_source": str(data_source),
        "ground_truth": ground_truth,
        "extra_info": extra_info,
        "prefix_ids": full_prefix_ids,
        "remaining_budget": max(args.max_new_tokens - len(prefix_response_ids), 0),
        "prefix_response_ids": prefix_response_ids,
    }


def summarize_results(results: list[InterventionResult]) -> dict[str, Any]:
    summary = {}
    by_selector = defaultdict(list)
    for item in results:
        by_selector[item.selector].append(item)

    for selector, items in by_selector.items():
        flip_rate = sum(item.flipped for item in items) / max(len(items), 1)
        reward_drop_mean = sum(item.reward_drop for item in items) / max(len(items), 1)
        score_delta_mean = sum(item.score_delta for item in items) / max(len(items), 1)
        summary[selector] = {
            "count": len(items),
            "flip_rate": flip_rate,
            "reward_drop_mean": reward_drop_mean,
            "score_delta_mean": score_delta_mean,
            "mean_position_ratio": sum(item.position_ratio for item in items) / max(len(items), 1),
            "mean_js_value": sum(item.js_value for item in items) / max(len(items), 1),
            "mean_num_replaced_tokens": sum(item.num_replaced_tokens for item in items) / max(len(items), 1),
            "mean_num_intervention_rounds": sum(item.num_intervention_rounds for item in items) / max(len(items), 1),
        }
    return summary


def aggregate_outputs(output_dir: Path):
    partial_dir = output_dir / "partials"
    partial_dir.mkdir(parents=True, exist_ok=True)
    combined_jsonl = output_dir / "interventions.jsonl"
    results: list[InterventionResult] = []
    with combined_jsonl.open("w", encoding="utf-8") as fout:
        for partial_path in sorted(partial_dir.glob("rank_*.jsonl")):
            with partial_path.open("r", encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)
                    results.append(InterventionResult(**json.loads(line)))
    return results


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    use_dist = args.shard_rank is None and args.num_shards is None
    if use_dist:
        rank, world_size, _ = init_distributed()
    else:
        if args.shard_rank is None or args.num_shards is None:
            raise ValueError("Both --shard-rank and --num-shards must be provided together.")
        rank = int(args.shard_rank)
        world_size = int(args.num_shards)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_dir = output_dir / "partials"
    partial_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=get_torch_dtype(args.dtype),
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",  # Enable FlashAttention-2
    )
    model.eval()

    # Optional: compile model for additional speedup (PyTorch 2.0+)
    if args.use_torch_compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model, mode="reduce-overhead")

    dataset = pd.read_parquet(args.data_path)
    selectors = [selector.strip() for selector in args.selectors.split(",") if selector.strip()]
    results: list[InterventionResult] = []
    success_count_local = 0
    scanned_count_local = 0
    target_successes_local = math.ceil(args.target_successes / max(world_size, 1))

    config_path = output_dir / "config.json"
    if rank == 0:
        config_payload = {
            **vars(args),
            "world_size": world_size,
            "target_successes_per_rank": target_successes_local,
        }
        config_path.write_text(json.dumps(config_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    jsonl_path = partial_dir / f"rank_{rank:03d}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fout:
        example_indices = list(range(rank, min(len(dataset), args.max_examples), world_size))
        progress = tqdm(example_indices, total=len(example_indices), disable=rank != 0)
        for example_index in progress:
            if scanned_count_local >= math.ceil(args.max_examples / max(world_size, 1)) or success_count_local >= target_successes_local:
                break
            scanned_count_local += 1

            row = dataset.iloc[example_index]
            prompt_messages = row[args.prompt_key]
            reward_model = row[args.reward_model_key]
            data_source = row[args.data_source_key]
            extra_info = row.get(args.extra_info_key, {}) if args.extra_info_key in row else {}
            ground_truth = reward_model["ground_truth"]

            prompt_ids = build_prompt_ids(tokenizer, prompt_messages, enable_thinking=args.enable_thinking)
            response_token_id_lists, raw_logits_lists = generate_responses_for_prompt(
                model=model,
                tokenizer=tokenizer,
                prompt_ids=prompt_ids,
                rollouts_per_prompt=args.rollouts_per_prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.t1,
                top_k=args.top_k,
                top_p=args.top_p,
                seed=args.seed + example_index,
            )
            intervention_jobs = []
            for rollout_index, (response_token_ids, raw_logits) in enumerate(zip(response_token_id_lists, raw_logits_lists)):
                if success_count_local >= target_successes_local:
                    break
                if not response_token_ids:
                    continue

                response_text = tokenizer.decode(response_token_ids, skip_special_tokens=True)
                original_score = compute_score_float(
                    data_source=data_source,
                    solution_str=response_text,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                )
                if original_score <= 0:
                    continue

                success_count_local += 1
                js_values, entropy_values, margin_values = compute_js_entropy_margin(raw_logits, t0=args.t0, t1=args.t1)
                candidate_positions = list(range(len(response_token_ids)))
                if not args.allow_eos_replacement and tokenizer.eos_token_id is not None:
                    candidate_positions = [
                        idx for idx in candidate_positions if response_token_ids[idx] != tokenizer.eos_token_id
                    ]
                if not candidate_positions:
                    continue

                filtered_js = [js_values[idx] for idx in candidate_positions]
                filtered_entropy = [entropy_values[idx] for idx in candidate_positions]
                filtered_margin = [margin_values[idx] for idx in candidate_positions]

                selector_span_items = []
                per_example_rng = random.Random(args.seed + example_index * 9973 + rollout_index)
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
                                rng=random.Random(args.seed + example_index * 1619 + rollout_index * 131 + repeat_idx),
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

                for selector_name, repeat_index, spans in selector_span_items:
                    for span in spans:
                        job = build_intervention_job(
                            example_index=example_index,
                            rollout_index=rollout_index,
                            selector_name=selector_name,
                            repeat_index=repeat_index,
                            positions=span,
                            response_token_ids=response_token_ids,
                            raw_logits=raw_logits,
                            prompt_ids=prompt_ids,
                            tokenizer=tokenizer,
                            args=args,
                            original_score=original_score,
                            data_source=data_source,
                            ground_truth=ground_truth,
                            extra_info=extra_info,
                            js_values=js_values,
                            entropy_values=entropy_values,
                            margin_values=margin_values,
                        )
                        if job is not None:
                            intervention_jobs.append(job)

                for start in range(0, len(intervention_jobs), args.analysis_batch_size):
                    batch_jobs = intervention_jobs[start : start + args.analysis_batch_size]
                    if not batch_jobs:
                        continue
                    prefix_batch = [job["prefix_ids"] for job in batch_jobs]
                    max_remaining_budget = max(job["remaining_budget"] for job in batch_jobs)
                    suffix_batches = greedy_continue_batch(
                        model=model,
                        tokenizer=tokenizer,
                        prefix_id_list=prefix_batch,
                        max_new_tokens=max_remaining_budget,
                    )
                    for job, suffix_ids in zip(batch_jobs, suffix_batches):
                        suffix_ids = suffix_ids[: job["remaining_budget"]]
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
                        results.append(result)
                        fout.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

                intervention_jobs = []

    summary_local = {
        "rank": rank,
        "world_size": world_size,
        "model_path": args.model_path,
        "data_path": args.data_path,
        "scanned_examples_local": scanned_count_local,
        "successful_exploration_trajectories_local": success_count_local,
        "selectors": selectors,
        "summary_local": summarize_results(results),
    }
    (partial_dir / f"summary_rank_{rank:03d}.json").write_text(
        json.dumps(summary_local, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if use_dist:
        distributed_barrier()

    if not args.skip_finalize and rank == 0:
        aggregated_results = aggregate_outputs(output_dir)
        summary = {
            "model_path": args.model_path,
            "data_path": args.data_path,
            "scanned_examples_requested": args.max_examples,
            "target_successes_requested": args.target_successes,
            "successful_exploration_trajectories": success_count_local if world_size == 1 else None,
            "selectors": selectors,
            "summary": summarize_results(aggregated_results),
            "world_size": world_size,
            "rollouts_per_prompt": args.rollouts_per_prompt,
            "matched_random_repeats": args.matched_random_repeats,
        }
        if world_size > 1:
            rank_summaries = []
            total_scanned = 0
            total_successes = 0
            for summary_path in sorted(partial_dir.glob("summary_rank_*.json")):
                rank_summary = json.loads(summary_path.read_text(encoding="utf-8"))
                rank_summaries.append(rank_summary)
                total_scanned += rank_summary["scanned_examples_local"]
                total_successes += rank_summary["successful_exploration_trajectories_local"]
            summary["rank_summaries"] = rank_summaries
            summary["scanned_examples"] = total_scanned
            summary["successful_exploration_trajectories"] = total_successes
        else:
            summary["scanned_examples"] = scanned_count_local

        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False))

    if use_dist and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
