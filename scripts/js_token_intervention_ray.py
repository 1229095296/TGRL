#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import ray

from js_token_intervention import aggregate_outputs, summarize_results


def parse_args():
    parser = argparse.ArgumentParser(description="Ray launcher for JS token intervention analysis.")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--ray-address", type=str, default=os.environ.get("RAY_ADDRESS", "auto"))
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--gpus-per-worker", type=float, default=8.0)
    parser.add_argument("--cpus-per-worker", type=float, default=4.0)
    parser.add_argument("--script-path", type=str, default=str(Path(__file__).with_name("js_token_intervention.py")))
    return parser.parse_known_args()


def worker_entry(script_path: str, forwarded_args: list[str], output_dir: str, shard_rank: int, num_shards: int):
    import os

    summary_path = Path(output_dir) / "partials" / f"summary_rank_{shard_rank:03d}.json"
    cmd = [
        sys.executable,
        script_path,
        "--output-dir",
        output_dir,
        "--shard-rank",
        str(shard_rank),
        "--num-shards",
        str(num_shards),
        "--skip-finalize",
        *forwarded_args,
    ]

    # Get CUDA_VISIBLE_DEVICES from Ray's allocation
    # Ray sets this automatically when num_gpus is specified
    env = os.environ.copy()

    # Print GPU allocation info for debugging
    cuda_visible = env.get("CUDA_VISIBLE_DEVICES", "not set")
    print(f"[Worker {shard_rank}] CUDA_VISIBLE_DEVICES={cuda_visible}")

    proc = subprocess.run(
        cmd,
        check=False,
        cwd=str(Path(script_path).resolve().parent.parent),
        text=True,
        stdout=sys.stdout,  # Stream output to console in real-time
        stderr=sys.stderr,  # Stream errors to console in real-time
        env=env,  # Pass environment variables to subprocess
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"js_token_intervention worker failed with return code {proc.returncode}\n"
            f"shard_rank={shard_rank} num_shards={num_shards}\n"
            f"cmd={' '.join(cmd)}"
        )
    return json.loads(summary_path.read_text(encoding="utf-8"))


def main():
    args, forwarded_args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "partials").mkdir(parents=True, exist_ok=True)

    ray.init(address=args.ray_address, ignore_reinit_error=True, log_to_driver=True)

    remote_worker = ray.remote(
        num_gpus=args.gpus_per_worker,
        num_cpus=args.cpus_per_worker,
        max_calls=1,
    )(worker_entry)

    refs = [
        remote_worker.options(scheduling_strategy="SPREAD").remote(
            script_path=args.script_path,
            forwarded_args=forwarded_args,
            output_dir=str(output_dir),
            shard_rank=worker_idx,
            num_shards=args.num_workers,
        )
        for worker_idx in range(args.num_workers)
    ]
    rank_summaries = ray.get(refs)

    aggregated_results = aggregate_outputs(output_dir)

    target_successes_requested = None
    max_examples_requested = None
    rollouts_per_prompt = None
    matched_random_repeats = None
    selectors = None
    model_path = None
    data_path = None
    config_path = output_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        model_path = config.get("model_path")
        data_path = config.get("data_path")
        target_successes_requested = config.get("target_successes")
        max_examples_requested = config.get("max_examples")
        rollouts_per_prompt = config.get("rollouts_per_prompt")
        matched_random_repeats = config.get("matched_random_repeats")
        selectors = [selector.strip() for selector in config.get("selectors", "").split(",") if selector.strip()]

    total_scanned = sum(item["scanned_examples_local"] for item in rank_summaries)
    total_successes = sum(item["successful_exploration_trajectories_local"] for item in rank_summaries)
    summary = {
        "model_path": model_path,
        "data_path": data_path,
        "world_size": args.num_workers,
        "gpus_per_worker": args.gpus_per_worker,
        "cpus_per_worker": args.cpus_per_worker,
        "scanned_examples_requested": max_examples_requested,
        "target_successes_requested": target_successes_requested,
        "scanned_examples": total_scanned,
        "successful_exploration_trajectories": total_successes,
        "rollouts_per_prompt": rollouts_per_prompt,
        "matched_random_repeats": matched_random_repeats,
        "selectors": selectors,
        "summary": summarize_results(aggregated_results),
        "rank_summaries": rank_summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
