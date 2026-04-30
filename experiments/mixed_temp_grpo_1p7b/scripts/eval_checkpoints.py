from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from experiments.mixed_temp_grpo_1p7b.scripts.analysis_helpers import (
    build_passk_summary,
    load_table,
    normalize_common_columns,
    parse_group_cols,
    write_table,
)
from experiments.mixed_temp_grpo_1p7b.scripts.prepare_gsm8k_medium import (
    DEFAULT_INSTRUCTION,
    normalize_input_dataframe,
)
from experiments.mixed_temp_grpo_1p7b.scripts.probe_utils import (
    create_probe_backend,
    maybe_load_hf_gsm8k,
    probe_dataset_with_backend,
)
from experiments.mixed_temp_grpo_1p7b.reward.gsm8k_reward import compute_score


def load_predictions(path: str) -> pd.DataFrame:
    return load_table(path)


def load_eval_dataset(args: argparse.Namespace) -> pd.DataFrame:
    if args.hf_dataset:
        _, val_raw = maybe_load_hf_gsm8k(
            dataset_name=args.hf_dataset,
            config_name=args.hf_config,
            train_split=args.hf_train_split,
            val_split=args.hf_val_split,
        )
        raw = val_raw
    elif args.dataset:
        raw = pd.read_parquet(args.dataset)
    else:
        raise ValueError("Either provide --predictions or --dataset/--hf-dataset for online evaluation.")

    if args.prompt_limit is not None:
        raw = raw.head(args.prompt_limit).reset_index(drop=True)

    return normalize_input_dataframe(
        raw,
        split=args.dataset_split,
        data_source=args.data_source,
        instruction_following=args.instruction_following,
    )


def generate_predictions_from_checkpoints(args: argparse.Namespace) -> pd.DataFrame:
    if not args.checkpoints:
        raise ValueError("--checkpoints is required when --predictions is not provided.")
    normalized_df = load_eval_dataset(args)
    checkpoint_names = list(args.checkpoint_names or [])
    if checkpoint_names and len(checkpoint_names) != len(args.checkpoints):
        raise ValueError("--checkpoint-names must have the same length as --checkpoints.")

    n_samples = max(int(k) for k in [int(part.strip()) for part in args.ks.split(",") if part.strip()])
    records = []
    for ckpt_idx, checkpoint in enumerate(args.checkpoints):
        checkpoint_name = checkpoint_names[ckpt_idx] if checkpoint_names else Path(checkpoint).name or f"ckpt_{ckpt_idx}"
        backend = create_probe_backend(
            backend_name=args.probe_backend,
            model_path=checkpoint,
            tokenizer_path=args.probe_tokenizer_path,
            device=args.probe_device,
            max_new_tokens=args.probe_max_new_tokens,
            top_p=args.probe_top_p,
            top_k=args.probe_top_k,
            trust_remote_code=args.probe_trust_remote_code,
            seed=args.seed,
        )
        generated = probe_dataset_with_backend(
            normalized_df=normalized_df,
            temperatures=[args.eval_temperature],
            n_probe=n_samples,
            backend=backend,
            split=args.dataset_split,
            response_method=args.method,
            batch_size=args.probe_batch_size,
        )
        generated = generated.rename(columns={"temperature": "temperature_used", "probe_idx": "sample_idx_within_prompt"})
        generated["checkpoint"] = checkpoint_name
        generated["checkpoint_path"] = checkpoint
        generated["method"] = args.experiment_method
        generated["seed"] = args.seed
        generated["eval_temperature"] = float(args.eval_temperature)
        records.append(generated)
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def score_predictions(df: pd.DataFrame, method: str = "strict") -> pd.DataFrame:
    frame = normalize_common_columns(df)
    if "reward" in frame.columns:
        frame["reward"] = pd.to_numeric(frame["reward"], errors="coerce").fillna(0.0)
        return frame

    if "response" not in frame.columns and "response_text" in frame.columns:
        frame["response"] = frame["response_text"]

    required = {"data_source", "response", "ground_truth"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Predictions must contain {sorted(required)} or a precomputed reward column.")

    scored = frame.apply(
        lambda row: compute_score(
            data_source=row["data_source"],
            solution_str=row["response"],
            ground_truth=row["ground_truth"],
            extra_info={"index": row.get("index")},
            method=method,
            return_dict=True,
        ),
        axis=1,
    )
    score_df = pd.DataFrame(scored.tolist())
    for col in score_df.columns:
        frame[col] = score_df[col]
    frame["reward"] = frame["score"]
    return frame


def aggregate_eval_metrics(
    df: pd.DataFrame,
    ks: list[int],
    group_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prompt_level, summary = build_passk_summary(df, ks=ks, group_cols=group_cols)
    summary = summary.rename(columns={"reward_mean": "mean_reward", "reward_std": "prompt_reward_std"})
    if "pass_at_1" not in summary.columns and ks:
        summary["pass_at_1"] = summary.get(f"pass_at_{ks[0]}", float("nan"))
    if "eval_temperature" not in summary.columns and "eval_temperature" in df.columns:
        temp_table = (
            df.groupby([col for col in group_cols if col in df.columns and col != "index"], dropna=False)["eval_temperature"]
            .mean()
            .reset_index()
        )
        summary = summary.merge(
            temp_table,
            on=[col for col in temp_table.columns if col != "eval_temperature"],
            how="left",
        )
    return prompt_level, summary


def build_eval_payload(summary_df: pd.DataFrame) -> dict[str, Any]:
    records = summary_df.to_dict(orient="records")
    best = {}
    if not summary_df.empty:
        sort_col = "pass_at_1" if "pass_at_1" in summary_df.columns else "mean_reward"
        best_row = summary_df.sort_values(sort_col, ascending=False).iloc[0]
        best = best_row.to_dict()
    return {"summary": records, "best": best}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate frozen checkpoints for mixed-temp GRPO analysis.")
    parser.add_argument("--predictions", default=None, help="Prediction parquet/jsonl/json/csv.")
    parser.add_argument("--dataset", default=None, help="Held-out parquet in raw GSM8K or verl format.")
    parser.add_argument("--dataset-split", default="val")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method", default="strict", choices=["strict", "flexible"])
    parser.add_argument("--ks", default="1,8,16", help="Comma-separated pass@k values.")
    parser.add_argument(
        "--group-cols",
        default="checkpoint,method,split,seed,index",
        help="Columns used to define a prompt group for pass@k.",
    )
    parser.add_argument("--checkpoints", nargs="+", default=None, help="Checkpoint/model paths for online eval.")
    parser.add_argument("--checkpoint-names", nargs="*", default=None)
    parser.add_argument("--experiment-method", default="mixed_grouped")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eval-temperature", type=float, default=0.6)
    parser.add_argument("--prompt-limit", type=int, default=None)
    parser.add_argument("--data-source", default="openai/gsm8k")
    parser.add_argument("--instruction-following", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--hf-dataset", default=None)
    parser.add_argument("--hf-config", default="main")
    parser.add_argument("--hf-train-split", default="train")
    parser.add_argument("--hf-val-split", default="test")
    parser.add_argument("--probe-backend", default="transformers", choices=["transformers"])
    parser.add_argument("--probe-tokenizer-path", default=None)
    parser.add_argument("--probe-device", default="auto")
    parser.add_argument("--probe-trust-remote-code", action="store_true")
    parser.add_argument("--probe-max-new-tokens", type=int, default=512)
    parser.add_argument("--probe-top-p", type=float, default=1.0)
    parser.add_argument("--probe-top-k", type=int, default=-1)
    parser.add_argument("--probe-batch-size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.predictions:
        predictions = load_predictions(args.predictions)
    else:
        predictions = generate_predictions_from_checkpoints(args)

    scored = score_predictions(predictions, method=args.method)
    ks = [int(part.strip()) for part in args.ks.split(",") if part.strip()]
    group_cols = parse_group_cols(args.group_cols)
    prompt_level, summary = aggregate_eval_metrics(scored, ks=ks, group_cols=group_cols)
    payload = build_eval_payload(summary)

    write_table(scored, output_dir / "scored_predictions.parquet")
    write_table(prompt_level, output_dir / "passk_eval.csv")
    write_table(summary, output_dir / "checkpoint_eval_summary.csv")
    write_table(summary, output_dir / "final_summary.csv")
    with open(output_dir / "checkpoint_eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


if __name__ == "__main__":
    main()
