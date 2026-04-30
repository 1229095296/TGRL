import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from experiments.mixed_temp_grpo_1p7b.reward.gsm8k_reward import extract_final_answer
from experiments.mixed_temp_grpo_1p7b.scripts.probe_utils import (
    PairSelectionConfig,
    ProbeThresholdConfig,
    apply_medium_filter_from_stats,
    build_medium_stats_for_pair,
    compute_pair_rankings,
    config_to_dict,
    create_probe_backend,
    maybe_load_hf_gsm8k,
    probe_dataset_with_backend,
    summarize_probe_records,
    write_json,
)


DEFAULT_INSTRUCTION = 'Let\'s think step by step and output the final answer after "####".'


@dataclass
class MediumFilterConfig:
    min_pass: float = 0.05
    max_pass: float = 0.8
    min_variance: float = 0.01
    min_extract_rate: float = 0.9
    allow_full_fallback: bool = True


def _coerce_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _coerce_split(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    split = str(value).strip().lower()
    return split or None


def _extract_ground_truth_from_answer(answer_raw: str) -> str:
    answer = extract_final_answer(answer_raw, method="strict")
    if answer is None:
        raise ValueError(f"Failed to extract GSM8K ground truth from answer: {answer_raw[:80]}")
    return answer


def build_verl_row(
    question_raw: str,
    answer_raw: str,
    split: str,
    index: int,
    instruction_following: str = DEFAULT_INSTRUCTION,
    data_source: str = "openai/gsm8k",
) -> dict[str, Any]:
    question = f"{question_raw} {instruction_following}".strip()
    solution = _extract_ground_truth_from_answer(answer_raw)
    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": question}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": solution},
        "extra_info": {
            "split": split,
            "index": int(index),
            "answer": answer_raw,
            "question": question_raw,
        },
    }


def normalize_input_dataframe(
    df: pd.DataFrame,
    split: str,
    data_source: str = "openai/gsm8k",
    instruction_following: str = DEFAULT_INSTRUCTION,
) -> pd.DataFrame:
    if {"data_source", "prompt", "reward_model", "extra_info"}.issubset(df.columns):
        normalized = df.copy()
        normalized["data_source"] = normalized["data_source"].fillna(data_source)
        return normalized

    if not {"question", "answer"}.issubset(df.columns):
        raise ValueError("Input parquet must be verl-format or contain 'question' and 'answer' columns.")

    rows = []
    for idx, row in df.reset_index(drop=True).iterrows():
        rows.append(
            build_verl_row(
                question_raw=str(row["question"]),
                answer_raw=str(row["answer"]),
                split=split,
                index=int(idx),
                instruction_following=instruction_following,
                data_source=data_source,
            )
        )
    return pd.DataFrame(rows)


def extract_split_series(df: pd.DataFrame) -> pd.Series | None:
    if "split" in df.columns:
        return df["split"].apply(_coerce_split)
    if "extra_info" in df.columns:
        return df["extra_info"].apply(lambda info: _coerce_split(_coerce_dict(info).get("split")))
    return None


def _build_stable_row_keys(df: pd.DataFrame) -> list[str]:
    if "extra_info" in df.columns:
        keys = []
        for idx, info in enumerate(df["extra_info"]):
            info_dict = _coerce_dict(info)
            if "index" in info_dict:
                keys.append(f"extra_index::{info_dict['index']}")
            elif "question" in info_dict:
                keys.append(f"extra_question::{info_dict['question']}")
            else:
                keys.append(f"row::{idx}")
        return keys

    if "question" in df.columns:
        return [f"question::{question}" for question in df["question"].astype(str).tolist()]

    if "prompt" in df.columns:
        return [f"prompt::{json.dumps(prompt, ensure_ascii=False, sort_keys=True)}" for prompt in df["prompt"].tolist()]

    return [f"row::{idx}" for idx in range(len(df))]


def deterministic_train_val_split(
    df: pd.DataFrame,
    val_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    keys = _build_stable_row_keys(df)
    is_val = []
    for key in keys:
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) / 0xFFFFFFFF
        is_val.append(bucket < val_fraction)

    split_mask = pd.Series(is_val, index=df.index)
    val_df = df[split_mask].reset_index(drop=True)
    train_df = df[~split_mask].reset_index(drop=True)

    if train_df.empty or val_df.empty:
        val_count = max(1, min(len(df) - 1, int(round(len(df) * val_fraction))))
        if val_count <= 0:
            raise ValueError(f"Cannot create a non-empty train/val split from {len(df)} rows.")
        val_df = df.iloc[:val_count].reset_index(drop=True)
        train_df = df.iloc[val_count:].reset_index(drop=True)

    return train_df, val_df


def split_single_input_dataframe(
    df: pd.DataFrame,
    train_split_names: list[str],
    val_split_names: list[str],
    allow_auto_split_fallback: bool = True,
    fallback_val_fraction: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_series = extract_split_series(df)
    if split_series is None:
        if allow_auto_split_fallback:
            return deterministic_train_val_split(df, val_fraction=fallback_val_fraction)
        raise ValueError(
            "Single-input mode requires a split column or extra_info.split so the script can separate train/val rows."
        )

    train_mask = split_series.isin({name.lower() for name in train_split_names})
    val_mask = split_series.isin({name.lower() for name in val_split_names})
    train_df = df[train_mask].reset_index(drop=True)
    val_df = df[val_mask].reset_index(drop=True)

    if train_df.empty or val_df.empty:
        if allow_auto_split_fallback:
            return deterministic_train_val_split(df, val_fraction=fallback_val_fraction)
        raise ValueError(
            f"Failed to split single input into train/val. "
            f"train_rows={len(train_df)}, val_rows={len(val_df)}, "
            f"train_split_names={train_split_names}, val_split_names={val_split_names}"
        )
    return train_df, val_df


def load_stats_dataframe(stats_path: str | None) -> pd.DataFrame | None:
    if not stats_path:
        return None
    from experiments.mixed_temp_grpo_1p7b.scripts.probe_utils import load_records_table

    return load_records_table(stats_path)


def maybe_generate_probe_stats(
    normalized_df: pd.DataFrame,
    split: str,
    args: argparse.Namespace,
    output_dir: Path,
    backend: Any | None = None,
    selected_pair_override: tuple[float, float] | None = None,
) -> tuple[pd.DataFrame | None, Path | None, dict[str, Any] | None]:
    if args.probe_backend is None:
        return None, None, None

    if backend is None:
        backend = create_probe_backend(
            backend_name=args.probe_backend,
            model_path=args.probe_model_path,
            tokenizer_path=args.probe_tokenizer_path,
            device=args.probe_device,
            max_new_tokens=args.probe_max_new_tokens,
            top_p=args.probe_top_p,
            top_k=args.probe_top_k,
            trust_remote_code=args.probe_trust_remote_code,
            seed=args.probe_seed,
        )
    low_candidates = args.probe_low_temperatures or [args.probe_low_temperature]
    high_candidates = args.probe_high_temperatures or [args.probe_high_temperature]
    temperatures = sorted(set([float(temp) for temp in low_candidates + high_candidates]))
    print(
        f"[prepare] probing split={split} rows={len(normalized_df)} "
        f"temperatures={temperatures} n_probe={args.probe_n}"
    )
    probe_records = probe_dataset_with_backend(
        normalized_df=normalized_df,
        temperatures=temperatures,
        n_probe=args.probe_n,
        backend=backend,
        split=split,
        response_method=args.probe_method,
        batch_size=args.probe_batch_size,
        max_prompts=args.probe_max_prompts,
        progress_desc=f"probe[{split}]",
    )
    probe_path = output_dir / f"{split}_probe_responses.parquet"
    probe_records.to_parquet(probe_path, index=False)

    prompt_stats = summarize_probe_records(probe_records)
    prompt_stats_path = output_dir / f"{split}_probe_prompt_stats.csv"
    prompt_stats.to_csv(prompt_stats_path, index=False)

    best_pair = None
    if selected_pair_override is not None:
        selected_low = float(selected_pair_override[0])
        selected_high = float(selected_pair_override[1])
    else:
        selected_low = float(args.probe_low_temperature)
        selected_high = float(args.probe_high_temperature)
    if args.auto_select_pair or len(low_candidates) > 1 or len(high_candidates) > 1:
        pair_cfg = PairSelectionConfig(
            min_extract_rate=args.min_extract_rate,
            min_variance=args.min_variance,
        )
        pair_rankings, best_pair = compute_pair_rankings(
            prompt_stats=prompt_stats,
            low_candidates=[float(temp) for temp in low_candidates],
            high_candidates=[float(temp) for temp in high_candidates],
            cfg=pair_cfg,
        )
        pair_rankings.to_csv(output_dir / f"{split}_probe_pair_ranking.csv", index=False)
        if selected_pair_override is None and best_pair:
            selected_low = float(best_pair["T_low"])
            selected_high = float(best_pair["T_high"])
        elif selected_pair_override is not None:
            best_pair = {
                "T_low": selected_low,
                "T_high": selected_high,
                "selection_source": "override",
            }

    pair_stats = build_medium_stats_for_pair(
        prompt_stats=prompt_stats,
        low_temperature=selected_low,
        high_temperature=selected_high,
    )
    pair_path = output_dir / f"{split}_probe_pair_stats.csv"
    pair_stats.to_csv(pair_path, index=False)
    print(
        f"[prepare] finished split={split} probe_records={len(probe_records)} "
        f"selected_pair=({selected_low:.3f}, {selected_high:.3f})"
    )
    return pair_stats, probe_path, best_pair


def attach_medium_stats(
    normalized_df: pd.DataFrame,
    stats_df: pd.DataFrame | None,
    split: str,
) -> pd.DataFrame:
    frame = normalized_df.copy()
    frame["_row_id"] = range(len(frame))
    frame["split"] = split
    frame["index"] = frame["extra_info"].apply(lambda info: _coerce_dict(info).get("index"))

    if stats_df is None:
        frame["pass_low"] = pd.NA
        frame["pass_high"] = pd.NA
        frame["reward_variance"] = pd.NA
        frame["extract_rate"] = pd.NA
        frame["medium_keep"] = True
        frame["medium_reason"] = "fallback_full"
        return frame

    stats = stats_df.copy()
    if "split" not in stats.columns:
        stats["split"] = split

    if {"pass_low", "pass_high", "reward_variance", "extract_rate"}.issubset(stats.columns):
        prompt_stats = stats.copy()
    else:
        required = {"temperature", "pass_rate", "reward_variance", "extract_rate"}
        if required.issubset(stats.columns):
            if "low_temperature" not in frame.columns or "high_temperature" not in frame.columns:
                raise ValueError(
                    "Temperature-level stats require low_temperature/high_temperature metadata before attachment."
                )
            low_temperature = float(frame["low_temperature"].iloc[0])
            high_temperature = float(frame["high_temperature"].iloc[0])
            prompt_stats = build_medium_stats_for_pair(stats, low_temperature=low_temperature, high_temperature=high_temperature)
        else:
            prompt_id_col = "index" if "index" in stats.columns else "prompt_id"
            if prompt_id_col not in stats.columns:
                raise ValueError("Stats file must contain prompt IDs and either pair stats or temperature stats.")
            stats = stats.rename(columns={prompt_id_col: "index"})
            prompt_stats = stats

    merged = frame.merge(
        prompt_stats[["split", "index", "pass_low", "pass_high", "reward_variance", "extract_rate"]],
        on=["split", "index"],
        how="left",
    )
    return merged


def select_medium_rows(df: pd.DataFrame, cfg: MediumFilterConfig) -> pd.DataFrame:
    frame = df.copy()
    if frame["pass_low"].isna().all() and frame["pass_high"].isna().all():
        frame["medium_keep"] = True
        frame["medium_reason"] = "fallback_full"
        return frame
    if "split" not in frame.columns:
        frame["split"] = frame.get("extra_info", pd.Series([{}] * len(frame))).apply(
            lambda info: _coerce_dict(info).get("split", "unknown")
        )
    if "index" not in frame.columns:
        frame["index"] = frame.get("extra_info", pd.Series([{}] * len(frame))).apply(
            lambda info: _coerce_dict(info).get("index")
        )
        if frame["index"].isna().any():
            frame.loc[frame["index"].isna(), "index"] = list(range(len(frame.loc[frame["index"].isna()])))
    filtered = apply_medium_filter_from_stats(
        frame[["split", "index", "pass_low", "pass_high", "reward_variance", "extract_rate"]].copy(),
        ProbeThresholdConfig(
            min_pass=cfg.min_pass,
            max_pass=cfg.max_pass,
            min_variance=cfg.min_variance,
            min_extract_rate=cfg.min_extract_rate,
            allow_full_fallback=cfg.allow_full_fallback,
        ),
    )
    frame["medium_keep"] = filtered["medium_keep"]
    frame["medium_reason"] = filtered["medium_reason"]
    return frame


def build_summary(train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict[str, Any]:
    def split_summary(df: pd.DataFrame) -> dict[str, Any]:
        medium_df = df[df["medium_keep"]]
        summary = {
            "full_count": int(len(df)),
            "medium_count": int(len(medium_df)),
            "medium_fraction": float(len(medium_df) / len(df)) if len(df) > 0 else 0.0,
            "mean_pass_low": float(pd.to_numeric(df["pass_low"], errors="coerce").dropna().mean())
            if "pass_low" in df.columns and df["pass_low"].notna().any()
            else None,
            "mean_pass_high": float(pd.to_numeric(df["pass_high"], errors="coerce").dropna().mean())
            if "pass_high" in df.columns and df["pass_high"].notna().any()
            else None,
            "mean_reward_variance": float(pd.to_numeric(df["reward_variance"], errors="coerce").dropna().mean())
            if "reward_variance" in df.columns and df["reward_variance"].notna().any()
            else None,
            "mean_extract_rate": float(pd.to_numeric(df["extract_rate"], errors="coerce").dropna().mean())
            if "extract_rate" in df.columns and df["extract_rate"].notna().any()
            else None,
        }
        return summary

    return {"train": split_summary(train_df), "val": split_summary(val_df)}


def _finalize_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [col for col in ["data_source", "prompt", "ability", "reward_model", "extra_info"] if col in df.columns]
    return df[keep_cols].reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare full/medium GSM8K parquet files for mixed-temp GRPO.")
    parser.add_argument("--input", default=None, help="Single parquet containing both train and val/test splits.")
    parser.add_argument("--train-input", default=None, help="Raw GSM8K parquet or verl-format train parquet.")
    parser.add_argument("--val-input", default=None, help="Raw GSM8K parquet or verl-format val/test parquet.")
    parser.add_argument("--hf-dataset", default=None, help="Optional HF dataset name, e.g. openai/gsm8k.")
    parser.add_argument("--hf-config", default="main")
    parser.add_argument("--hf-train-split", default="train")
    parser.add_argument("--hf-val-split", default="test")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-stats", default=None, help="Optional stats file for train medium filtering.")
    parser.add_argument("--val-stats", default=None, help="Optional stats file for val medium filtering.")
    parser.add_argument("--data-source", default="openai/gsm8k")
    parser.add_argument("--instruction-following", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--min-pass", type=float, default=0.05)
    parser.add_argument("--max-pass", type=float, default=0.8)
    parser.add_argument("--min-variance", type=float, default=0.01)
    parser.add_argument("--min-extract-rate", type=float, default=0.9)
    parser.add_argument("--disable-full-fallback", action="store_true")
    parser.add_argument("--train-split-names", default="train", help="Comma-separated split names for train in --input mode.")
    parser.add_argument(
        "--val-split-names",
        default="test,val,validation",
        help="Comma-separated split names for val in --input mode.",
    )
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--fallback-val-fraction", type=float, default=0.1)
    parser.add_argument("--disable-auto-split-fallback", action="store_true")
    parser.add_argument("--probe-backend", default=None, choices=["transformers"], help="Enable online probe generation.")
    parser.add_argument("--probe-model-path", default=None)
    parser.add_argument("--probe-tokenizer-path", default=None)
    parser.add_argument("--probe-device", default="auto")
    parser.add_argument("--probe-trust-remote-code", action="store_true")
    parser.add_argument("--probe-low-temperature", type=float, default=0.3)
    parser.add_argument("--probe-high-temperature", type=float, default=1.0)
    parser.add_argument("--probe-low-temperatures", nargs="*", type=float, default=None)
    parser.add_argument("--probe-high-temperatures", nargs="*", type=float, default=None)
    parser.add_argument("--auto-select-pair", action="store_true")
    parser.add_argument("--probe-n", type=int, default=8)
    parser.add_argument("--probe-batch-size", type=int, default=4)
    parser.add_argument("--probe-max-prompts", type=int, default=None)
    parser.add_argument("--probe-max-new-tokens", type=int, default=256)
    parser.add_argument("--probe-top-p", type=float, default=1.0)
    parser.add_argument("--probe-top-k", type=int, default=-1)
    parser.add_argument("--probe-method", default="strict", choices=["strict", "flexible"])
    parser.add_argument("--probe-seed", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.probe_backend is not None:
        if args.probe_n < 1:
            raise ValueError(f"--probe-n must be >= 1, got {args.probe_n}")
        if args.probe_batch_size < 1:
            raise ValueError(f"--probe-batch-size must be >= 1, got {args.probe_batch_size}")
        if args.probe_max_prompts is not None and args.probe_max_prompts < 1:
            raise ValueError(f"--probe-max-prompts must be >= 1 when provided, got {args.probe_max_prompts}")
    cfg = MediumFilterConfig(
        min_pass=args.min_pass,
        max_pass=args.max_pass,
        min_variance=args.min_variance,
        min_extract_rate=args.min_extract_rate,
        allow_full_fallback=not args.disable_full_fallback,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[prepare] output_dir={output_dir}")

    if args.hf_dataset:
        print(f"[prepare] loading HF dataset {args.hf_dataset}:{args.hf_config}")
        train_raw, val_raw = maybe_load_hf_gsm8k(
            dataset_name=args.hf_dataset,
            config_name=args.hf_config,
            train_split=args.hf_train_split,
            val_split=args.hf_val_split,
        )
    elif args.input:
        print(f"[prepare] loading single input parquet {args.input}")
        raw = pd.read_parquet(args.input)
        train_raw, val_raw = split_single_input_dataframe(
            raw,
            train_split_names=[name.strip() for name in args.train_split_names.split(",") if name.strip()],
            val_split_names=[name.strip() for name in args.val_split_names.split(",") if name.strip()],
            allow_auto_split_fallback=not args.disable_auto_split_fallback,
            fallback_val_fraction=args.fallback_val_fraction,
        )
    else:
        if not args.train_input or not args.val_input:
            raise ValueError("Either provide --input or both --train-input and --val-input.")
        print(f"[prepare] loading train parquet {args.train_input}")
        train_raw = pd.read_parquet(args.train_input)
        print(f"[prepare] loading val parquet {args.val_input}")
        val_raw = pd.read_parquet(args.val_input)

    if args.train_limit is not None:
        train_raw = train_raw.head(args.train_limit).reset_index(drop=True)
    if args.val_limit is not None:
        val_raw = val_raw.head(args.val_limit).reset_index(drop=True)
    if train_raw.empty or val_raw.empty:
        raise ValueError(f"Loaded empty split(s): train_rows={len(train_raw)} val_rows={len(val_raw)}")
    print(f"[prepare] loaded rows train={len(train_raw)} val={len(val_raw)}")

    train_norm = normalize_input_dataframe(
        train_raw,
        split="train",
        data_source=args.data_source,
        instruction_following=args.instruction_following,
    )
    val_norm = normalize_input_dataframe(
        val_raw,
        split="val",
        data_source=args.data_source,
        instruction_following=args.instruction_following,
    )
    train_norm["low_temperature"] = args.probe_low_temperature
    train_norm["high_temperature"] = args.probe_high_temperature
    val_norm["low_temperature"] = args.probe_low_temperature
    val_norm["high_temperature"] = args.probe_high_temperature
    print(f"[prepare] normalized rows train={len(train_norm)} val={len(val_norm)}")

    train_stats = load_stats_dataframe(args.train_stats)
    val_stats = load_stats_dataframe(args.val_stats)
    train_probe_path = None
    val_probe_path = None
    train_best_pair = None
    val_best_pair = None
    shared_probe_backend = None
    if args.probe_backend is not None and (train_stats is None or val_stats is None):
        print("[prepare] creating shared probe backend once for all splits")
        shared_probe_backend = create_probe_backend(
            backend_name=args.probe_backend,
            model_path=args.probe_model_path,
            tokenizer_path=args.probe_tokenizer_path,
            device=args.probe_device,
            max_new_tokens=args.probe_max_new_tokens,
            top_p=args.probe_top_p,
            top_k=args.probe_top_k,
            trust_remote_code=args.probe_trust_remote_code,
            seed=args.probe_seed,
        )

    if train_stats is None:
        train_stats, train_probe_path, train_best_pair = maybe_generate_probe_stats(
            train_norm,
            split="train",
            args=args,
            output_dir=output_dir,
            backend=shared_probe_backend,
        )
    if val_stats is None:
        selected_pair_override = None
        if train_best_pair is not None and train_best_pair.get("T_low") is not None and train_best_pair.get("T_high") is not None:
            selected_pair_override = (float(train_best_pair["T_low"]), float(train_best_pair["T_high"]))
        val_stats, val_probe_path, val_best_pair = maybe_generate_probe_stats(
            val_norm,
            split="val",
            args=args,
            output_dir=output_dir,
            backend=shared_probe_backend,
            selected_pair_override=selected_pair_override,
        )

    train_medium = select_medium_rows(attach_medium_stats(train_norm, train_stats, split="train"), cfg)
    val_medium = select_medium_rows(attach_medium_stats(val_norm, val_stats, split="val"), cfg)

    _finalize_output_columns(train_norm).to_parquet(output_dir / "train_full.parquet", index=False)
    _finalize_output_columns(val_norm).to_parquet(output_dir / "val_full.parquet", index=False)
    _finalize_output_columns(train_medium[train_medium["medium_keep"]]).to_parquet(
        output_dir / "train_medium.parquet", index=False
    )
    _finalize_output_columns(val_medium[val_medium["medium_keep"]]).to_parquet(
        output_dir / "val_medium.parquet", index=False
    )

    summary = build_summary(train_medium, val_medium)
    summary["dataset_variants"] = {
        "full_gsm8k": {
            "train_file": "train_full.parquet",
            "val_file": "val_full.parquet",
        },
        "filtered_medium_gsm8k": {
            "train_file": "train_medium.parquet",
            "val_file": "val_medium.parquet",
        },
    }
    summary["medium_filter_config"] = config_to_dict(
        ProbeThresholdConfig(
            min_pass=cfg.min_pass,
            max_pass=cfg.max_pass,
            min_variance=cfg.min_variance,
            min_extract_rate=cfg.min_extract_rate,
            allow_full_fallback=cfg.allow_full_fallback,
        )
    )
    summary["probe"] = {
        "backend": args.probe_backend,
        "low_temperature": args.probe_low_temperature,
        "high_temperature": args.probe_high_temperature,
        "low_temperature_candidates": args.probe_low_temperatures,
        "high_temperature_candidates": args.probe_high_temperatures,
        "n_probe": args.probe_n,
        "method": args.probe_method,
        "seed": args.probe_seed,
        "train_probe_path": None if train_probe_path is None else train_probe_path.name,
        "val_probe_path": None if val_probe_path is None else val_probe_path.name,
        "selected_pair_train": train_best_pair,
        "selected_pair_val": val_best_pair,
    }
    write_json(output_dir / "summary.json", summary)
    print(
        f"[prepare] done train_medium={int(train_medium['medium_keep'].sum())}/{len(train_medium)} "
        f"val_medium={int(val_medium['medium_keep'].sum())}/{len(val_medium)}"
    )


if __name__ == "__main__":
    main()
