import argparse
from pathlib import Path

import pandas as pd

from experiments.mixed_temp_grpo_1p7b.scripts.prepare_gsm8k_medium import (
    DEFAULT_INSTRUCTION,
    normalize_input_dataframe,
)
from experiments.mixed_temp_grpo_1p7b.scripts.probe_utils import (
    PairSelectionConfig,
    build_medium_stats_for_pair,
    compute_pair_rankings,
    config_to_dict,
    create_probe_backend,
    load_records_table,
    maybe_load_hf_gsm8k,
    probe_dataset_with_backend,
    summarize_probe_records,
    write_json,
)


def load_prompt_tables(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    if args.hf_dataset:
        return maybe_load_hf_gsm8k(
            dataset_name=args.hf_dataset,
            config_name=args.hf_config,
            train_split=args.hf_train_split,
            val_split=args.hf_val_split,
        )

    if args.input:
        frame = pd.read_parquet(args.input)
        if "split" not in frame.columns and "extra_info" not in frame.columns:
            raise ValueError("--input mode for calibration requires split metadata.")
        split_series = frame["split"] if "split" in frame.columns else frame["extra_info"].apply(lambda x: x.get("split"))
        train_df = frame[split_series.astype(str).str.lower() == args.single_input_train_split.lower()].reset_index(drop=True)
        val_df = frame[split_series.astype(str).str.lower() == args.single_input_val_split.lower()].reset_index(drop=True)
        if train_df.empty or val_df.empty:
            raise ValueError("Failed to separate train/val rows from --input.")
        return train_df, val_df

    if args.train_input and args.val_input:
        return pd.read_parquet(args.train_input), pd.read_parquet(args.val_input)

    raise ValueError("Provide --responses, or prompt sources via --hf-dataset, --input, or --train-input/--val-input.")


def maybe_limit(frame: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    if limit is None:
        return frame
    return frame.head(limit).reset_index(drop=True)


def run_online_probe(args: argparse.Namespace, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_raw, val_raw = load_prompt_tables(args)
    train_df = normalize_input_dataframe(
        maybe_limit(train_raw, args.max_train_prompts),
        split="train",
        data_source=args.data_source,
        instruction_following=args.instruction_following,
    )
    val_df = normalize_input_dataframe(
        maybe_limit(val_raw, args.max_val_prompts),
        split="val",
        data_source=args.data_source,
        instruction_following=args.instruction_following,
    )

    temperatures = sorted(set(args.low_temperatures + args.high_temperatures))
    backend = create_probe_backend(
        backend_name=args.probe_backend,
        model_path=args.probe_model_path,
        tokenizer_path=args.probe_tokenizer_path,
        device=args.probe_device,
        max_new_tokens=args.probe_max_new_tokens,
        top_p=args.probe_top_p,
        top_k=args.probe_top_k,
        trust_remote_code=args.probe_trust_remote_code,
        seed=args.seed,
    )
    train_records = probe_dataset_with_backend(
        normalized_df=train_df,
        temperatures=temperatures,
        n_probe=args.n_probe,
        backend=backend,
        split="train",
        response_method=args.method,
        batch_size=args.probe_batch_size,
    )
    val_records = probe_dataset_with_backend(
        normalized_df=val_df,
        temperatures=temperatures,
        n_probe=args.n_probe,
        backend=backend,
        split="val",
        response_method=args.method,
        batch_size=args.probe_batch_size,
    )
    combined = pd.concat([train_records, val_records], ignore_index=True)
    combined.to_parquet(output_dir / "probe_responses.parquet", index=False)
    return combined, pd.concat([train_df, val_df], ignore_index=True)


def run_offline_probe(args: argparse.Namespace, output_dir: Path) -> pd.DataFrame:
    responses = load_records_table(args.responses)
    responses.to_parquet(output_dir / "probe_responses.parquet", index=False)
    return responses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate candidate T_high values for mixed-temp GRPO.")
    parser.add_argument("--responses", default=None, help="Offline probe responses parquet/jsonl/json/csv.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--low-temperatures", nargs="+", type=float, default=[0.3, 0.4])
    parser.add_argument("--high-temperatures", nargs="+", type=float, default=[0.8, 1.0, 1.2])
    parser.add_argument("--method", default="strict", choices=["strict", "flexible"])
    parser.add_argument("--data-source", default="openai/gsm8k")
    parser.add_argument("--instruction-following", default=DEFAULT_INSTRUCTION)

    parser.add_argument("--hf-dataset", default=None)
    parser.add_argument("--hf-config", default="main")
    parser.add_argument("--hf-train-split", default="train")
    parser.add_argument("--hf-val-split", default="test")
    parser.add_argument("--input", default=None)
    parser.add_argument("--single-input-train-split", default="train")
    parser.add_argument("--single-input-val-split", default="test")
    parser.add_argument("--train-input", default=None)
    parser.add_argument("--val-input", default=None)
    parser.add_argument("--max-train-prompts", type=int, default=128)
    parser.add_argument("--max-val-prompts", type=int, default=128)

    parser.add_argument("--probe-backend", default=None, choices=["transformers"])
    parser.add_argument("--probe-model-path", default=None)
    parser.add_argument("--probe-tokenizer-path", default=None)
    parser.add_argument("--probe-device", default="auto")
    parser.add_argument("--probe-trust-remote-code", action="store_true")
    parser.add_argument("--probe-max-new-tokens", type=int, default=256)
    parser.add_argument("--probe-top-p", type=float, default=1.0)
    parser.add_argument("--probe-top-k", type=int, default=-1)
    parser.add_argument("--probe-batch-size", type=int, default=4)
    parser.add_argument("--n-probe", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--min-extract-rate", type=float, default=0.9)
    parser.add_argument("--min-variance", type=float, default=0.01)
    parser.add_argument("--gain-std-weight", type=float, default=1.0)
    parser.add_argument("--positive-gain-weight", type=float, default=0.5)
    parser.add_argument("--variance-weight", type=float, default=0.25)
    parser.add_argument("--extract-rate-weight", type=float, default=1.0)
    parser.add_argument("--eligible-fraction-weight", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.responses is not None:
        probe_records = run_offline_probe(args, output_dir)
    else:
        if args.probe_backend is None:
            raise ValueError("Online calibration requires --probe-backend and prompt sources.")
        probe_records, _ = run_online_probe(args, output_dir)

    prompt_stats = summarize_probe_records(probe_records)
    prompt_stats.to_csv(output_dir / "prompt_temperature_stats.csv", index=False)

    pair_cfg = PairSelectionConfig(
        min_extract_rate=args.min_extract_rate,
        min_variance=args.min_variance,
        gain_std_weight=args.gain_std_weight,
        positive_gain_weight=args.positive_gain_weight,
        variance_weight=args.variance_weight,
        extract_rate_weight=args.extract_rate_weight,
        eligible_fraction_weight=args.eligible_fraction_weight,
    )
    pair_df, best = compute_pair_rankings(prompt_stats, args.low_temperatures, args.high_temperatures, pair_cfg)
    pair_df.to_csv(output_dir / "temperature_pair_ranking.csv", index=False)

    medium_stats_path = None
    if best:
        medium_stats = build_medium_stats_for_pair(
            prompt_stats,
            low_temperature=float(best["T_low"]),
            high_temperature=float(best["T_high"]),
        )
        medium_stats_path = output_dir / "best_pair_medium_stats.csv"
        medium_stats.to_csv(medium_stats_path, index=False)

    best_payload = {
        "best_pair": best,
        "selection_config": config_to_dict(pair_cfg),
        "n_probe": args.n_probe,
        "seed": args.seed,
        "method": args.method,
        "medium_stats_file": None if medium_stats_path is None else medium_stats_path.name,
    }
    write_json(output_dir / "best_pair.json", best_payload)


if __name__ == "__main__":
    main()
