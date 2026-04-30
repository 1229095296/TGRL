from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.mixed_temp_grpo_1p7b.scripts.analysis_helpers import (
    build_carrier_table,
    compute_carrier_expectation_summary,
    compute_gain_identification_summary,
    compute_proxy_delta_table,
    compute_true_delta_from_mc,
    infer_arm_from_temperatures,
    load_table,
    merge_final_summary,
    normalize_common_columns,
    parse_group_cols,
    write_table,
)


def load_response_table(path: str) -> pd.DataFrame:
    return load_table(path)


def apply_single_high_pseudosplit(
    df: pd.DataFrame,
    control_count: int,
    sample_index_col: str = "sample_idx_within_prompt",
) -> pd.DataFrame:
    frame = df.copy()
    if sample_index_col not in frame.columns:
        raise ValueError(f"Missing sample index column: {sample_index_col}")
    frame["arm"] = frame[sample_index_col].apply(lambda idx: "control" if int(idx) < control_count else "explore")
    return frame


def compute_prompt_gain_table(df: pd.DataFrame) -> pd.DataFrame:
    proxy = compute_proxy_delta_table(df, group_cols=["split", "index"])
    return proxy.rename(
        columns={
            "reward_mean_control": "control",
            "reward_mean_explore": "explore",
            "hat_delta": "exploration_gain",
        }
    )[["split", "index", "control", "explore", "exploration_gain"]]


def compute_exploration_carrier_table(df: pd.DataFrame, epsilon: float = 1e-6) -> pd.DataFrame:
    return build_carrier_table(df, carrier_col="carrier", epsilon=epsilon)


def summarize_gain_and_carrier(prompt_gain_df: pd.DataFrame, carrier_df: pd.DataFrame) -> dict[str, Any]:
    payload = {
        "num_prompt_groups": int(len(prompt_gain_df)),
        "mean_exploration_gain": float(prompt_gain_df["exploration_gain"].mean()) if not prompt_gain_df.empty else 0.0,
        "gain_std": float(prompt_gain_df["exploration_gain"].std(ddof=0)) if not prompt_gain_df.empty else 0.0,
        "positive_gain_fraction": float((prompt_gain_df["exploration_gain"] > 0).mean()) if not prompt_gain_df.empty else 0.0,
        "mean_explore_carrier": float(carrier_df["carrier"].mean()) if not carrier_df.empty else 0.0,
        "positive_carrier_fraction": float((carrier_df["carrier"] > 0).mean()) if not carrier_df.empty else 0.0,
    }
    return payload


def maybe_infer_arm_labels(
    responses: pd.DataFrame,
    low_temperature: float | None,
    high_temperature: float | None,
    pseudo_split_control_count: int | None,
    sample_index_col: str,
) -> pd.DataFrame:
    if pseudo_split_control_count is not None:
        return apply_single_high_pseudosplit(
            responses,
            control_count=pseudo_split_control_count,
            sample_index_col=sample_index_col,
        )
    if "arm" in responses.columns:
        return responses
    return infer_arm_from_temperatures(
        responses,
        low_temperature=low_temperature,
        high_temperature=high_temperature,
    )


def build_true_delta_table(
    true_delta_table_path: str | None,
    true_delta_responses_path: str | None,
    low_temperature: float | None,
    high_temperature: float | None,
    group_cols: list[str],
) -> pd.DataFrame | None:
    if true_delta_table_path:
        frame = normalize_common_columns(load_table(true_delta_table_path))
        if "true_delta" not in frame.columns:
            raise ValueError("Provided true-delta table must contain a true_delta column.")
        if "true_delta_sign" not in frame.columns:
            frame["true_delta_sign"] = np.sign(pd.to_numeric(frame["true_delta"], errors="coerce").fillna(0.0))
        return frame

    if true_delta_responses_path:
        if low_temperature is None or high_temperature is None:
            raise ValueError("low/high temperatures are required to compute true_delta from MC responses.")
        mc_responses = normalize_common_columns(load_table(true_delta_responses_path))
        return compute_true_delta_from_mc(
            mc_responses,
            low_temperature=low_temperature,
            high_temperature=high_temperature,
            group_cols=group_cols,
        )
    return None


def build_report_payload(
    final_summary: pd.DataFrame,
    gain_identification: pd.DataFrame,
    carrier_expectation: pd.DataFrame,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "rows_final_summary": int(len(final_summary)),
        "rows_gain_identification": int(len(gain_identification)),
        "rows_carrier_expectation": int(len(carrier_expectation)),
    }
    if not final_summary.empty:
        sort_col = "pass_at_1" if "pass_at_1" in final_summary.columns else final_summary.columns[-1]
        best_row = final_summary.sort_values(sort_col, ascending=False).iloc[0]
        payload["best_final_summary_row"] = best_row.to_dict()
    if not gain_identification.empty:
        best_gain = gain_identification.sort_values("pearson_hat_vs_true", ascending=False).iloc[0]
        payload["best_gain_row"] = best_gain.to_dict()
    if not carrier_expectation.empty:
        best_carrier = carrier_expectation.sort_values("carrier_gap", ascending=False).iloc[0]
        payload["best_carrier_row"] = best_carrier.to_dict()
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze prompt-level gain and exploration carrier statistics.")
    parser.add_argument("--responses", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pseudo-split-control-count", type=int, default=None)
    parser.add_argument("--sample-index-col", default="sample_idx_within_prompt")
    parser.add_argument("--true-delta-responses", default=None, help="Offline MC responses used to estimate true_delta.")
    parser.add_argument("--true-delta-table", default=None, help="Precomputed prompt-level table with true_delta.")
    parser.add_argument("--eval-summary", default=None, help="checkpoint_eval_summary.csv from eval_checkpoints.py")
    parser.add_argument("--carrier-col", default="advantage_scalar", help="Column used as exploration carrier.")
    parser.add_argument("--low-temperature", type=float, default=None)
    parser.add_argument("--high-temperature", type=float, default=None)
    parser.add_argument("--group-cols", default="checkpoint,method,split,seed,index")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    group_cols = parse_group_cols(args.group_cols)
    responses = normalize_common_columns(load_response_table(args.responses))
    responses = maybe_infer_arm_labels(
        responses,
        low_temperature=args.low_temperature,
        high_temperature=args.high_temperature,
        pseudo_split_control_count=args.pseudo_split_control_count,
        sample_index_col=args.sample_index_col,
    )

    proxy_delta = compute_proxy_delta_table(responses, group_cols=group_cols)
    carrier_by_sample = build_carrier_table(responses, carrier_col=args.carrier_col)
    true_delta = build_true_delta_table(
        true_delta_table_path=args.true_delta_table,
        true_delta_responses_path=args.true_delta_responses,
        low_temperature=args.low_temperature,
        high_temperature=args.high_temperature,
        group_cols=group_cols,
    )

    if true_delta is not None:
        join_cols = [col for col in group_cols if col in proxy_delta.columns and col in true_delta.columns]
        prompt_delta_join = proxy_delta.merge(true_delta, on=join_cols, how="left")
        carrier_join_cols = [col for col in group_cols if col in carrier_by_sample.columns and col in true_delta.columns]
        carrier_by_sample = carrier_by_sample.merge(
            true_delta[carrier_join_cols + ["true_delta", "true_delta_sign"]],
            on=carrier_join_cols,
            how="left",
        )
        gain_identification = compute_gain_identification_summary(prompt_delta_join, group_cols=group_cols)
        carrier_expectation = compute_carrier_expectation_summary(carrier_by_sample, group_cols=group_cols)
    else:
        prompt_delta_join = proxy_delta.copy()
        gain_identification = pd.DataFrame()
        carrier_expectation = pd.DataFrame()

    eval_summary = normalize_common_columns(load_table(args.eval_summary)) if args.eval_summary else pd.DataFrame()
    final_summary = merge_final_summary(
        eval_summary=eval_summary if not eval_summary.empty else None,
        gain_summary=gain_identification if not gain_identification.empty else None,
        carrier_summary=carrier_expectation if not carrier_expectation.empty else None,
        group_cols=[col for col in group_cols if col != "index"],
    )

    write_table(proxy_delta, output_dir / "proxy_delta_by_prompt.csv")
    write_table(carrier_by_sample, output_dir / "carrier_by_sample.csv")
    if true_delta is not None:
        write_table(true_delta, output_dir / "true_delta_mc.csv")
        write_table(prompt_delta_join, output_dir / "prompt_delta_join.csv")
    write_table(gain_identification, output_dir / "gain_identification.csv")
    write_table(carrier_expectation, output_dir / "carrier_expectation.csv")
    write_table(final_summary, output_dir / "final_summary.csv")

    prompt_gain = compute_prompt_gain_table(responses)
    carrier_basic = compute_exploration_carrier_table(responses)
    summary = summarize_gain_and_carrier(prompt_gain, carrier_basic)
    with open(output_dir / "gain_carrier_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)

    report_payload = build_report_payload(final_summary, gain_identification, carrier_expectation)
    with open(output_dir / "analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2, ensure_ascii=True)


if __name__ == "__main__":
    main()
