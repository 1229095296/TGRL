from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_GROUP_COLS = ["checkpoint", "method", "split", "seed", "index"]


def load_table(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    if file_path.suffix == ".jsonl":
        return pd.read_json(file_path, lines=True)
    if file_path.suffix == ".parquet":
        return pd.read_parquet(file_path)
    if file_path.suffix == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return pd.DataFrame(payload)
    return pd.read_csv(file_path)


def write_table(df: pd.DataFrame, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".parquet":
        try:
            df.to_parquet(output_path, index=False)
            return output_path
        except Exception:
            fallback_path = output_path.with_suffix(".csv")
            df.to_csv(fallback_path, index=False)
            return fallback_path
    elif output_path.suffix == ".jsonl":
        df.to_json(output_path, orient="records", lines=True, force_ascii=True)
    elif output_path.suffix == ".json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(df.to_dict(orient="records"), f, indent=2, ensure_ascii=True)
    else:
        df.to_csv(output_path, index=False)
    return output_path


def normalize_common_columns(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    rename_map = {}
    if "prompt_id" in frame.columns and "index" not in frame.columns:
        rename_map["prompt_id"] = "index"
    if "response_text" in frame.columns and "response" not in frame.columns:
        rename_map["response_text"] = "response"
    if "answer" in frame.columns and "ground_truth" not in frame.columns:
        rename_map["answer"] = "ground_truth"
    if "temperature" in frame.columns and "temperature_used" not in frame.columns:
        rename_map["temperature"] = "temperature_used"
    if rename_map:
        frame = frame.rename(columns=rename_map)

    defaults = {
        "checkpoint": "unknown",
        "method": "unknown",
        "split": "unknown",
        "seed": -1,
    }
    for col, default in defaults.items():
        if col not in frame.columns:
            frame[col] = default

    if "index" not in frame.columns:
        frame["index"] = np.arange(len(frame), dtype=np.int64)
    if "reward" in frame.columns:
        frame["reward"] = pd.to_numeric(frame["reward"], errors="coerce").fillna(0.0)
    return frame


def parse_group_cols(group_cols: str | list[str] | None) -> list[str]:
    if group_cols is None:
        return list(DEFAULT_GROUP_COLS)
    if isinstance(group_cols, list):
        return group_cols
    return [part.strip() for part in str(group_cols).split(",") if part.strip()]


def ensure_order_column(df: pd.DataFrame, sample_col_candidates: list[str] | None = None) -> pd.DataFrame:
    frame = df.copy()
    candidates = sample_col_candidates or [
        "sample_idx_within_prompt",
        "sample_idx",
        "sample_id",
        "rollout_idx",
    ]
    for col in candidates:
        if col in frame.columns:
            frame["_sample_order"] = pd.to_numeric(frame[col], errors="coerce")
            return frame

    group_cols = [col for col in DEFAULT_GROUP_COLS if col in frame.columns]
    if not group_cols:
        frame["_sample_order"] = np.arange(len(frame), dtype=np.int64)
        return frame
    frame["_sample_order"] = frame.groupby(group_cols, dropna=False).cumcount()
    return frame


def infer_arm_from_temperatures(
    df: pd.DataFrame,
    low_temperature: float | None,
    high_temperature: float | None,
    temperature_col: str = "temperature_used",
    atol: float = 1e-6,
) -> pd.DataFrame:
    frame = df.copy()
    if "arm" in frame.columns:
        return frame
    if temperature_col not in frame.columns:
        raise ValueError("Cannot infer arm labels without arm column or temperature column.")
    if low_temperature is None or high_temperature is None:
        raise ValueError("low_temperature and high_temperature are required to infer arm labels.")

    temps = pd.to_numeric(frame[temperature_col], errors="coerce")
    arm = np.where(np.isclose(temps, low_temperature, atol=atol), "control", None)
    arm = np.where(np.isclose(temps, high_temperature, atol=atol), "explore", arm)
    frame["arm"] = arm
    return frame


def empirical_pass_at_k(rewards: list[float] | np.ndarray, k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    arr = np.asarray(rewards, dtype=float)
    if arr.size == 0:
        return float("nan")
    actual_k = min(int(k), int(arr.size))
    return float(np.max(arr[:actual_k]) > 0.0)


def build_passk_summary(
    df: pd.DataFrame,
    ks: list[int],
    group_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = normalize_common_columns(df)
    frame = ensure_order_column(frame)
    working_group_cols = [col for col in group_cols if col in frame.columns]

    prompt_level_rows: list[dict[str, Any]] = []
    prompt_group_cols = working_group_cols
    for group_key, group_df in frame.groupby(prompt_group_cols, dropna=False):
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        row = {col: value for col, value in zip(prompt_group_cols, key_values)}
        group_df = group_df.sort_values("_sample_order")
        rewards = group_df["reward"].astype(float).tolist()
        row["num_samples"] = len(rewards)
        row["reward_mean"] = float(np.mean(rewards)) if rewards else float("nan")
        row["reward_std"] = float(np.std(rewards, ddof=0)) if rewards else float("nan")
        for k in ks:
            row[f"pass_at_{k}"] = empirical_pass_at_k(rewards, k)
        prompt_level_rows.append(row)

    prompt_level = pd.DataFrame(prompt_level_rows)
    summary_group_cols = [col for col in working_group_cols if col != "index"]
    summary = (
        prompt_level.groupby(summary_group_cols, dropna=False)
        .agg(
            num_prompts=("index", "size"),
            mean_num_samples=("num_samples", "mean"),
            reward_mean=("reward_mean", "mean"),
            reward_std=("reward_mean", "std"),
            **{f"pass_at_{k}": (f"pass_at_{k}", "mean") for k in ks},
        )
        .reset_index()
    )
    if "extraction_success" in frame.columns:
        ext = (
            frame.groupby(summary_group_cols, dropna=False)["extraction_success"]
            .mean()
            .reset_index(name="extraction_success_rate")
        )
        summary = summary.merge(ext, on=summary_group_cols, how="left")
    summary["reward_std"] = summary["reward_std"].fillna(0.0)
    return prompt_level, summary


def compute_true_delta_from_mc(
    df: pd.DataFrame,
    low_temperature: float,
    high_temperature: float,
    temperature_col: str = "temperature_used",
    group_cols: list[str] | None = None,
    atol: float = 1e-6,
) -> pd.DataFrame:
    frame = normalize_common_columns(df)
    if temperature_col not in frame.columns:
        raise ValueError(f"Missing temperature column: {temperature_col}")

    group_cols = [col for col in parse_group_cols(group_cols) if col in frame.columns]
    if "index" not in group_cols:
        group_cols.append("index")

    frame["_temperature_bucket"] = np.where(
        np.isclose(pd.to_numeric(frame[temperature_col], errors="coerce"), low_temperature, atol=atol),
        "low",
        np.where(
            np.isclose(pd.to_numeric(frame[temperature_col], errors="coerce"), high_temperature, atol=atol),
            "high",
            None,
        ),
    )
    frame = frame[frame["_temperature_bucket"].isin(["low", "high"])].copy()
    if frame.empty:
        raise ValueError("No MC responses matched low/high temperature buckets.")

    grouped = (
        frame.groupby(group_cols + ["_temperature_bucket"], dropna=False)
        .agg(
            reward_mean=("reward", "mean"),
            reward_std=("reward", "std"),
            count=("reward", "size"),
        )
        .reset_index()
    )
    pivot = grouped.pivot_table(
        index=group_cols,
        columns="_temperature_bucket",
        values=["reward_mean", "reward_std", "count"],
        aggfunc="first",
    ).reset_index()
    pivot.columns = [
        "_".join([str(part) for part in col if part]).rstrip("_") if isinstance(col, tuple) else str(col)
        for col in pivot.columns
    ]
    for base_col in ["reward_mean_low", "reward_mean_high", "reward_std_low", "reward_std_high", "count_low", "count_high"]:
        if base_col not in pivot.columns:
            pivot[base_col] = np.nan
    pivot["reward_std_low"] = pivot["reward_std_low"].fillna(0.0)
    pivot["reward_std_high"] = pivot["reward_std_high"].fillna(0.0)
    pivot["true_delta"] = pivot["reward_mean_high"] - pivot["reward_mean_low"]
    pivot["true_delta_sign"] = np.sign(pivot["true_delta"]).astype(float)
    return pivot


def compute_proxy_delta_table(
    df: pd.DataFrame,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    frame = normalize_common_columns(df)
    required = {"arm", "reward", "index"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns for proxy delta table: {sorted(missing)}")

    group_cols = [col for col in parse_group_cols(group_cols) if col in frame.columns]
    if "index" not in group_cols:
        group_cols.append("index")

    grouped = (
        frame.groupby(group_cols + ["arm"], dropna=False)
        .agg(
            reward_mean=("reward", "mean"),
            reward_std=("reward", "std"),
            sample_count=("reward", "size"),
        )
        .reset_index()
    )
    pivot = grouped.pivot_table(
        index=group_cols,
        columns="arm",
        values=["reward_mean", "reward_std", "sample_count"],
        aggfunc="first",
    ).reset_index()
    pivot.columns = [
        "_".join([str(part) for part in col if part]).rstrip("_") if isinstance(col, tuple) else str(col)
        for col in pivot.columns
    ]
    for col in [
        "reward_mean_control",
        "reward_mean_explore",
        "reward_std_control",
        "reward_std_explore",
        "sample_count_control",
        "sample_count_explore",
    ]:
        if col not in pivot.columns:
            pivot[col] = np.nan
    pivot["reward_std_control"] = pivot["reward_std_control"].fillna(0.0)
    pivot["reward_std_explore"] = pivot["reward_std_explore"].fillna(0.0)
    pivot["hat_delta"] = pivot["reward_mean_explore"] - pivot["reward_mean_control"]
    pivot["hat_delta_sign"] = np.sign(pivot["hat_delta"]).astype(float)
    return pivot


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return float("nan")
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = pd.Series(np.asarray(x, dtype=float)).rank(method="average").to_numpy()
    y = pd.Series(np.asarray(y, dtype=float)).rank(method="average").to_numpy()
    return safe_pearson(x, y)


def binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    mask = np.isfinite(labels) & np.isfinite(scores)
    labels = labels[mask]
    scores = scores[mask]
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    rank_sum_pos = float(ranks[labels == 1].sum())
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def compute_gain_identification_summary(
    merged_prompt_df: pd.DataFrame,
    group_cols: list[str] | None = None,
    epsilon: float = 1e-12,
) -> pd.DataFrame:
    frame = merged_prompt_df.copy()
    required = {"hat_delta", "true_delta"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns for gain identification summary: {sorted(missing)}")

    group_cols = [col for col in parse_group_cols(group_cols) if col in frame.columns and col != "index"]
    rows: list[dict[str, Any]] = []
    for group_key, group_df in frame.groupby(group_cols, dropna=False):
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        row = {col: value for col, value in zip(group_cols, key_values)}
        valid = group_df[np.isfinite(group_df["hat_delta"]) & np.isfinite(group_df["true_delta"])].copy()
        row["num_prompts"] = int(len(valid))
        if valid.empty:
            rows.append(row)
            continue

        labels = (valid["true_delta"].to_numpy(dtype=float) > epsilon).astype(int)
        true_sign = np.sign(valid["true_delta"].to_numpy(dtype=float))
        pred_sign = np.sign(valid["hat_delta"].to_numpy(dtype=float))
        nonzero_mask = np.abs(valid["true_delta"].to_numpy(dtype=float)) > epsilon

        row["pearson_hat_vs_true"] = safe_pearson(valid["hat_delta"].to_numpy(), valid["true_delta"].to_numpy())
        row["spearman_hat_vs_true"] = safe_spearman(valid["hat_delta"].to_numpy(), valid["true_delta"].to_numpy())
        row["auroc_true_delta_positive"] = binary_auroc(labels, valid["hat_delta"].to_numpy(dtype=float))
        row["sign_accuracy"] = (
            float((pred_sign[nonzero_mask] == true_sign[nonzero_mask]).mean()) if nonzero_mask.any() else float("nan")
        )

        pos_mask = valid["true_delta"].to_numpy(dtype=float) > epsilon
        neg_mask = valid["true_delta"].to_numpy(dtype=float) < -epsilon
        row["positive_hit_rate"] = (
            float((valid.loc[pos_mask, "hat_delta"].to_numpy(dtype=float) > 0.0).mean()) if pos_mask.any() else float("nan")
        )
        row["negative_hit_rate"] = (
            float((valid.loc[neg_mask, "hat_delta"].to_numpy(dtype=float) < 0.0).mean()) if neg_mask.any() else float("nan")
        )
        row["mean_true_delta"] = float(valid["true_delta"].mean())
        row["mean_hat_delta"] = float(valid["hat_delta"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def build_carrier_table(
    df: pd.DataFrame,
    carrier_col: str = "advantage_scalar",
    epsilon: float = 1e-6,
) -> pd.DataFrame:
    frame = normalize_common_columns(df)
    if "arm" not in frame.columns:
        raise ValueError("Carrier table requires an arm column.")

    if carrier_col in frame.columns:
        frame["carrier"] = pd.to_numeric(frame[carrier_col], errors="coerce")
        explore_only = frame[frame["arm"] == "explore"].copy()
        return explore_only

    group_cols = [col for col in DEFAULT_GROUP_COLS if col in frame.columns]
    group_stats = frame.groupby(group_cols, dropna=False)["reward"].agg(["mean", "std"]).reset_index()
    group_stats = group_stats.rename(columns={"mean": "group_mean", "std": "group_std"})
    group_stats["group_std"] = group_stats["group_std"].fillna(0.0)
    merged = frame.merge(group_stats, on=group_cols, how="left")
    merged["carrier"] = (merged["reward"] - merged["group_mean"]) / (merged["group_std"] + epsilon)
    return merged[merged["arm"] == "explore"].copy()


def compute_carrier_expectation_summary(
    carrier_df: pd.DataFrame,
    group_cols: list[str] | None = None,
    epsilon: float = 1e-12,
) -> pd.DataFrame:
    frame = carrier_df.copy()
    required = {"carrier", "true_delta"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns for carrier expectation summary: {sorted(missing)}")

    group_cols = [col for col in parse_group_cols(group_cols) if col in frame.columns and col != "index"]
    rows: list[dict[str, Any]] = []
    for group_key, group_df in frame.groupby(group_cols, dropna=False):
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        row = {col: value for col, value in zip(group_cols, key_values)}
        valid = group_df[np.isfinite(group_df["carrier"]) & np.isfinite(group_df["true_delta"])].copy()
        row["num_explore_samples"] = int(len(valid))
        if valid.empty:
            rows.append(row)
            continue

        pos_mask = valid["true_delta"].to_numpy(dtype=float) > epsilon
        neg_mask = valid["true_delta"].to_numpy(dtype=float) < -epsilon
        row["carrier_mean_true_positive"] = float(valid.loc[pos_mask, "carrier"].mean()) if pos_mask.any() else float("nan")
        row["carrier_mean_true_negative"] = float(valid.loc[neg_mask, "carrier"].mean()) if neg_mask.any() else float("nan")
        row["carrier_std_true_positive"] = float(valid.loc[pos_mask, "carrier"].std(ddof=0)) if pos_mask.any() else float("nan")
        row["carrier_std_true_negative"] = float(valid.loc[neg_mask, "carrier"].std(ddof=0)) if neg_mask.any() else float("nan")
        row["carrier_gap"] = (
            row["carrier_mean_true_positive"] - row["carrier_mean_true_negative"]
            if not (math.isnan(row["carrier_mean_true_positive"]) or math.isnan(row["carrier_mean_true_negative"]))
            else float("nan")
        )
        rows.append(row)
    return pd.DataFrame(rows)


def merge_final_summary(
    eval_summary: pd.DataFrame | None = None,
    gain_summary: pd.DataFrame | None = None,
    carrier_summary: pd.DataFrame | None = None,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    group_cols = parse_group_cols(group_cols)
    frames = [frame for frame in [eval_summary, gain_summary, carrier_summary] if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame()

    merged = None
    for frame in frames:
        available = [col for col in group_cols if col in frame.columns]
        if merged is None:
            merged = frame.copy()
            continue
        merged = merged.merge(frame, on=available, how="outer")
    if merged is None:
        return pd.DataFrame()

    suffix_bases = sorted({col[:-2] for col in merged.columns if col.endswith("_x") and f"{col[:-2]}_y" in merged.columns})
    for base in suffix_bases:
        merged[base] = merged[f"{base}_x"].combine_first(merged[f"{base}_y"])
        merged = merged.drop(columns=[f"{base}_x", f"{base}_y"])

    if "index" in merged.columns and "index" not in group_cols:
        merged = merged.drop(columns=["index"])
    return merged


def format_report_section(title: str, bullets: list[str]) -> str:
    lines = [f"## {title}"]
    for bullet in bullets:
        lines.append(f"- {bullet}")
    lines.append("")
    return "\n".join(lines)
