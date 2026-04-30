import pandas as pd

from experiments.mixed_temp_grpo_1p7b.scripts.analysis_helpers import (
    compute_carrier_expectation_summary,
    compute_gain_identification_summary,
    compute_true_delta_from_mc,
)
from experiments.mixed_temp_grpo_1p7b.scripts.analyze_gain_carrier import (
    apply_single_high_pseudosplit,
    compute_exploration_carrier_table,
    compute_prompt_gain_table,
    summarize_gain_and_carrier,
)


def test_apply_single_high_pseudosplit():
    df = pd.DataFrame(
        {
            "sample_idx_within_prompt": [0, 1, 2, 0, 1, 2],
            "reward": [0, 1, 1, 0, 0, 1],
        }
    )
    split_df = apply_single_high_pseudosplit(df, control_count=1)
    assert split_df["arm"].tolist() == ["control", "explore", "explore", "control", "explore", "explore"]


def test_compute_prompt_gain_and_carrier_stats():
    df = pd.DataFrame(
        {
            "split": ["val"] * 6,
            "index": [0, 0, 0, 1, 1, 1],
            "arm": ["control", "explore", "explore", "control", "explore", "explore"],
            "reward": [0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
        }
    )
    gain_df = compute_prompt_gain_table(df)
    carrier_df = compute_exploration_carrier_table(df)
    summary = summarize_gain_and_carrier(gain_df, carrier_df)

    assert set(gain_df.columns) >= {"control", "explore", "exploration_gain"}
    assert "carrier" in carrier_df.columns
    assert summary["num_prompt_groups"] == 2
    assert summary["positive_gain_fraction"] >= 0.0


def test_true_delta_and_gain_identification_summary():
    mc = pd.DataFrame(
        {
            "checkpoint": ["ckpt"] * 8,
            "method": ["mixed_grouped"] * 8,
            "split": ["val"] * 8,
            "seed": [1] * 8,
            "index": [0, 0, 0, 0, 1, 1, 1, 1],
            "temperature_used": [0.3, 0.3, 1.2, 1.2, 0.3, 0.3, 1.2, 1.2],
            "reward": [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        }
    )
    true_delta = compute_true_delta_from_mc(
        mc,
        low_temperature=0.3,
        high_temperature=1.2,
        group_cols=["checkpoint", "method", "split", "seed", "index"],
    )
    proxy = pd.DataFrame(
        {
            "checkpoint": ["ckpt", "ckpt"],
            "method": ["mixed_grouped", "mixed_grouped"],
            "split": ["val", "val"],
            "seed": [1, 1],
            "index": [0, 1],
            "hat_delta": [1.0, -1.0],
        }
    )
    merged = proxy.merge(true_delta, on=["checkpoint", "method", "split", "seed", "index"], how="left")
    summary = compute_gain_identification_summary(merged, group_cols=["checkpoint", "method", "split", "seed", "index"])
    assert len(summary) == 1
    assert summary.iloc[0]["positive_hit_rate"] == 1.0
    assert summary.iloc[0]["negative_hit_rate"] == 1.0


def test_carrier_expectation_summary():
    carrier_df = pd.DataFrame(
        {
            "checkpoint": ["ckpt"] * 4,
            "method": ["mixed_grouped"] * 4,
            "split": ["val"] * 4,
            "seed": [1] * 4,
            "index": [0, 0, 1, 1],
            "carrier": [1.0, 1.5, -0.5, -1.0],
            "true_delta": [0.5, 0.5, -0.25, -0.25],
        }
    )
    summary = compute_carrier_expectation_summary(
        carrier_df,
        group_cols=["checkpoint", "method", "split", "seed", "index"],
    )
    assert len(summary) == 1
    assert summary.iloc[0]["carrier_mean_true_positive"] == 1.25
    assert summary.iloc[0]["carrier_mean_true_negative"] == -0.75
