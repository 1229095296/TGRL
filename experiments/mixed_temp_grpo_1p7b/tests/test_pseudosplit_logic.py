import pandas as pd

from experiments.mixed_temp_grpo_1p7b.scripts.analysis_helpers import compute_proxy_delta_table
from experiments.mixed_temp_grpo_1p7b.scripts.analyze_gain_carrier import apply_single_high_pseudosplit


def test_single_high_pseudosplit_creates_control_and_explore_arms():
    frame = pd.DataFrame(
        {
            "checkpoint": ["ckpt"] * 8,
            "method": ["single_high"] * 8,
            "split": ["val"] * 8,
            "seed": [1] * 8,
            "index": [0] * 8,
            "sample_idx_within_prompt": list(range(8)),
            "reward": [0, 0, 1, 1, 1, 1, 0, 1],
        }
    )
    split_df = apply_single_high_pseudosplit(frame, control_count=2)
    assert split_df["arm"].tolist()[:2] == ["control", "control"]
    assert split_df["arm"].tolist()[2:] == ["explore"] * 6


def test_single_high_proxy_delta_matches_first_b_vs_last_e_split():
    frame = pd.DataFrame(
        {
            "checkpoint": ["ckpt"] * 8,
            "method": ["single_high"] * 8,
            "split": ["val"] * 8,
            "seed": [1] * 8,
            "index": [7] * 8,
            "sample_idx_within_prompt": list(range(8)),
            "reward": [0, 0, 1, 1, 1, 1, 1, 0],
        }
    )
    split_df = apply_single_high_pseudosplit(frame, control_count=2)
    proxy = compute_proxy_delta_table(split_df, group_cols=["checkpoint", "method", "split", "seed", "index"])
    assert len(proxy) == 1
    assert proxy.iloc[0]["reward_mean_control"] == 0.0
    assert proxy.iloc[0]["reward_mean_explore"] == 5.0 / 6.0
    assert proxy.iloc[0]["hat_delta"] == 5.0 / 6.0
