import pandas as pd

from experiments.mixed_temp_grpo_1p7b.scripts.prepare_gsm8k_medium import (
    MediumFilterConfig,
    build_verl_row,
    normalize_input_dataframe,
    select_medium_rows,
)


def test_normalize_input_dataframe_from_raw_gsm8k():
    raw = pd.DataFrame(
        [
            {"question": "What is 1+1?", "answer": "Reasoning #### 2"},
            {"question": "What is 2+2?", "answer": "Reasoning #### 4"},
        ]
    )
    normalized = normalize_input_dataframe(raw, split="train")
    assert list(normalized.columns) == ["data_source", "prompt", "ability", "reward_model", "extra_info"]
    assert normalized.iloc[0]["reward_model"]["ground_truth"] == "2"
    assert normalized.iloc[1]["extra_info"]["index"] == 1


def test_select_medium_rows_uses_stats_thresholds():
    rows = [
        build_verl_row("q1", "a #### 2", "train", 0),
        build_verl_row("q2", "a #### 4", "train", 1),
    ]
    df = pd.DataFrame(rows)
    df["pass_low"] = [0.1, 0.95]
    df["pass_high"] = [0.4, 0.98]
    df["reward_variance"] = [0.2, 0.3]
    df["extract_rate"] = [1.0, 1.0]

    filtered = select_medium_rows(df, MediumFilterConfig())
    assert filtered["medium_keep"].tolist() == [True, False]


def test_select_medium_rows_fallback_to_full_without_stats():
    rows = [build_verl_row("q1", "a #### 2", "train", 0)]
    df = pd.DataFrame(rows)
    df["pass_low"] = [pd.NA]
    df["pass_high"] = [pd.NA]
    df["reward_variance"] = [pd.NA]
    df["extract_rate"] = [pd.NA]

    filtered = select_medium_rows(df, MediumFilterConfig())
    assert filtered["medium_keep"].tolist() == [True]
    assert filtered["medium_reason"].tolist() == ["fallback_full"]
