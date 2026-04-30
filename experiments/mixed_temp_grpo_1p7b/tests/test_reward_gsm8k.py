from experiments.mixed_temp_grpo_1p7b.reward.gsm8k_reward import compute_score, extract_final_answer


def test_extract_final_answer_strict():
    text = "We solve it carefully. #### 1,234"
    assert extract_final_answer(text, method="strict") == "1234"


def test_extract_final_answer_flexible():
    text = "Reasoning... answer is 17."
    assert extract_final_answer(text, method="flexible") == "17"


def test_compute_score_reports_extraction_failure():
    result = compute_score(
        data_source="openai/gsm8k",
        solution_str="No final answer marker here",
        ground_truth="42",
        return_dict=True,
    )
    assert result["score"] == 0.0
    assert result["extraction_failure"] is True
    assert result["extraction_success"] is False


def test_compute_score_exact_match():
    result = compute_score(
        data_source="openai/gsm8k",
        solution_str="#### 42",
        ground_truth="42",
        return_dict=True,
    )
    assert result["score"] == 1.0
    assert result["exact_match"] is True
