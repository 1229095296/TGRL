from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation


_ANSWER_CLIP_CHARS = 512


def extract_final_answer(text: str, method: str = "strict") -> str | None:
    if not text:
        return None

    tail = text[-_ANSWER_CLIP_CHARS:] if len(text) > _ANSWER_CLIP_CHARS else text
    if method == "strict":
        matches = re.findall(r"####\s*([\-]?[0-9\.,]+)", tail)
        if not matches:
            return None
        return _normalize_number(matches[-1])

    if method == "flexible":
        matches = re.findall(r"([\-]?[0-9][0-9\.,]*)", tail)
        for match in reversed(matches):
            normalized = _normalize_number(match)
            if normalized is not None:
                return normalized
        return None

    raise ValueError(f"Unsupported extraction method: {method}")


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    method: str = "strict",
    format_score: float = 0.0,
    score: float = 1.0,
    return_dict: bool = True,
):
    """Rule-based GSM8K reward shared by training and offline evaluation."""
    extracted = extract_final_answer(solution_str, method=method)
    normalized_gt = _normalize_number(ground_truth)
    exact_match = extracted is not None and normalized_gt is not None and extracted == normalized_gt
    reward = score if exact_match else (0.0 if extracted is None else format_score)

    result = {
        "score": float(reward),
        "exact_match": bool(exact_match),
        "extracted_answer": extracted,
        "ground_truth_normalized": normalized_gt,
        "extraction_success": bool(extracted is not None),
        "extraction_failure": bool(extracted is None),
        "data_source": data_source,
    }

    if extra_info and "index" in extra_info:
        result["prompt_index"] = extra_info["index"]

    if return_dict:
        return result
    return result["score"]


def _normalize_number(text: str | None) -> str | None:
    if text is None:
        return None
    normalized = str(text).strip().replace(",", "").replace("$", "")
    normalized = normalized.rstrip(".")
    if normalized in {"", "-", "."}:
        return None
    try:
        value = Decimal(normalized)
    except InvalidOperation:
        return None
    if value == value.to_integral():
        return str(value.quantize(Decimal("1")))
    return format(value.normalize(), "f").rstrip("0").rstrip(".")
