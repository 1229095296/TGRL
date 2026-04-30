import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd

from experiments.mixed_temp_grpo_1p7b.reward.gsm8k_reward import compute_score


@dataclass
class ProbeThresholdConfig:
    min_pass: float = 0.05
    max_pass: float = 0.8
    min_variance: float = 0.01
    min_extract_rate: float = 0.9
    allow_full_fallback: bool = True


@dataclass
class PairSelectionConfig:
    min_extract_rate: float = 0.9
    min_variance: float = 0.01
    gain_std_weight: float = 1.0
    positive_gain_weight: float = 0.5
    variance_weight: float = 0.25
    extract_rate_weight: float = 1.0
    eligible_fraction_weight: float = 0.5


def load_records_table(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(path)
    if file_path.suffix == ".jsonl":
        return pd.read_json(file_path, lines=True)
    if file_path.suffix == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        if isinstance(payload, dict):
            for key in ("records", "rows", "items"):
                if isinstance(payload.get(key), list):
                    return pd.DataFrame(payload[key])
        raise ValueError("Unsupported JSON payload.")
    if file_path.suffix == ".parquet":
        return pd.read_parquet(file_path)
    return pd.read_csv(file_path)


def maybe_load_hf_gsm8k(
    dataset_name: str,
    config_name: str,
    train_split: str,
    val_split: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        import datasets
    except ImportError as exc:
        raise RuntimeError("datasets is required for --hf-dataset mode.") from exc

    dataset = datasets.load_dataset(dataset_name, config_name)
    train_df = dataset[train_split].to_pandas()
    val_df = dataset[val_split].to_pandas()
    return train_df, val_df


def extract_prompt_text(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        parts = []
        for message in prompt:
            if not isinstance(message, dict):
                parts.append(str(message))
                continue
            role = message.get("role", "unknown")
            content = message.get("content", "")
            parts.append(f"{role}: {content}")
        return "\n".join(parts)
    return str(prompt)


def get_ground_truth(row: pd.Series) -> str:
    reward_model = row.get("reward_model", {})
    if isinstance(reward_model, dict) and reward_model.get("ground_truth") is not None:
        return str(reward_model["ground_truth"])
    if row.get("ground_truth") is not None:
        return str(row["ground_truth"])
    raise ValueError("Missing ground truth in row.")


def get_prompt_index(row: pd.Series, fallback_index: int) -> int:
    extra_info = row.get("extra_info", {})
    if isinstance(extra_info, dict) and extra_info.get("index") is not None:
        return int(extra_info["index"])
    if row.get("index") is not None and not (isinstance(row.get("index"), float) and math.isnan(row["index"])):
        return int(row["index"])
    return int(fallback_index)


class TransformersProbeBackend:
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str | None = None,
        device: str = "auto",
        max_new_tokens: int = 256,
        top_p: float = 1.0,
        top_k: int = -1,
        trust_remote_code: bool = False,
        seed: int = 1,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers and torch are required for the transformers probe backend.") from exc

        self._torch = torch
        tokenizer_path = tokenizer_path or model_path
        resolved_device = device
        if resolved_device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "auto" and resolved_device == "cpu":
            print("[probe] warning: CUDA not available, probe backend is falling back to CPU and will be very slow")
        if resolved_device.startswith("cuda"):
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                load_dtype = torch.bfloat16
            else:
                load_dtype = torch.float16
        else:
            load_dtype = torch.float32

        print(
            f"[probe] loading transformers backend model={model_path} "
            f"tokenizer={tokenizer_path} device={resolved_device} dtype={str(load_dtype).replace('torch.', '')}"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=trust_remote_code)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=load_dtype,
            low_cpu_mem_usage=True,
        )
        self.device = resolved_device
        self.model.to(self.device)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.seed = int(seed)
        print(
            f"[probe] backend ready device={self.device} max_new_tokens={self.max_new_tokens} "
            f"top_p={self.top_p} top_k={self.top_k}"
        )

    def _build_generation_kwargs(self, temperature: float) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "do_sample": True,
            "temperature": float(temperature),
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.top_p is not None and float(self.top_p) < 1.0:
            kwargs["top_p"] = float(self.top_p)
        # vLLM convention uses top_k=-1 to mean "disable top-k"; transformers expects
        # either a positive integer or the argument omitted entirely.
        if self.top_k is not None and int(self.top_k) > 0:
            kwargs["top_k"] = int(self.top_k)
        return kwargs

    def _render_prompt(self, prompt: Any) -> str:
        if hasattr(self.tokenizer, "apply_chat_template") and isinstance(prompt, list):
            try:
                return self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
            except Exception:
                pass
        return extract_prompt_text(prompt)

    def generate_responses(
        self,
        prompts: list[Any],
        temperature: float,
        n_samples: int,
        batch_size: int,
        progress_callback: Callable[[int], None] | None = None,
    ) -> list[list[str]]:
        torch = self._torch
        print(f"[probe] rendering {len(prompts)} prompts on device={self.device}")
        rendered_prompts = [self._render_prompt(prompt) for prompt in prompts]
        all_outputs: list[list[str]] = [[] for _ in rendered_prompts]

        for start in range(0, len(rendered_prompts), batch_size):
            prompt_batch = rendered_prompts[start : start + batch_size]
            tokenized = self.tokenizer(
                prompt_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)
            prompt_len = tokenized["input_ids"].shape[1]
            for sample_idx in range(n_samples):
                sample_seed = self.seed + int(sample_idx) + int(start)
                torch.manual_seed(sample_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(sample_seed)
                generation_kwargs = self._build_generation_kwargs(temperature)
                with torch.no_grad():
                    output_ids = self.model.generate(
                        **tokenized,
                        **generation_kwargs,
                    )
                generated = output_ids[:, prompt_len:]
                decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
                for offset, text in enumerate(decoded):
                    all_outputs[start + offset].append(text)
                if progress_callback is not None:
                    progress_callback(1)
        return all_outputs


def create_probe_backend(
    backend_name: str,
    model_path: str | None,
    tokenizer_path: str | None,
    device: str,
    max_new_tokens: int,
    top_p: float,
    top_k: int,
    trust_remote_code: bool,
    seed: int = 1,
):
    if backend_name == "transformers":
        if not model_path:
            raise ValueError("--probe-model-path is required when --probe-backend=transformers")
        return TransformersProbeBackend(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            device=device,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            trust_remote_code=trust_remote_code,
            seed=seed,
        )
    raise NotImplementedError(f"Unsupported probe backend: {backend_name}")


def probe_dataset_with_backend(
    normalized_df: pd.DataFrame,
    temperatures: Iterable[float],
    n_probe: int,
    backend,
    split: str,
    response_method: str = "strict",
    batch_size: int = 4,
    max_prompts: int | None = None,
    progress_desc: str | None = None,
) -> pd.DataFrame:
    if n_probe < 1:
        raise ValueError(f"n_probe must be >= 1, got {n_probe}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if max_prompts is not None and max_prompts < 1:
        raise ValueError(f"max_prompts must be >= 1 when provided, got {max_prompts}")
    frame = normalized_df.reset_index(drop=True)
    if max_prompts is not None:
        frame = frame.head(max_prompts).copy()
    if frame.empty:
        raise ValueError(f"split={split} has 0 rows after applying probe limits")

    prompts = frame["prompt"].tolist()
    temperatures = [float(temperature) for temperature in temperatures]
    total_batches = math.ceil(len(prompts) / batch_size) if prompts else 0
    total_generate_calls = len(temperatures) * n_probe * total_batches
    records: list[dict[str, Any]] = []
    print(
        f"[probe] split={split} prompts={len(prompts)} "
        f"temperatures={temperatures} n_probe={n_probe} batch_size={batch_size} "
        f"generate_calls={total_generate_calls}"
    )
    progress = None
    try:
        if total_generate_calls > 0:
            try:
                from tqdm.auto import tqdm
            except ImportError:
                tqdm = None
            if tqdm is not None:
                progress = tqdm(
                    total=total_generate_calls,
                    desc=progress_desc or f"probe[{split}]",
                    unit="gen",
                    dynamic_ncols=True,
                )
        for temperature_idx, temperature in enumerate(temperatures, start=1):
            print(
                f"[probe] split={split} temperature={temperature:.3f} "
                f"({temperature_idx}/{len(temperatures)})"
            )
            generated_groups = backend.generate_responses(
                prompts=prompts,
                temperature=float(temperature),
                n_samples=n_probe,
                batch_size=batch_size,
                progress_callback=None if progress is None else progress.update,
            )
            score_progress = None
            try:
                score_total = sum(len(responses) for responses in generated_groups)
                if score_total > 0:
                    try:
                        from tqdm.auto import tqdm
                    except ImportError:
                        tqdm = None
                    if tqdm is not None:
                        score_progress = tqdm(
                            total=score_total,
                            desc=f"score[{split}|T={temperature:.3f}]",
                            unit="resp",
                            dynamic_ncols=True,
                            leave=False,
                        )
                print(
                    f"[probe] scoring split={split} temperature={temperature:.3f} "
                    f"responses={score_total}"
                )
                for row_idx, responses in enumerate(generated_groups):
                    row = frame.iloc[row_idx]
                    prompt_index = get_prompt_index(row, fallback_index=row_idx)
                    ground_truth = get_ground_truth(row)
                    data_source = row.get("data_source", "openai/gsm8k")
                    for probe_idx, response in enumerate(responses):
                        score_payload = compute_score(
                            data_source=data_source,
                            solution_str=response,
                            ground_truth=ground_truth,
                            extra_info={"index": prompt_index},
                            method=response_method,
                            return_dict=True,
                        )
                        records.append(
                            {
                                "split": split,
                                "index": prompt_index,
                                "temperature": float(temperature),
                                "probe_idx": int(probe_idx),
                                "data_source": data_source,
                                "ground_truth": ground_truth,
                                "response": response,
                                "reward": float(score_payload["score"]),
                                "score": float(score_payload["score"]),
                                "extraction_success": bool(score_payload["extraction_success"]),
                                "extraction_failure": bool(score_payload["extraction_failure"]),
                                "exact_match": bool(score_payload["exact_match"]),
                            }
                        )
                        if score_progress is not None:
                            score_progress.update(1)
            finally:
                if score_progress is not None:
                    score_progress.close()
    finally:
        if progress is not None:
            progress.close()
    print(f"[probe] split={split} completed records={len(records)}")
    return pd.DataFrame(records)


def summarize_probe_records(df: pd.DataFrame) -> pd.DataFrame:
    required = {"split", "index", "temperature", "reward", "extraction_success"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns for probe summary: {sorted(missing)}")

    summary = (
        df.groupby(["split", "index", "temperature"], dropna=False)
        .agg(
            reward_mean=("reward", "mean"),
            reward_variance=("reward", "var"),
            pass_rate=("reward", "mean"),
            extract_rate=("extraction_success", "mean"),
            samples=("reward", "size"),
        )
        .reset_index()
    )
    summary["reward_variance"] = summary["reward_variance"].fillna(0.0)
    return summary


def compute_pair_rankings(
    prompt_stats: pd.DataFrame,
    low_candidates: list[float],
    high_candidates: list[float],
    cfg: PairSelectionConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for low in low_candidates:
        low_df = prompt_stats[prompt_stats["temperature"] == low][
            ["split", "index", "pass_rate", "reward_variance", "extract_rate"]
        ].rename(
            columns={
                "pass_rate": "pass_low",
                "reward_variance": "reward_variance_low",
                "extract_rate": "extract_rate_low",
            }
        )
        for high in high_candidates:
            if high <= low:
                continue
            high_df = prompt_stats[prompt_stats["temperature"] == high][
                ["split", "index", "pass_rate", "reward_variance", "extract_rate"]
            ].rename(
                columns={
                    "pass_rate": "pass_high",
                    "reward_variance": "reward_variance_high",
                    "extract_rate": "extract_rate_high",
                }
            )
            merged = low_df.merge(high_df, on=["split", "index"], how="inner")
            if merged.empty:
                continue
            gain = merged["pass_high"] - merged["pass_low"]
            pair_variance = 0.5 * (merged["reward_variance_low"] + merged["reward_variance_high"])
            pair_extract = 0.5 * (merged["extract_rate_low"] + merged["extract_rate_high"])
            eligible = (
                pair_variance.ge(cfg.min_variance) & pair_extract.ge(cfg.min_extract_rate)
            )
            row = {
                "T_low": float(low),
                "T_high": float(high),
                "num_prompts": int(len(merged)),
                "mean_pass_low": float(merged["pass_low"].mean()),
                "mean_pass_high": float(merged["pass_high"].mean()),
                "mean_gain": float(gain.mean()),
                "gain_std": float(gain.std(ddof=0)),
                "positive_gain_frac": float((gain > 0).mean()),
                "mean_extract_rate": float(pair_extract.mean()),
                "mean_reward_variance": float(pair_variance.mean()),
                "eligible_prompt_fraction": float(eligible.mean()),
            }
            row["selection_score"] = (
                cfg.gain_std_weight * row["gain_std"]
                + cfg.positive_gain_weight * row["positive_gain_frac"]
                + cfg.variance_weight * row["mean_reward_variance"]
                + cfg.eligible_fraction_weight * row["eligible_prompt_fraction"]
            ) * max(1e-8, cfg.extract_rate_weight * row["mean_extract_rate"])
            rows.append(row)

    pair_df = pd.DataFrame(rows)
    if pair_df.empty:
        return pair_df, {}
    pair_df = pair_df.sort_values("selection_score", ascending=False).reset_index(drop=True)
    return pair_df, pair_df.iloc[0].to_dict()


def build_medium_stats_for_pair(
    prompt_stats: pd.DataFrame,
    low_temperature: float,
    high_temperature: float,
) -> pd.DataFrame:
    low_df = prompt_stats[prompt_stats["temperature"] == low_temperature][
        ["split", "index", "pass_rate", "reward_variance", "extract_rate"]
    ].rename(
        columns={
            "pass_rate": "pass_low",
            "reward_variance": "reward_variance_low",
            "extract_rate": "extract_rate_low",
        }
    )
    high_df = prompt_stats[prompt_stats["temperature"] == high_temperature][
        ["split", "index", "pass_rate", "reward_variance", "extract_rate"]
    ].rename(
        columns={
            "pass_rate": "pass_high",
            "reward_variance": "reward_variance_high",
            "extract_rate": "extract_rate_high",
        }
    )
    merged = low_df.merge(high_df, on=["split", "index"], how="outer")
    merged["reward_variance"] = 0.5 * (
        merged["reward_variance_low"].fillna(0.0) + merged["reward_variance_high"].fillna(0.0)
    )
    merged["extract_rate"] = 0.5 * (
        merged["extract_rate_low"].fillna(0.0) + merged["extract_rate_high"].fillna(0.0)
    )
    return merged[["split", "index", "pass_low", "pass_high", "reward_variance", "extract_rate"]]


def apply_medium_filter_from_stats(stats_df: pd.DataFrame, cfg: ProbeThresholdConfig) -> pd.DataFrame:
    frame = stats_df.copy()
    max_pass = pd.concat([frame["pass_low"], frame["pass_high"]], axis=1).max(axis=1)
    keep = (
        max_pass.ge(cfg.min_pass)
        & max_pass.le(cfg.max_pass)
        & frame["reward_variance"].fillna(0.0).gt(cfg.min_variance)
        & frame["extract_rate"].fillna(0.0).ge(cfg.min_extract_rate)
    )
    frame["medium_keep"] = keep
    frame["medium_reason"] = "filtered"
    if keep.sum() == 0 and cfg.allow_full_fallback:
        frame["medium_keep"] = True
        frame["medium_reason"] = "fallback_full"
    return frame


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def config_to_dict(cfg: Any) -> dict[str, Any]:
    return asdict(cfg)
