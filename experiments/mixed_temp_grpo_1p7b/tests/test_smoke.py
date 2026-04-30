from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("ray")

from experiments.mixed_temp_grpo_1p7b.trainer.mixed_grpo_trainer import MixedTemperatureGRPOTrainer


class DummyTokenizer:
    def batch_decode(self, tokens, skip_special_tokens: bool = True):
        return [f"decoded_{idx}" for idx in range(tokens.shape[0])]


def build_fake_batch(num_prompts: int = 32, group_size: int = 8):
    total_rows = num_prompts * group_size
    control_count = 2
    prompt_ids = np.repeat(np.arange(num_prompts), group_size)
    sample_idx = np.tile(np.arange(group_size), num_prompts)
    arm = np.where(sample_idx < control_count, "control", "explore").astype(object)
    temperature = np.where(sample_idx < control_count, 0.3, 1.2).astype(np.float32)
    uid = np.asarray([f"step1_prompt{pid}" for pid in prompt_ids], dtype=object)

    token_rewards = torch.zeros((total_rows, 3), dtype=torch.float32)
    token_rewards[:, -1] = torch.tensor((sample_idx >= control_count).astype(np.float32))
    advantages = torch.ones_like(token_rewards) * torch.tensor((sample_idx >= control_count).astype(np.float32)).unsqueeze(-1)

    return SimpleNamespace(
        batch={
            "prompts": torch.ones((total_rows, 4), dtype=torch.long),
            "responses": torch.ones((total_rows, 3), dtype=torch.long),
            "response_mask": torch.ones((total_rows, 3), dtype=torch.float32),
            "token_level_rewards": token_rewards,
            "advantages": advantages,
            "old_log_probs": torch.full((total_rows, 3), -0.5, dtype=torch.float32),
        },
        non_tensor_batch={
            "uid": uid,
            "prompt_id": prompt_ids.astype(object),
            "arm": arm,
            "temperature_used": temperature.astype(object),
            "sample_idx_within_prompt": sample_idx.astype(object),
            "global_step": np.full(total_rows, 1, dtype=object),
            "extracted_answer": np.asarray(["42"] * total_rows, dtype=object),
            "extraction_success": np.asarray([True] * total_rows, dtype=object),
            "score": np.asarray([1.0] * total_rows, dtype=object),
            "data_source": np.asarray(["openai/gsm8k"] * total_rows, dtype=object),
            "split": np.asarray(["train"] * total_rows, dtype=object),
            "reward_model": np.asarray([{"ground_truth": "42"}] * total_rows, dtype=object),
        },
    )


def test_artifact_smoke_exports_expected_files(tmp_path: Path):
    trainer = object.__new__(MixedTemperatureGRPOTrainer)
    trainer.tokenizer = DummyTokenizer()
    trainer.global_steps = 1
    trainer.sample_dir = tmp_path / "sample_tables"
    trainer.sample_dir.mkdir()
    trainer.response_dir = tmp_path / "responses"
    trainer.response_dir.mkdir()
    trainer.metrics_path = tmp_path / "metrics.jsonl"
    trainer.training_curves_path = tmp_path / "training_curves.csv"
    trainer.final_summary_path = tmp_path / "final_summary.csv"
    trainer.sample_dump_steps = set()
    trainer.record_all_samples = True
    trainer.dump_legacy_samples = False
    trainer.curve_rows = []
    trainer._best_val_row = None
    trainer.method = "mixed_grouped"
    trainer.config = SimpleNamespace(trainer=SimpleNamespace(experiment_name="smoke-exp"))

    batch = build_fake_batch()
    frame = trainer._maybe_dump_step_samples(batch)
    assert len(frame) == 32 * 8
    assert {
        "uid",
        "response_uid",
        "arm",
        "temperature_used",
        "sample_idx_within_prompt",
        "global_step",
        "reward",
        "extracted_answer",
        "advantage_scalar",
        "old_log_prob",
        "split",
        "index",
    }.issubset(frame.columns)
    assert (tmp_path / "responses" / "responses_step_000001.parquet").exists() or (
        tmp_path / "responses" / "responses_step_000001.csv"
    ).exists()

    metrics_step1 = {
        "training/global_step": 1,
        "training/epoch": 0,
        "critic/rewards/mean": 0.25,
        "response_length/mean": 3.0,
        "reward/extraction_failure_rate": 0.0,
        "reward/control_mean": 0.0,
        "reward/explore_mean": 1.0,
        "reward/gap": 1.0,
        "reward_gap": 1.0,
        "adv/explore_mean": 1.0,
        "adv/explore_std": 0.0,
        "val/test_score/mean@8": 0.5,
    }
    trainer._write_training_curves(metrics_step1)
    trainer._update_best_val_row(metrics_step1)
    trainer._write_final_summary(metrics_step1)

    metrics_step2 = dict(metrics_step1)
    metrics_step2["training/global_step"] = 2
    metrics_step2["val/test_score/mean@8"] = 0.6
    trainer._write_training_curves(metrics_step2)
    trainer._update_best_val_row(metrics_step2)
    trainer._write_final_summary(metrics_step2)

    curves = pd.read_csv(trainer.training_curves_path)
    summary = pd.read_csv(trainer.final_summary_path)
    assert len(curves) == 2
    assert "best::val/test_score/mean@8" in summary.columns
    assert summary.iloc[0]["best::val/test_score/mean@8"] == pytest.approx(0.6)
