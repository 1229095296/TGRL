from __future__ import annotations

import json
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
import ray
import torch
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    compute_response_mask,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.profiler import marked_timer

from .mixed_advantage import (
    compute_group_reward_statistics,
    compute_mixed_grouped_advantages,
    compute_pseudosplit_statistics,
    compute_standard_grpo_advantages,
)
from .rollout_utils import (
    attach_arm_metadata,
    build_group_uids,
    build_prompt_ids,
    ensure_dir,
    pop_generation_batch,
    repeat_prompt_metadata,
)


class MixedTemperatureGRPOTrainer(RayPPOTrainer):
    """Experiment-local trainer for single-temp and mixed-temp grouped GRPO."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.experiment_cfg = self.config.get("mixed_temp_experiment", None) or self.config.get("experiment", None)
        if self.experiment_cfg is None or self.experiment_cfg.get("method", None) is None:
            raise ValueError("Missing mixed-temp experiment config. Expected 'mixed_temp_experiment.method'.")
        self.method = str(self.experiment_cfg.method)
        self.control_count = int(self.experiment_cfg.control_count)
        self.explore_count = int(self.experiment_cfg.explore_count)
        self.total_group_size = int(self.experiment_cfg.total_group_size)
        self.low_temperature = float(self.experiment_cfg.temperatures.low)
        self.high_temperature = float(self.experiment_cfg.temperatures.high)
        self.update_control_arm = bool(self.experiment_cfg.get("update_control_arm", False))
        self.norm_adv_by_std = bool(self.config.algorithm.get("norm_adv_by_std_in_grpo", True))

        self.analysis_dir = ensure_dir(
            self.experiment_cfg.get(
                "analysis_dir",
                str(Path(self.config.trainer.default_local_dir) / "mixed_temp_analysis"),
            )
        )
        analysis_cfg = self.experiment_cfg.get("analysis", {})
        self.sample_dir = ensure_dir(self.analysis_dir / "sample_tables")
        self.response_dir = ensure_dir(self.analysis_dir / "responses")
        self.metrics_path = self.analysis_dir / "metrics.jsonl"
        self.training_curves_path = self.analysis_dir / "training_curves.csv"
        self.final_summary_path = self.analysis_dir / "final_summary.csv"
        self.sample_dump_steps = {int(step) for step in analysis_cfg.get("sample_dump_steps", [])}
        self.record_all_samples = bool(analysis_cfg.get("record_all_samples", True))
        self.dump_legacy_samples = bool(analysis_cfg.get("dump_legacy_samples", False))
        self.curve_rows: list[dict[str, object]] = []
        self._best_val_row: dict[str, object] | None = None
        self._validate_experiment_config()

    def _validate_experiment_config(self) -> None:
        valid_methods = {"single_low", "single_high", "mixed_grouped"}
        if self.method not in valid_methods:
            raise ValueError(f"Unsupported experiment.method={self.method}. Expected one of {sorted(valid_methods)}")

        if self.total_group_size <= 0:
            raise ValueError("experiment.total_group_size must be positive")

        if self.method == "mixed_grouped":
            if self.control_count <= 0 or self.explore_count <= 0:
                raise ValueError("mixed_grouped requires control_count>0 and explore_count>0")
            if self.control_count + self.explore_count != self.total_group_size:
                raise ValueError("control_count + explore_count must equal total_group_size")
        else:
            if self.total_group_size != int(self.config.actor_rollout_ref.rollout.n):
                raise ValueError("single-temperature runs require rollout.n == experiment.total_group_size")

        if (
            self.method == "mixed_grouped"
            and not self.update_control_arm
            and (
                float(self.config.actor_rollout_ref.actor.entropy_coeff) != 0.0
                or (
                    bool(self.config.actor_rollout_ref.actor.use_kl_loss)
                    and float(self.config.actor_rollout_ref.actor.kl_loss_coef) != 0.0
                )
            )
        ):
            raise ValueError(
                "To keep control-arm gradients exactly zero, set entropy_coeff=0 and disable KL loss "
                "or set kl_loss_coef=0."
            )

    def _generate_with_temperature(
        self,
        prompt_batch: DataProto,
        gen_batch: DataProto,
        prompt_ids: np.ndarray,
        uid_base: np.ndarray,
        repeat_times: int,
        arm_name: str,
        temperature: float,
        sample_idx_start: int,
    ) -> tuple[DataProto, dict[str, float]]:
        rollout_batch = gen_batch.repeat(repeat_times=repeat_times, interleave=True)
        rollout_batch.meta_info["global_steps"] = self.global_steps
        rollout_batch.meta_info["temperature"] = float(temperature)
        rollout_batch.meta_info["do_sample"] = True

        if not self.async_rollout_mode:
            rollout_output = self.actor_rollout_wg.generate_sequences(rollout_batch)
        else:
            rollout_output = self.async_rollout_manager.generate_sequences(rollout_batch)
        timing = dict(rollout_output.meta_info.get("timing", {}))
        rollout_output.meta_info.pop("timing", None)

        train_batch = prompt_batch.repeat(repeat_times=repeat_times, interleave=True)
        train_batch = train_batch.union(rollout_output)
        uid, prompt_id, arm, temperature_used, sample_idx, global_step_arr = repeat_prompt_metadata(
            prompt_ids=prompt_ids,
            uid_base=uid_base,
            repeat_times=repeat_times,
            arm_name=arm_name,
            temperature=temperature,
            global_step=self.global_steps,
            sample_idx_start=sample_idx_start,
        )
        train_batch = attach_arm_metadata(
            batch=train_batch,
            uid=uid,
            prompt_id=prompt_id,
            arm=arm,
            temperature_used=temperature_used,
            sample_idx_within_prompt=sample_idx,
            global_step_arr=global_step_arr,
        )
        train_batch.meta_info["temperature"] = float(temperature)
        return train_batch, timing

    def _build_train_batch(self, batch_dict: dict) -> tuple[DataProto, dict[str, float]]:
        prompt_batch = DataProto.from_single_dict(batch_dict)
        gen_batch = pop_generation_batch(prompt_batch)
        prompt_ids = build_prompt_ids(prompt_batch)
        uid_base = build_group_uids(prompt_ids=prompt_ids, global_step=self.global_steps)

        timing = {}
        if self.method == "single_low":
            train_batch, arm_timing = self._generate_with_temperature(
                prompt_batch=prompt_batch,
                gen_batch=gen_batch,
                prompt_ids=prompt_ids,
                uid_base=uid_base,
                repeat_times=self.total_group_size,
                arm_name="explore",
                temperature=self.low_temperature,
                sample_idx_start=0,
            )
            timing.update(arm_timing)
        elif self.method == "single_high":
            train_batch, arm_timing = self._generate_with_temperature(
                prompt_batch=prompt_batch,
                gen_batch=gen_batch,
                prompt_ids=prompt_ids,
                uid_base=uid_base,
                repeat_times=self.total_group_size,
                arm_name="explore",
                temperature=self.high_temperature,
                sample_idx_start=0,
            )
            timing.update(arm_timing)
        else:
            control_batch, control_timing = self._generate_with_temperature(
                prompt_batch=prompt_batch,
                gen_batch=gen_batch,
                prompt_ids=prompt_ids,
                uid_base=uid_base,
                repeat_times=self.control_count,
                arm_name="control",
                temperature=self.low_temperature,
                sample_idx_start=0,
            )
            explore_batch, explore_timing = self._generate_with_temperature(
                prompt_batch=prompt_batch,
                gen_batch=gen_batch,
                prompt_ids=prompt_ids,
                uid_base=uid_base,
                repeat_times=self.explore_count,
                arm_name="explore",
                temperature=self.high_temperature,
                sample_idx_start=self.control_count,
            )
            timing.update({f"control_{k}": v for k, v in control_timing.items()})
            timing.update({f"explore_{k}": v for k, v in explore_timing.items()})
            train_batch = DataProto.concat([control_batch, explore_batch])
            train_batch.meta_info["temperature"] = float(self.high_temperature)

        if "response_mask" not in train_batch.batch.keys():
            train_batch.batch["response_mask"] = compute_response_mask(train_batch)
        return train_batch, timing

    def _compute_advantages(self, batch: DataProto) -> tuple[DataProto, dict[str, float]]:
        if self.method == "mixed_grouped":
            advantages, returns, diag = compute_mixed_grouped_advantages(
                token_level_rewards=batch.batch["token_level_rewards"],
                response_mask=batch.batch["response_mask"],
                uid=batch.non_tensor_batch["uid"],
                arm=batch.non_tensor_batch["arm"],
                update_control_arm=self.update_control_arm,
                norm_adv_by_std=self.norm_adv_by_std,
            )
        else:
            advantages, returns = compute_standard_grpo_advantages(
                token_level_rewards=batch.batch["token_level_rewards"],
                response_mask=batch.batch["response_mask"],
                uid=batch.non_tensor_batch["uid"],
                norm_adv_by_std_in_grpo=self.norm_adv_by_std,
            )
            diag = {}

        batch.batch["advantages"] = advantages
        batch.batch["returns"] = returns
        return batch, diag

    def _method_specific_metrics(self, batch: DataProto) -> dict[str, float]:
        metrics = {}
        if self.method == "mixed_grouped":
            metrics.update(
                compute_group_reward_statistics(
                    token_level_rewards=batch.batch["token_level_rewards"],
                    uid=batch.non_tensor_batch["uid"],
                    arm=batch.non_tensor_batch["arm"],
                )
            )
        elif self.method == "single_high" and self.control_count > 0 and self.control_count < self.total_group_size:
            rewards = batch.batch["token_level_rewards"].sum(dim=-1).detach().cpu().numpy()
            metrics.update(
                compute_pseudosplit_statistics(
                    rewards=rewards,
                    prompt_ids=batch.non_tensor_batch["prompt_id"],
                    control_count=self.control_count,
                )
            )
        return metrics

    def _append_metrics_jsonl(self, metrics: dict[str, float]) -> None:
        payload = {k: _to_jsonable(v) for k, v in metrics.items()}
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _build_response_frame(self, batch: DataProto) -> pd.DataFrame:
        prompt_texts = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
        response_texts = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
        rewards = batch.batch["token_level_rewards"].sum(dim=-1).detach().cpu().numpy().tolist()
        response_mask = batch.batch["response_mask"].to(dtype=batch.batch["advantages"].dtype)
        advantage_scalar = (
            (batch.batch["advantages"] * response_mask).sum(dim=-1)
            / response_mask.sum(dim=-1).clamp_min(1.0)
        ).detach().cpu().numpy().tolist()
        response_length = response_mask.sum(dim=-1).detach().cpu().numpy().tolist()
        old_log_probs = None
        if "old_log_probs" in batch.batch:
            mask = batch.batch["response_mask"].to(dtype=batch.batch["old_log_probs"].dtype)
            old_log_probs = (batch.batch["old_log_probs"] * mask).sum(dim=-1).detach().cpu().numpy().tolist()

        frame = pd.DataFrame(
            {
            "uid": batch.non_tensor_batch["uid"].tolist(),
            "response_uid": [
                f"{uid}#sample{sample_idx}"
                for uid, sample_idx in zip(
                    batch.non_tensor_batch["uid"].tolist(),
                    batch.non_tensor_batch["sample_idx_within_prompt"].tolist(),
                    strict=True,
                )
            ],
            "prompt_id": batch.non_tensor_batch["prompt_id"].tolist(),
            "index": batch.non_tensor_batch["prompt_id"].tolist(),
            "arm": batch.non_tensor_batch["arm"].tolist(),
            "temperature_used": batch.non_tensor_batch["temperature_used"].tolist(),
            "sample_idx_within_prompt": batch.non_tensor_batch["sample_idx_within_prompt"].tolist(),
            "global_step": batch.non_tensor_batch["global_step"].tolist(),
            "prompt_text": prompt_texts,
            "response_text": response_texts,
            "reward": rewards,
            "advantage_scalar": advantage_scalar,
            "response_length": response_length,
        }
        )
        if old_log_probs is not None:
            frame["old_log_prob"] = old_log_probs
            frame["old_logprob_sum"] = old_log_probs
        for key in ["extracted_answer", "extraction_success", "extraction_failure", "score", "data_source"]:
            if key in batch.non_tensor_batch:
                frame[key] = batch.non_tensor_batch[key].tolist()
        if "reward_model" in batch.non_tensor_batch:
            frame["ground_truth"] = [
                item.get("ground_truth") if isinstance(item, dict) else None
                for item in batch.non_tensor_batch["reward_model"].tolist()
            ]
        elif "extra_info" in batch.non_tensor_batch:
            frame["ground_truth"] = [
                (item.get("answer") if isinstance(item, dict) else None)
                for item in batch.non_tensor_batch["extra_info"].tolist()
            ]
        if "split" in batch.non_tensor_batch:
            frame["split"] = batch.non_tensor_batch["split"].tolist()
        elif "extra_info" in batch.non_tensor_batch:
            frame["split"] = [
                info.get("split") if isinstance(info, dict) else None
                for info in batch.non_tensor_batch["extra_info"].tolist()
            ]
        else:
            frame["split"] = "train"
        frame["response"] = frame["response_text"]
        if "extraction_failure" not in frame.columns and "extraction_success" in frame.columns:
            frame["extraction_failure"] = (~pd.Series(frame["extraction_success"]).astype(bool)).tolist()
        return frame

    def _write_response_frame(self, frame: pd.DataFrame, output_path: Path) -> Path:
        try:
            frame.to_parquet(output_path, index=False)
            return output_path
        except Exception:
            fallback_path = output_path.with_suffix(".csv")
            frame.to_csv(fallback_path, index=False)
            return fallback_path

    def _maybe_dump_step_samples(self, batch: DataProto) -> pd.DataFrame:
        frame = self._build_response_frame(batch)
        if self.record_all_samples:
            self._write_response_frame(frame, self.response_dir / f"responses_step_{self.global_steps:06d}.parquet")
        if self.dump_legacy_samples or (self.sample_dump_steps and self.global_steps in self.sample_dump_steps):
            self._write_response_frame(frame, self.sample_dir / f"sample_table_step_{self.global_steps:06d}.parquet")
        return frame

    def _write_training_curves(self, metrics: dict[str, float]) -> None:
        row = {k: _to_jsonable(v) for k, v in metrics.items()}
        row["method"] = self.method
        row["experiment_name"] = self.config.trainer.experiment_name
        self.curve_rows.append(row)
        pd.DataFrame(self.curve_rows).to_csv(self.training_curves_path, index=False)

    def _update_best_val_row(self, metrics: dict[str, float]) -> None:
        val_candidates = {
            key: float(_to_jsonable(value))
            for key, value in metrics.items()
            if key.startswith("val") and isinstance(_to_jsonable(value), (int, float))
        }
        if not val_candidates:
            return
        target_key = next((key for key in val_candidates if "mean@8" in key), None)
        if target_key is None:
            target_key = next((key for key in val_candidates if "mean@1" in key), None)
        if target_key is None:
            target_key = next(iter(val_candidates))
        if self._best_val_row is None or val_candidates[target_key] > float(self._best_val_row.get(target_key, -1e9)):
            self._best_val_row = {k: _to_jsonable(v) for k, v in metrics.items()}

    def _write_final_summary(self, metrics: dict[str, float]) -> None:
        summary = {
            "experiment_name": self.config.trainer.experiment_name,
            "method": self.method,
            "global_step": int(metrics.get("training/global_step", self.global_steps)),
            "epoch": int(metrics.get("training/epoch", 0)),
            "train_reward_mean": float(metrics.get("critic/rewards/mean", 0.0)),
            "response_length_mean": float(metrics.get("response_length/mean", 0.0)),
            "extraction_failure_rate": float(metrics.get("reward/extraction_failure_rate", 0.0)),
            "reward/control_mean": float(metrics.get("reward/control_mean", 0.0)),
            "reward/explore_mean": float(metrics.get("reward/explore_mean", 0.0)),
            "reward_gap": float(metrics.get("reward_gap", metrics.get("reward/gap", 0.0))),
            "adv/explore_mean": float(metrics.get("adv/explore_mean", 0.0)),
            "adv/explore_std": float(metrics.get("adv/explore_std", 0.0)),
        }
        for key, value in metrics.items():
            if key.startswith("val"):
                summary[f"latest::{key}"] = _to_jsonable(value)
        if self._best_val_row is not None:
            for key, value in self._best_val_row.items():
                if key.startswith("val"):
                    summary[f"best::{key}"] = _to_jsonable(value)
        pd.DataFrame([summary]).to_csv(self.final_summary_path, index=False)

    def fit(self):
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Mixed GRPO Progress")
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics: dict[str, float] = {}
                timing_raw: dict[str, float] = {}

                do_profile = (
                    self.global_steps in self.config.trainer.profile_steps
                    if self.config.trainer.profile_steps is not None
                    else False
                )
                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(do_profile)

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    with marked_timer("gen", timing_raw, color="red"):
                        batch, gen_timing = self._build_train_batch(batch_dict)
                        timing_raw.update(gen_timing)

                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    batch.meta_info["temperature"] = 1.0

                    with marked_timer("reward", timing_raw, color="yellow"):
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    if self.config.reward_model.launch_reward_fn_async:
                        reward_tensor, reward_extra_infos_dict = ray.get(future_reward)

                    batch.batch["token_level_scores"] = reward_tensor
                    if reward_extra_infos_dict:
                        batch.non_tensor_batch.update({k: np.array(v, dtype=object) for k, v in reward_extra_infos_dict.items()})
                    batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                    with marked_timer("adv", timing_raw, color="brown"):
                        batch, adv_metrics = self._compute_advantages(batch)
                        metrics.update(adv_metrics)
                        metrics.update(self._method_specific_metrics(batch))

                    response_frame = self._maybe_dump_step_samples(batch)

                    with marked_timer("update_actor", timing_raw, color="red"):
                        batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                        batch.meta_info["temperature"] = 1.0
                        actor_output = self.actor_rollout_wg.update_actor(batch)
                    metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))

                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("testing", timing_raw, color="green"):
                            val_metrics = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    self._stop_profiling(do_profile)

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                        "experiment/control_count": float(self.control_count),
                        "experiment/explore_count": float(self.explore_count),
                        "experiment/total_group_size": float(self.total_group_size),
                        "experiment/temperature_low": self.low_temperature,
                        "experiment/temperature_high": self.high_temperature,
                    }
                )
                if "extraction_failure" in response_frame.columns:
                    metrics["reward/extraction_failure_rate"] = float(
                        pd.to_numeric(response_frame["extraction_failure"], errors="coerce").fillna(0.0).mean()
                    )
                elif "extraction_success" in response_frame.columns:
                    metrics["reward/extraction_failure_rate"] = float(
                        1.0
                        - pd.to_numeric(response_frame["extraction_success"], errors="coerce").fillna(0.0).mean()
                    )
                else:
                    metrics["reward/extraction_failure_rate"] = 0.0
                metrics["reward_gap"] = float(metrics.get("reward/gap", 0.0))
                metrics.update(compute_data_metrics(batch=batch, use_critic=False))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                metrics.update(
                    compute_throughout_metrics(
                        batch=batch,
                        timing_raw=timing_raw,
                        n_gpus=self.resource_pool_manager.get_n_gpus(),
                    )
                )

                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                logger.log(data=metrics, step=self.global_steps)
                self._append_metrics_jsonl(metrics)
                self._write_training_curves(metrics)
                self._update_best_val_row(metrics)
                self._write_final_summary(metrics)
                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    progress_bar.close()
                    if last_val_metrics is not None:
                        pprint(f"Final validation metrics: {last_val_metrics}")
                    return


def _to_jsonable(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    return value
