import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


def _to_builtin(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float | str):
        return value

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}

    if isinstance(value, list | tuple):
        return [_to_builtin(v) for v in value]

    return str(value)


class WallClockStudyRecorder:
    def __init__(self, trainer_config, experiment_name: str):
        study_cfg = trainer_config.get("wallclock_study", {})
        self.enabled = bool(study_cfg.get("enable", False))
        self.start_after_step = int(study_cfg.get("start_after_step", 20))
        self.fixed_budget_hours = float(study_cfg.get("fixed_budget_hours", 12.0))
        self.primary_metric_key = study_cfg.get("primary_metric_key", "val-core/aime_combined_score")

        self._start_time = None
        self._raw_cycle_index = 0
        self._effective_update_index = 0
        self._pending_effective_time_s = 0.0
        self._pending_effective_timing = defaultdict(float)

        self.output_dir = None
        self.events_path = None
        self.meta_path = None

        if not self.enabled:
            return

        output_subdir = study_cfg.get("output_subdir", "wallclock_study")
        self.output_dir = Path(trainer_config.default_local_dir) / output_subdir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.output_dir / "events.jsonl"
        self.meta_path = self.output_dir / "meta.json"

        meta = {
            "experiment_name": experiment_name,
            "start_after_step": self.start_after_step,
            "fixed_budget_hours": self.fixed_budget_hours,
            "primary_metric_key": self.primary_metric_key,
            "trainer_n_gpus_per_node": int(trainer_config.n_gpus_per_node),
            "trainer_nnodes": int(trainer_config.nnodes),
            "total_gpus": int(trainer_config.n_gpus_per_node) * int(trainer_config.nnodes),
        }
        self.meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")

    def should_track(self, global_step: int) -> bool:
        return self.enabled and int(global_step) > self.start_after_step

    def _ensure_started(self) -> None:
        if self._start_time is None:
            self._start_time = time.perf_counter()

    def record_cycle(
        self,
        *,
        global_step: int,
        raw_cycle_step: int,
        is_effective_update: bool,
        num_gen_batches: int,
        timing_raw: dict[str, float],
        metrics: dict[str, Any],
        note: str | None = None,
    ) -> None:
        if not self.should_track(global_step):
            return

        self._ensure_started()
        self._raw_cycle_index += 1

        cycle_timing = {str(k): float(v) for k, v in timing_raw.items() if isinstance(v, int | float)}
        cycle_time_s = float(cycle_timing.get("step", 0.0))
        self._pending_effective_time_s += cycle_time_s
        for name, value in cycle_timing.items():
            self._pending_effective_timing[name] += float(value)

        elapsed_s = float(time.perf_counter() - self._start_time)
        event = {
            "event_type": "train_cycle",
            "global_step": int(global_step),
            "raw_cycle_step": int(raw_cycle_step),
            "raw_cycle_index": int(self._raw_cycle_index),
            "effective_update_index": int(self._effective_update_index),
            "is_effective_update": bool(is_effective_update),
            "num_gen_batches": int(num_gen_batches),
            "cycle_time_s": cycle_time_s,
            "elapsed_time_s": elapsed_s,
            "elapsed_hours": elapsed_s / 3600.0,
            "within_fixed_budget": bool(elapsed_s / 3600.0 <= self.fixed_budget_hours),
            "timing_s": cycle_timing,
            "metrics": {
                str(k): _to_builtin(v)
                for k, v in metrics.items()
                if isinstance(v, bool | int | float) or hasattr(v, "item")
            },
        }

        if note:
            event["note"] = note

        if is_effective_update:
            self._effective_update_index += 1
            event["effective_update_index"] = int(self._effective_update_index)
            event["effective_update_time_s"] = float(self._pending_effective_time_s)
            event["effective_timing_s"] = {
                str(k): float(v) for k, v in self._pending_effective_timing.items()
            }
            self._pending_effective_time_s = 0.0
            self._pending_effective_timing = defaultdict(float)

        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
