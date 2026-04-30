from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from verl import DataProto


def pop_generation_batch(batch: DataProto) -> DataProto:
    """Split the prompt-side generation inputs out of a training batch."""
    batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
    non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
    if "multi_modal_data" in batch.non_tensor_batch:
        non_tensor_batch_keys_to_pop.append("multi_modal_data")
    if "raw_prompt" in batch.non_tensor_batch:
        non_tensor_batch_keys_to_pop.append("raw_prompt")
    if "tools_kwargs" in batch.non_tensor_batch:
        non_tensor_batch_keys_to_pop.append("tools_kwargs")
    if "interaction_kwargs" in batch.non_tensor_batch:
        non_tensor_batch_keys_to_pop.append("interaction_kwargs")
    if "agent_name" in batch.non_tensor_batch:
        non_tensor_batch_keys_to_pop.append("agent_name")
    return batch.pop(
        batch_keys=batch_keys_to_pop,
        non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
    )


def build_prompt_ids(batch: DataProto, allow_fallback: bool = False) -> np.ndarray:
    """Use dataset-provided stable prompt ids when available."""
    if "index" in batch.non_tensor_batch:
        return np.asarray(batch.non_tensor_batch["index"], dtype=object)
    extra_info = batch.non_tensor_batch.get("extra_info", None)
    if extra_info is None:
        if not allow_fallback:
            raise ValueError("Mixed-temp experiment requires stable prompt ids via non_tensor_batch['index'] or extra_info.index.")
        return np.arange(len(batch), dtype=object)
    prompt_ids = []
    for idx, item in enumerate(extra_info):
        if isinstance(item, dict) and "index" in item:
            prompt_ids.append(item["index"])
        else:
            if not allow_fallback:
                raise ValueError("Mixed-temp experiment requires extra_info.index for every prompt row.")
            prompt_ids.append(idx)
    return np.asarray(prompt_ids, dtype=object)


def build_group_uids(prompt_ids: np.ndarray, global_step: int) -> np.ndarray:
    return np.asarray([f"step{global_step}_prompt{pid}" for pid in prompt_ids], dtype=object)


def repeat_prompt_metadata(
    prompt_ids: np.ndarray,
    uid_base: np.ndarray,
    repeat_times: int,
    arm_name: str,
    temperature: float,
    global_step: int,
    sample_idx_start: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    num_prompts = len(prompt_ids)
    uid = np.repeat(uid_base, repeat_times)
    prompt_id = np.repeat(prompt_ids, repeat_times)
    arm = np.asarray([arm_name] * (num_prompts * repeat_times), dtype=object)
    temperature_used = np.full(num_prompts * repeat_times, temperature, dtype=np.float32)
    sample_idx = np.tile(np.arange(sample_idx_start, sample_idx_start + repeat_times, dtype=np.int64), num_prompts)
    global_step_arr = np.full(num_prompts * repeat_times, global_step, dtype=np.int64)
    return uid, prompt_id, arm, temperature_used, sample_idx, global_step_arr


def attach_arm_metadata(
    batch: DataProto,
    uid: np.ndarray,
    prompt_id: np.ndarray,
    arm: np.ndarray,
    temperature_used: np.ndarray,
    sample_idx_within_prompt: np.ndarray,
    global_step_arr: np.ndarray,
) -> DataProto:
    """Attach experiment-specific per-sample metadata after rollout merge."""
    batch.non_tensor_batch["uid"] = uid
    batch.non_tensor_batch["prompt_id"] = prompt_id
    batch.non_tensor_batch["arm"] = arm
    batch.non_tensor_batch["temperature_used"] = temperature_used
    batch.non_tensor_batch["sample_idx_within_prompt"] = sample_idx_within_prompt
    batch.non_tensor_batch["global_step"] = global_step_arr
    batch.batch["sample_temperatures"] = torch.tensor(temperature_used, dtype=torch.float32)
    return batch


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
