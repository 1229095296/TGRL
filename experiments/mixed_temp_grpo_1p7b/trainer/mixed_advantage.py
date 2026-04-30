from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch

from verl.trainer.ppo import core_algos


def compute_standard_grpo_advantages(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    uid: np.ndarray,
    norm_adv_by_std_in_grpo: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Wrapper around the existing GRPO implementation for experiment-local use."""
    return core_algos.compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=uid,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
    )


def compute_mixed_grouped_advantages(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    uid: np.ndarray,
    arm: np.ndarray,
    update_control_arm: bool = False,
    norm_adv_by_std: bool = True,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Compute mixed-temperature grouped GRPO advantages.

    Each prompt group uses all control+explore rewards to estimate the group baseline.
    By default only exploration samples carry non-zero policy gradient.
    """

    scores = token_level_rewards.sum(dim=-1)
    advantages_scalar = torch.zeros_like(scores)

    uid_to_indices: dict[Any, list[int]] = defaultdict(list)
    for i, group_uid in enumerate(uid):
        uid_to_indices[group_uid].append(i)

    control_reward_means = []
    explore_reward_means = []
    reward_gaps = []
    explore_adv_values = []
    control_adv_values = []

    with torch.no_grad():
        for group_uid, indices in uid_to_indices.items():
            group_scores = scores[indices]
            group_arms = arm[indices]

            mean_reward = group_scores.mean()
            if group_scores.numel() <= 1:
                std_reward = torch.tensor(1.0, device=group_scores.device, dtype=group_scores.dtype)
            else:
                std_reward = group_scores.std(unbiased=False)
                if float(std_reward.item()) < epsilon:
                    std_reward = torch.tensor(1.0, device=group_scores.device, dtype=group_scores.dtype)

            if norm_adv_by_std:
                group_adv = (group_scores - mean_reward) / (std_reward + epsilon)
            else:
                group_adv = group_scores - mean_reward

            for local_pos, batch_idx in enumerate(indices):
                sample_arm = group_arms[local_pos]
                adv_value = group_adv[local_pos]
                if sample_arm == "control" and not update_control_arm:
                    adv_value = torch.zeros_like(adv_value)
                advantages_scalar[batch_idx] = adv_value

            control_mask = group_arms == "control"
            explore_mask = group_arms == "explore"
            if control_mask.any():
                control_mean = group_scores[control_mask].mean().item()
                control_reward_means.append(control_mean)
                if update_control_arm:
                    control_adv_values.extend(group_adv[control_mask].cpu().tolist())
            if explore_mask.any():
                explore_mean = group_scores[explore_mask].mean().item()
                explore_reward_means.append(explore_mean)
                reward_gaps.append(explore_mean - (control_reward_means[-1] if control_mask.any() else mean_reward.item()))
                explore_adv_values.extend(group_adv[explore_mask].cpu().tolist())

        advantages = advantages_scalar.unsqueeze(-1) * response_mask
        returns = advantages.clone()

    diag = {
        "reward/control_mean": _safe_mean(control_reward_means),
        "reward/explore_mean": _safe_mean(explore_reward_means),
        "reward/gap": _safe_mean(reward_gaps),
        "reward_gap": _safe_mean(reward_gaps),
        "adv/explore_mean": _safe_mean(explore_adv_values),
        "adv/explore_std": _safe_std(explore_adv_values),
        "adv/control_mean": _safe_mean(control_adv_values),
        "carrier/positive_gain_expected": _safe_mean([v for v in explore_adv_values if v > 0]),
        "carrier/negative_gain_expected": _safe_mean([v for v in explore_adv_values if v < 0]),
    }
    return advantages, returns, diag


def compute_group_reward_statistics(
    token_level_rewards: torch.Tensor,
    uid: np.ndarray,
    arm: np.ndarray,
) -> dict[str, float]:
    """Compute reward diagnostics shared by training and offline scripts."""
    scores = token_level_rewards.sum(dim=-1).detach().cpu()
    uid_to_rows: dict[Any, list[int]] = defaultdict(list)
    for i, group_uid in enumerate(uid):
        uid_to_rows[group_uid].append(i)

    control_means = []
    explore_means = []
    gaps = []
    for group_uid, rows in uid_to_rows.items():
        group_scores = scores[rows]
        group_arms = arm[rows]
        control_mask = group_arms == "control"
        explore_mask = group_arms == "explore"
        if control_mask.any():
            control_mean = float(group_scores[control_mask].mean().item())
            control_means.append(control_mean)
        else:
            control_mean = float(group_scores.mean().item())
        if explore_mask.any():
            explore_mean = float(group_scores[explore_mask].mean().item())
            explore_means.append(explore_mean)
            gaps.append(explore_mean - control_mean)

    return {
        "reward/control_mean": _safe_mean(control_means),
        "reward/explore_mean": _safe_mean(explore_means),
        "reward/gap": _safe_mean(gaps),
        "reward_gap": _safe_mean(gaps),
    }


def compute_pseudosplit_statistics(
    rewards: np.ndarray,
    prompt_ids: np.ndarray,
    control_count: int,
) -> dict[str, float]:
    """Compute same-temperature pseudo-split diagnostics from offline samples."""
    prompt_to_rewards: dict[Any, list[float]] = defaultdict(list)
    for prompt_id, reward in zip(prompt_ids, rewards, strict=True):
        prompt_to_rewards[prompt_id].append(float(reward))

    deltas = []
    for prompt_id, group_rewards in prompt_to_rewards.items():
        if len(group_rewards) <= control_count:
            continue
        control = np.mean(group_rewards[:control_count])
        explore = np.mean(group_rewards[control_count:])
        deltas.append(float(explore - control))

    return {
        "pseudo_split/delta_mean": _safe_mean(deltas),
        "pseudo_split/delta_std": _safe_std(deltas),
        "pseudo_split/count": float(len(deltas)),
    }


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(values))


def _safe_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(np.std(values))
