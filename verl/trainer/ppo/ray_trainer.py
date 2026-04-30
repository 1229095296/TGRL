# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import shutil
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Any, Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, TAMPOMetaPolicyController, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.wallclock_study import WallClockStudyRecorder
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.js_token_visualization import render_js_token_visualization_html
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger

WorkerType = type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(
                    f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}"
                    + "cannot be satisfied in this ray cluster"
                )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
    global_steps: int = 0,
    total_training_steps: int = 1,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.
        global_steps (int, optional): Current training step for adaptive alpha scheduling. Defaults to 0.
        total_training_steps (int, optional): Total training steps for adaptive alpha scheduling. Defaults to 1.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.reweight_method,
                config.pf_ppo.weight_pow,
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )

        # No IS weights needed
        data.batch["is_weights"] = None

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


def apply_rowwise_quantile_clip(
    values: torch.Tensor,
    token_mask: torch.Tensor,
    row_selector: torch.Tensor,
    lower_quantile: float,
    upper_quantile: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Clip per-token gates to row-wise quantile bounds on selected rows."""
    clipped_values = values.clone()
    stats = {
        "lower_clip_rate": 0.0,
        "upper_clip_rate": 0.0,
        "lower_bound_mean": 0.0,
        "upper_bound_mean": 0.0,
    }

    selected_rows = torch.where(row_selector)[0].tolist()
    if not selected_rows:
        return clipped_values, stats

    lower_hits = 0
    upper_hits = 0
    total_tokens = 0
    lower_bounds = []
    upper_bounds = []

    for row_idx in selected_rows:
        row_token_mask = token_mask[row_idx]
        row_values = values[row_idx][row_token_mask]
        if row_values.numel() == 0:
            continue

        row_values_float = row_values.float()
        lower_bound = torch.quantile(row_values_float, lower_quantile)
        upper_bound = torch.quantile(row_values_float, upper_quantile)
        if upper_bound < lower_bound:
            lower_bound, upper_bound = upper_bound, lower_bound

        lower_hits += int((row_values_float < lower_bound).sum().item())
        upper_hits += int((row_values_float > upper_bound).sum().item())
        total_tokens += int(row_values.numel())
        lower_bounds.append(float(lower_bound.item()))
        upper_bounds.append(float(upper_bound.item()))

        clipped_row = torch.clamp(row_values_float, min=float(lower_bound.item()), max=float(upper_bound.item()))
        clipped_values[row_idx, row_token_mask] = clipped_row.to(dtype=row_values.dtype)

    if total_tokens > 0:
        stats["lower_clip_rate"] = lower_hits / total_tokens
        stats["upper_clip_rate"] = upper_hits / total_tokens
    if lower_bounds:
        stats["lower_bound_mean"] = float(np.mean(lower_bounds))
        stats["upper_bound_mean"] = float(np.mean(upper_bounds))
    return clipped_values, stats


def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sparsemax projection for TAMPO likelihood normalization."""
    logits = logits - logits.max(dim=dim, keepdim=True).values
    zs = torch.sort(logits, dim=dim, descending=True).values
    range_shape = [1] * logits.ndim
    range_shape[dim] = zs.size(dim)
    k = torch.arange(1, zs.size(dim) + 1, device=logits.device, dtype=logits.dtype).view(range_shape)
    bound = 1 + k * zs
    cumulative_zs = torch.cumsum(zs, dim=dim)
    support = bound > cumulative_zs
    support_size = support.sum(dim=dim, keepdim=True).clamp(min=1)
    tau = (torch.gather(cumulative_zs, dim, support_size - 1) - 1) / support_size.to(dtype=logits.dtype)
    return torch.clamp(logits - tau, min=0)


def resolve_hybrid_group_counts(rollout_config) -> tuple[int, int, int]:
    rollout_n = int(rollout_config.n)
    if rollout_n <= 0:
        raise ValueError(f"rollout.n must be positive, got {rollout_n}")

    baseline_count = 1
    exploration_count = max(rollout_n - baseline_count, 0)
    hybrid_cfg = getattr(rollout_config, "hybrid_exploration", None)
    if hybrid_cfg is not None and hybrid_cfg.get("enable", False):
        baseline_count = int(hybrid_cfg.get("baseline_count", 1))
        exploration_override = hybrid_cfg.get("exploration_count", None)
        exploration_count = (
            int(exploration_override) if exploration_override is not None else max(rollout_n - baseline_count, 0)
        )

    if baseline_count < 0 or exploration_count < 0:
        raise ValueError(
            f"Hybrid exploration counts must be non-negative, got baseline_count={baseline_count}, "
            f"exploration_count={exploration_count}"
        )
    if baseline_count + exploration_count != rollout_n:
        raise ValueError(
            f"Hybrid exploration counts must sum to rollout.n={rollout_n}, "
            f"got baseline_count={baseline_count}, exploration_count={exploration_count}"
        )
    if rollout_n >= 2 and baseline_count == 0:
        raise ValueError("Hybrid exploration requires at least one baseline sample per group")
    if rollout_n >= 2 and exploration_count == 0:
        raise ValueError("Hybrid exploration requires at least one exploration sample per group")

    return baseline_count, exploration_count, rollout_n


def apply_token_js_weighting(
    data: DataProto,
    config,
    rollout_config,
    global_steps: int = 0,
    temperature_override: Optional[tuple[float, float]] = None,
) -> DataProto:
    """
    Apply token-level JS-divergence-weighted advantage reweighting.

    Warmup Strategy (steps < warmup_steps):
    - All samples use T=T1 (standard GRPO)
    - No advantage reweighting

    After Warmup (steps >= warmup_steps):
    - Samples [0, baseline_count) use T=T0 as baseline, weight=0 (不参与梯度更新)
    - Samples [baseline_count, rollout_n) use T=T1 as exploration samples
        - ALL exploration samples use JS-divergence weighting (no A≥0 vs A<0 distinction)

    Token-level JS-Divergence Weighting (for ALL exploration samples):
        JS[t] = 0.5 * (KL(P_T0 || M) + KL(P_T1 || M))
            where M = 0.5 * (P_T0 + P_T1)
        r[t] = (JS[t] + eps) / (mean(JS) + eps)
        w_raw[t] = log(1 + r[t])
        w_clip[t] = rowwise_quantile_clip(w_raw[t])
        w[t] = w_clip[t] / mean(w_clip)
        A_token[t] = A_traj * w[t]  # Apply weight

    Args:
        data: DataProto containing advantages, response_mask, js_logits, and uid
        config: PERConfig with JS weighting parameters
        rollout_config: Rollout config containing n and temperatures
        global_steps: Current training step for warmup control

    Returns:
        DataProto with reweighted advantages
    """
    if not config.enable_token_entropy_weighting:
        return data

    # Warmup phase: use standard GRPO (no reweighting)
    if global_steps < config.warmup_steps:
        print(f"[Warmup] Step {global_steps}/{config.warmup_steps}: Using standard GRPO (all samples T={config.entropy_t1})")
        return data

    # Check if js_weights are available
    if "js_weights" not in data.batch:
        print("[Token JS Weighting] Warning: js_weights not found in batch, skipping")
        return data

    # Get required data
    advantages = data.batch["advantages"]  # [batch_size, response_length]
    response_mask = data.batch["response_mask"]  # [batch_size, response_length]
    js_div_raw = data.batch["js_weights"]  # [batch_size, response_length] - raw JS divergence values
    token_level_rewards = data.batch.get("token_level_rewards", None)  # [batch_size, response_length]

    # Get rollout_n, group composition, and temperatures
    baseline_count_per_group, exploration_count_per_group, rollout_n = resolve_hybrid_group_counts(rollout_config)
    if rollout_n < 2:
        print(f"[Token JS Weighting] Warning: rollout_n={rollout_n} < 2, skipping")
        return data

    # Get temperatures from hybrid_exploration config
    if temperature_override is not None:
        T0, T1 = temperature_override
    elif hasattr(rollout_config, 'hybrid_exploration') and rollout_config.hybrid_exploration.get('enable', False):
        temperatures = rollout_config.hybrid_exploration.temperatures
        T0, T1 = temperatures[0], temperatures[1]
    else:
        # Fallback to config values
        T0, T1 = config.entropy_t0, config.entropy_t1

    # Determine sample index within each group (0 to rollout_n-1)
    if "sample_indices" in data.batch:
        sample_indices = data.batch["sample_indices"].to(device=advantages.device, dtype=torch.long)
    else:
        uid = data.non_tensor_batch["uid"]  # [batch_size] - UUID per prompt group
        sample_indices = []
        uid_counts = {}
        for u in uid:
            count = uid_counts.get(u, 0)
            sample_indices.append(count % rollout_n)
            uid_counts[u] = count + 1
        sample_indices = torch.tensor(sample_indices, device=advantages.device)

    # Identify sample type. Samples [0, baseline_count) are low-temperature baselines.
    is_baseline = sample_indices < baseline_count_per_group  # [batch_size]
    is_exploration = ~is_baseline

    # Statistics for logging
    num_baseline = is_baseline.sum().item()
    num_exploration = is_exploration.sum().item()

    mask_bool = response_mask.bool()
    mask_float = response_mask.to(dtype=advantages.dtype)
    valid_rows = mask_bool.any(dim=-1)
    baseline_valid = is_baseline & valid_rows
    exploration_valid = is_exploration & valid_rows

    # Batch-wise trajectory statistics.
    num_tokens = mask_float.sum(dim=-1)
    safe_num_tokens = num_tokens + config.entropy_eps
    A_traj = (advantages * mask_float).sum(dim=-1) / safe_num_tokens

    # Fixed ATJS1ctrl gate: ratio -> log1p -> row-wise quantile clip -> renorm.
    js_div_masked = js_div_raw * mask_float
    mean_js = js_div_masked.sum(dim=-1) / safe_num_tokens
    mean_js_expanded = mean_js.unsqueeze(-1)
    js_ratio = (js_div_raw + config.entropy_eps) / (mean_js_expanded + config.entropy_eps)
    js_weight_raw = torch.log1p(js_ratio) * mask_float
    quantile_clip_enabled = bool(getattr(config, "js_weight_quantile_clip", True))
    quantile_low = float(getattr(config, "js_weight_quantile_low", 0.05))
    quantile_high = float(getattr(config, "js_weight_quantile_high", 0.95))
    quantile_clip_stats = None
    if quantile_clip_enabled:
        js_weight_bounded, quantile_clip_stats = apply_rowwise_quantile_clip(
            values=js_weight_raw,
            token_mask=mask_bool,
            row_selector=exploration_valid,
            lower_quantile=quantile_low,
            upper_quantile=quantile_high,
        )
    else:
        js_weight_bounded = js_weight_raw
    js_weight_bounded = js_weight_bounded * mask_float
    js_weight_mean = js_weight_bounded.sum(dim=-1) / safe_num_tokens
    js_weights = js_weight_bounded / (js_weight_mean.unsqueeze(-1) + config.entropy_eps)
    js_weights = js_weights * mask_float
    js_weights = js_weights * exploration_valid.unsqueeze(-1).to(dtype=js_weights.dtype)

    if quantile_clip_stats is not None:
        print("[JS Advantage Reweighting] Quantile clip statistics:")
        print(
            f"  q=({quantile_low:.2f}, {quantile_high:.2f}), "
            f"lower_clip_rate={quantile_clip_stats['lower_clip_rate']:.4f}, "
            f"upper_clip_rate={quantile_clip_stats['upper_clip_rate']:.4f}, "
            f"lower_bound_mean={quantile_clip_stats['lower_bound_mean']:.4f}, "
            f"upper_bound_mean={quantile_clip_stats['upper_bound_mean']:.4f}"
        )

    # Apply weights: baseline rows are zeroed, exploration rows use the token-level reweighting.
    advantages_out = advantages.clone()
    if baseline_valid.any():
        advantages_out[baseline_valid] = 0
    if exploration_valid.any():
        A_token = A_traj.unsqueeze(-1) * js_weights
        A_token = torch.clamp(A_token, -config.entropy_adv_clip, config.entropy_adv_clip)
        advantages_out[exploration_valid] = A_token[exploration_valid]

    # Reward statistics (if available)
    count_baseline = baseline_valid.sum().item()
    count_exploration_js = exploration_valid.sum().item()
    count_baseline_correct = 0
    count_exploration_correct = 0
    if token_level_rewards is not None and valid_rows.any():
        last_valid_indices = torch.clamp(num_tokens.long() - 1, min=0)
        final_rewards = token_level_rewards.gather(1, last_valid_indices.unsqueeze(-1)).squeeze(-1)
        positive_reward = final_rewards > 0
        count_baseline_correct = (baseline_valid & positive_reward).sum().item()
        count_exploration_correct = (exploration_valid & positive_reward).sum().item()

    # Update batch
    js_reweight_factors = js_weights
    data.batch["advantages"] = advantages_out
    data.batch["js_reweight_factors"] = js_reweight_factors

    # Logging
    print(f"[JS Advantage Reweighting] Sample distribution:")
    print(
        f"  Baseline samples (count/group={baseline_count_per_group}, T={T0:.1f}): "
        f"{count_baseline}/{num_baseline} ({100*count_baseline/max(num_baseline,1):.1f}%) → weight=0"
    )
    print(
        f"  Exploration samples (count/group={exploration_count_per_group}, T={T1:.1f}): "
        f"{count_exploration_js}/{num_exploration} ({100*count_exploration_js/max(num_exploration,1):.1f}%) → weight=w(JS)"
    )

    # Reward statistics (if available)
    if token_level_rewards is not None:
        baseline_acc = 100 * count_baseline_correct / max(num_baseline, 1)
        exploration_acc = 100 * count_exploration_correct / max(num_exploration, 1)
        print(f"[JS Advantage Reweighting] Reward statistics:")
        print(f"  Baseline samples (T={T0:.1f}) Accuracy: {count_baseline_correct}/{num_baseline} ({baseline_acc:.1f}%)")
        print(f"  Exploration samples (T={T1:.1f}) Accuracy: {count_exploration_correct}/{num_exploration} ({exploration_acc:.1f}%)")
        if num_baseline > 0 and num_exploration > 0:
            acc_diff = exploration_acc - baseline_acc
            print(f"  Accuracy Difference (Exploration - Baseline): {acc_diff:+.1f}%")

    # JS divergence statistics
    exploration_token_mask = exploration_valid.unsqueeze(-1) & mask_bool
    js_values_valid = js_weights[exploration_token_mask]
    if js_values_valid.numel() > 0:
        js_values_valid = js_values_valid.float()
        mean_js = js_values_valid.mean().item()
        std_js = js_values_valid.std(unbiased=False).item()
        min_js = js_values_valid.min().item()
        max_js = js_values_valid.max().item()
        print(f"[JS Advantage Reweighting] JS weight statistics:")
        print(f"  Mean: {mean_js:.4f}, Std: {std_js:.4f}, Min: {min_js:.4f}, Max: {max_js:.4f}")

    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, and vLLM integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.OPO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
            AdvantageEstimator.GPG,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._init_adaptive_temperature_state()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)
        self._init_tampo_state()
        self._init_wallclock_study_state()

        # Initialize best checkpoint tracking for AIME metrics
        self.best_aime_score = -float('inf')  # Track the best sum of AIME 2024 and 2025 accuracy
        self.best_aime_ckpt_path = None  # Path to the best checkpoint

        # Initialize best checkpoint tracking for livecodebench metrics
        self.best_livecodebench_score = -float('inf')  # Track the best livecodebench/reward/mean@1 score
        self.best_livecodebench_ckpt_path = None  # Path to the best livecodebench checkpoint

    def _init_wallclock_study_state(self) -> None:
        self.wallclock_study = WallClockStudyRecorder(
            trainer_config=self.config.trainer,
            experiment_name=self.config.trainer.experiment_name,
        )

    def _init_tampo_state(self) -> None:
        self.tampo_enabled = False
        self.tampo_controller = None
        self.tampo_candidate_temperatures = []
        self.current_tampo_temperature = None
        self.current_tampo_temperature_index = None

        per_config = self.config.algorithm.per
        if per_config is None or not bool(getattr(per_config, "tampo_enable", False)):
            return

        self.tampo_enabled = True
        self.tampo_candidate_temperatures = [float(t) for t in per_config.tampo_candidate_temperatures]
        self.tampo_controller = TAMPOMetaPolicyController(
            temperatures=self.tampo_candidate_temperatures,
            ema_alpha=per_config.tampo_ema_alpha,
            top_p=per_config.tampo_top_p,
            seed=per_config.tampo_seed,
            eps=per_config.entropy_eps,
        )
        print(
            "[TAMPO] Enabled with candidate temperatures="
            f"{self.tampo_candidate_temperatures}, warmup_steps={self._get_tampo_warmup_steps()}, "
            f"warmup_temperature={per_config.tampo_warmup_temperature}, "
            f"ema_alpha={per_config.tampo_ema_alpha}, top_p={per_config.tampo_top_p}"
        )

    def _get_tampo_warmup_steps(self) -> int:
        per_config = self.config.algorithm.per
        if per_config is None:
            return 0
        configured_steps = int(getattr(per_config, "tampo_warmup_steps", 0) or 0)
        if configured_steps > 0:
            return configured_steps
        warmup_ratio = float(getattr(per_config, "tampo_warmup_ratio", 0.1))
        return max(0, int(round(self.total_training_steps * warmup_ratio)))

    def _select_tampo_temperature_for_step(self) -> dict[str, float]:
        if not self.tampo_enabled or self.tampo_controller is None:
            return {}

        per_config = self.config.algorithm.per
        warmup_steps = self._get_tampo_warmup_steps()
        in_tampo_warmup = warmup_steps > 0 and self.global_steps <= warmup_steps
        if in_tampo_warmup:
            temperature = float(per_config.tampo_warmup_temperature)
            temperature_index = -1
        else:
            temperature, temperature_index = self.tampo_controller.sample()

        self.current_tampo_temperature = float(temperature)
        self.current_tampo_temperature_index = int(temperature_index)
        print(
            f"[TAMPO] Step {self.global_steps}: selected temperature={temperature:.4f}, "
            f"index={temperature_index}, warmup={in_tampo_warmup}"
        )
        return {
            "tampo/current_temperature": float(temperature),
            "tampo/current_temperature_index": int(temperature_index),
            "tampo/in_warmup": float(in_tampo_warmup),
        }

    def _update_tampo_meta_policy(self, batch: DataProto) -> dict[str, float]:
        if not self.tampo_enabled or self.tampo_controller is None:
            return {}
        if "tampo_likelihoods" not in batch.batch:
            print("[TAMPO] Warning: tampo_likelihoods missing, skip meta-policy update")
            return {}

        response_mask = batch.batch["response_mask"].to(dtype=batch.batch["advantages"].dtype)
        valid_rows = response_mask.bool().any(dim=-1)
        if not valid_rows.any():
            print("[TAMPO] Warning: no valid response rows, skip meta-policy update")
            return {}

        safe_num_tokens = response_mask.sum(dim=-1).clamp(min=1.0)
        trajectory_advantages = (batch.batch["advantages"] * response_mask).sum(dim=-1) / safe_num_tokens
        trajectory_advantages = trajectory_advantages[valid_rows].float()
        likelihoods = batch.batch["tampo_likelihoods"][valid_rows].to(
            device=trajectory_advantages.device,
            dtype=torch.float32,
        )
        normalized_likelihoods = sparsemax(likelihoods, dim=-1)
        temperature_advantages = (normalized_likelihoods * trajectory_advantages.unsqueeze(-1)).mean(dim=0)
        temperature_advantages_np = temperature_advantages.detach().cpu().numpy()
        update_state = self.tampo_controller.update(temperature_advantages_np)

        policy = update_state["policy"]
        ema = update_state["temperature_advantage_ema"]
        sampled_policy_prob = 0.0
        if (
            self.current_tampo_temperature_index is not None
            and 0 <= self.current_tampo_temperature_index < len(policy)
        ):
            sampled_policy_prob = float(policy[self.current_tampo_temperature_index])

        metrics: dict[str, float] = {
            "tampo/batch_temperature_advantage_mean": float(np.mean(temperature_advantages_np)),
            "tampo/policy_entropy": float(-(policy * np.log(policy + 1e-12)).sum()),
            "tampo/sampled_temperature_policy_prob": sampled_policy_prob,
        }
        for temperature, prob, adv_ema in zip(self.tampo_candidate_temperatures, policy, ema, strict=True):
            temperature_key = str(temperature).replace(".", "_")
            metrics[f"tampo/policy_T_{temperature_key}"] = float(prob)
            metrics[f"tampo/adv_ema_T_{temperature_key}"] = float(adv_ema)

        print(
            f"[TAMPO] Step {self.global_steps}: updated meta-policy, "
            f"batch_adv={temperature_advantages_np.tolist()}, policy={policy.tolist()}"
        )
        return metrics

    def _record_wallclock_cycle(
        self,
        *,
        raw_cycle_step: int,
        is_effective_update: bool,
        num_gen_batches: int,
        timing_raw: dict[str, float],
        metrics: dict[str, float],
        note: str | None = None,
    ) -> None:
        self.wallclock_study.record_cycle(
            global_step=self.global_steps,
            raw_cycle_step=raw_cycle_step,
            is_effective_update=is_effective_update,
            num_gen_batches=num_gen_batches,
            timing_raw=timing_raw,
            metrics=metrics,
            note=note,
        )

    def _init_adaptive_temperature_state(self):
        self.current_t0 = None
        self.current_t1 = float(self.config.actor_rollout_ref.rollout.temperature)
        self.adaptive_t1_enabled = False
        self.temp_controller = None

        per_config = self.config.algorithm.per
        if per_config is None:
            return

        self.current_t0 = float(per_config.entropy_t0)
        self.current_t1 = float(per_config.entropy_t1)

        hybrid_cfg = self.config.actor_rollout_ref.rollout.get("hybrid_exploration", {})
        if hybrid_cfg.get("enable", False):
            temperatures = hybrid_cfg.get("temperatures", [self.current_t0, self.current_t1])
            if len(temperatures) >= 2:
                self.current_t0 = float(temperatures[0])
                self.current_t1 = float(temperatures[1])

        self.adaptive_t1_enabled = bool(getattr(per_config, "adaptive_t1_enable", False))
        if self.adaptive_t1_enabled:
            self.temp_controller = core_algos.AdaptiveTemperatureController(
                init_t1=self.current_t1,
                t1_min=per_config.adaptive_t1_min,
                t1_max=per_config.adaptive_t1_max,
                ema_beta=per_config.adaptive_t1_ema_beta,
                step_size=per_config.adaptive_t1_step_size,
                max_delta=per_config.adaptive_t1_max_delta,
            )
            self.current_t1 = float(self.temp_controller.value)

    def _get_runtime_temperatures(self) -> tuple[float, float]:
        if self.tampo_enabled and self.current_tampo_temperature is not None:
            temperature = float(self.current_tampo_temperature)
            return temperature, temperature
        per_config = self.config.algorithm.per
        if per_config is None:
            temperature = float(self.config.actor_rollout_ref.rollout.temperature)
            return temperature, temperature
        t0 = float(self.current_t0 if self.current_t0 is not None else per_config.entropy_t0)
        t1 = float(self.current_t1 if self.current_t1 is not None else per_config.entropy_t1)
        return t0, t1

    def _get_batch_sample_indices(self, batch: DataProto, device: torch.device) -> torch.Tensor:
        if "sample_indices" in batch.batch:
            return batch.batch["sample_indices"].to(device=device, dtype=torch.long)

        rollout_n = self.config.actor_rollout_ref.rollout.n
        uid = batch.non_tensor_batch["uid"]
        sample_indices = []
        uid_counts = {}
        for u in uid:
            count = uid_counts.get(u, 0)
            sample_indices.append(count % rollout_n)
            uid_counts[u] = count + 1
        return torch.tensor(sample_indices, device=device, dtype=torch.long)

    def _get_hybrid_baseline_count_per_group(self) -> int:
        return resolve_hybrid_group_counts(self.config.actor_rollout_ref.rollout)[0]

    def _ensure_runtime_sample_temperatures(
        self,
        batch: DataProto,
        step_t0: float,
        step_t1: float,
        in_warmup: bool,
    ) -> None:
        # Always keep the scalar fallback aligned with the runtime exploration temperature.
        batch.meta_info["temperature"] = float(step_t1)

        if "sample_temperatures" in batch.batch:
            return

        response_mask = batch.batch.get("response_mask", None)
        if response_mask is None:
            response_mask = batch.batch.get("responses", None)
        if response_mask is None:
            raise KeyError("Cannot infer device for sample_temperatures because neither response_mask nor responses exists")
        device = response_mask.device

        batch_size = batch.batch["responses"].size(0)
        if batch_size == 0:
            batch.batch["sample_temperatures"] = torch.empty(0, dtype=torch.float32, device=device)
            return

        hybrid_cfg = self.config.actor_rollout_ref.rollout.get("hybrid_exploration", {})
        hybrid_enabled = bool(hybrid_cfg.get("enable", False))
        rollout_n = int(getattr(self.config.actor_rollout_ref.rollout, "n", 1))

        if in_warmup or not hybrid_enabled or rollout_n < 2:
            sample_temperatures = torch.full((batch_size,), float(step_t1), dtype=torch.float32, device=device)
            source = "uniform_t1"
        else:
            baseline_count = self._get_hybrid_baseline_count_per_group()
            sample_indices = self._get_batch_sample_indices(batch, device=device)
            sample_temperatures = torch.where(
                sample_indices < baseline_count,
                torch.full_like(sample_indices, float(step_t0), dtype=torch.float32),
                torch.full_like(sample_indices, float(step_t1), dtype=torch.float32),
            )
            source = "reconstructed_from_sample_indices"

        batch.batch["sample_temperatures"] = sample_temperatures
        print(
            f"[Runtime Temperature Sync] Step {self.global_steps}: injected sample_temperatures "
            f"from {source} with T0={step_t0:.6f}, T1={step_t1:.6f}"
        )

    def _collect_adaptive_t1_stats(self, batch: DataProto) -> Optional[dict[str, float]]:
        if "js_weights" not in batch.batch or "token_level_rewards" not in batch.batch:
            return None

        response_mask = batch.batch["response_mask"]
        sample_indices = self._get_batch_sample_indices(batch, response_mask.device).cpu().tolist()
        baseline_count_per_group = self._get_hybrid_baseline_count_per_group()
        uid = batch.non_tensor_batch.get("uid", None)
        if uid is None:
            return None

        token_level_rewards = batch.batch["token_level_rewards"]
        js_weights = batch.batch["js_weights"]
        grouped_stats = defaultdict(lambda: {"baseline_rewards": [], "exploration_rewards": [], "exploration_js": []})

        for i, sample_index in enumerate(sample_indices):
            mask_seq = response_mask[i].bool()
            valid_indices = torch.where(mask_seq)[0]
            if len(valid_indices) == 0:
                continue

            final_reward = float(token_level_rewards[i][valid_indices[-1]].item())
            group = grouped_stats[uid[i]]

            if sample_index < baseline_count_per_group:
                group["baseline_rewards"].append(final_reward)
                continue

            js_seq = js_weights[i][mask_seq]
            if js_seq.numel() == 0:
                continue

            group["exploration_rewards"].append(final_reward)
            group["exploration_js"].append(float(js_seq.float().mean().item()))

        baseline_reward_means = []
        exploration_reward_means = []
        reward_gap_means = []
        raw_js_means = []
        utility_values = []

        for group in grouped_stats.values():
            baseline_rewards = group["baseline_rewards"]
            if not baseline_rewards or not group["exploration_rewards"] or not group["exploration_js"]:
                continue

            baseline_reward_mean = float(np.mean(baseline_rewards))
            exploration_reward_mean = float(np.mean(group["exploration_rewards"]))
            exploration_js_mean = float(np.mean(group["exploration_js"]))
            reward_gap = exploration_reward_mean - baseline_reward_mean
            utility = reward_gap * exploration_js_mean

            baseline_reward_means.append(baseline_reward_mean)
            exploration_reward_means.append(exploration_reward_mean)
            reward_gap_means.append(reward_gap)
            raw_js_means.append(exploration_js_mean)
            utility_values.append(utility)

        if not utility_values:
            return None

        return {
            "baseline_reward_mean": float(np.mean(baseline_reward_means)),
            "exploration_reward_mean": float(np.mean(exploration_reward_means)),
            "reward_gap_mean": float(np.mean(reward_gap_means)),
            "raw_js_mean": float(np.mean(raw_js_means)),
            "utility_step": float(np.mean(utility_values)),
        }

    def _update_adaptive_temperature(self, batch: DataProto) -> dict[str, float]:
        metrics = {
            "adaptive_t1/current_t0": float(self.current_t0) if self.current_t0 is not None else 0.0,
            "adaptive_t1/current_t1": float(self.current_t1),
        }

        if not self.adaptive_t1_enabled or self.temp_controller is None:
            return metrics

        per_config = self.config.algorithm.per
        if per_config is None or self.global_steps < per_config.warmup_steps:
            metrics["adaptive_t1/utility_signal"] = (
                float(self.temp_controller.utility_signal) if self.temp_controller.utility_signal is not None else 0.0
            )
            return metrics

        adaptive_stats = self._collect_adaptive_t1_stats(batch)
        if adaptive_stats is None:
            print(f"[Adaptive T1] Step {self.global_steps}: skipped update because adaptive statistics are unavailable")
            metrics["adaptive_t1/utility_signal"] = (
                float(self.temp_controller.utility_signal) if self.temp_controller.utility_signal is not None else 0.0
            )
            return metrics

        update_stats = self.temp_controller.update(adaptive_stats["utility_step"])
        self.current_t1 = float(self.temp_controller.value)
        metrics.update(
            {
                "adaptive_t1/baseline_reward_mean": adaptive_stats["baseline_reward_mean"],
                "adaptive_t1/exploration_reward_mean": adaptive_stats["exploration_reward_mean"],
                "adaptive_t1/reward_gap_mean": adaptive_stats["reward_gap_mean"],
                "adaptive_t1/raw_js_mean": adaptive_stats["raw_js_mean"],
                "adaptive_t1/utility_step": update_stats["utility_step"],
                "adaptive_t1/utility_signal": update_stats["utility_signal"],
                "adaptive_t1/delta_t1": update_stats["delta_t1"],
                "adaptive_t1/current_t1": update_stats["current_t1"],
            }
        )
        print(
            f"[Adaptive T1] Step {self.global_steps}: "
            f"baseline_reward_mean={adaptive_stats['baseline_reward_mean']:.6f}, "
            f"exploration_reward_mean={adaptive_stats['exploration_reward_mean']:.6f}, "
            f"reward_gap_mean={adaptive_stats['reward_gap_mean']:.6f}, "
            f"raw_js_mean={adaptive_stats['raw_js_mean']:.6f}, "
            f"utility_step={update_stats['utility_step']:.6f}, "
            f"utility_signal={update_stats['utility_signal']:.6f}, "
            f"delta_t1={update_stats['delta_t1']:+.6f}, next_T1={update_stats['current_t1']:.6f}"
        )
        return metrics

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
        if config.actor_rollout_ref.actor.strategy == "megatron":
            model_parallel_size = (
                config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size
                * config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
            )
            assert (
                n_gpus % (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size) == 0
            ), (
                f"n_gpus ({n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times "
                f"context_parallel_size ({config.actor_rollout_ref.actor.megatron.context_parallel_size})"
            )
            megatron_dp = n_gpus // (
                model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size
            )
            minimal_bsz = megatron_dp * config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
        else:
            minimal_bsz = n_gpus

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % minimal_bsz == 0, (
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size "
            f"({minimal_bsz})"
        )

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            """Validate mutually exclusive micro batch size configuration options.

            Ensures that users don't set both deprecated micro_batch_size and
            the new micro_batch_size_per_gpu parameters simultaneously.

            Args:
                mbs: Deprecated micro batch size parameter value.
                mbs_per_gpu: New micro batch size per GPU parameter value.
                name (str): Configuration section name for error messages.

            Raises:
                ValueError: If both parameters are set or neither is set.
            """
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(
                        f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'."
                    )

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(
                        f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                        f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
                    )

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(
                config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic"
            )

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(
                config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model"
            )

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert (
                    config.actor_rollout_ref.actor.ppo_mini_batch_size
                    % config.actor_rollout_ref.actor.ppo_micro_batch_size
                    == 0
                )
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if self.config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"} and (
            config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1
            or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1
        ):
            assert config.actor_rollout_ref.model.use_remove_padding, (
                "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."
            )

        if self.use_critic and config.critic.strategy in {"fsdp", "fsdp2"}:
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, (
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."
                )

        if config.data.get("val_batch_size", None) is not None:
            print(
                "WARNING: val_batch_size is deprecated."
                + " Validation datasets are sent to inference engines as a whole batch,"
                + " which will schedule the memory themselves."
            )

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, (
                "validation gen temperature should be greater than 0 when enabling do_sample"
            )

        hybrid_cfg = config.actor_rollout_ref.rollout.get("hybrid_exploration", {})
        if hybrid_cfg.get("enable", False):
            resolve_hybrid_group_counts(config.actor_rollout_ref.rollout)

        per_config = config.algorithm.per
        if per_config is not None and getattr(per_config, "tampo_enable", False):
            assert config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}, (
                "TAMPO currently supports FSDP actor training only"
            )
            assert not config.actor_rollout_ref.model.use_fused_kernels, (
                "TAMPO virtual-temperature likelihoods require raw logits and do not support fused kernels"
            )
            assert not per_config.enable_token_entropy_weighting, (
                "TAMPO is a step-level temperature baseline; disable ATJS token entropy weighting"
            )
            assert not getattr(per_config, "adaptive_t1_enable", False), (
                "TAMPO and adaptive_t1 are mutually exclusive"
            )
            hybrid_cfg = config.actor_rollout_ref.rollout.get("hybrid_exploration", {})
            assert not hybrid_cfg.get("enable", False), "TAMPO uses one sampled temperature per step; disable hybrid_exploration"
            assert len(per_config.tampo_candidate_temperatures) >= 1, "TAMPO requires candidate temperatures"

        if per_config is not None and getattr(per_config, "adaptive_t1_enable", False):
            assert config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}, (
                "Adaptive T1 currently supports FSDP actor training only"
            )
            assert not config.actor_rollout_ref.model.use_fused_kernels, (
                "Adaptive T1 does not support fused kernels because per-sample temperatures are required"
            )
            hybrid_cfg = config.actor_rollout_ref.rollout.hybrid_exploration
            assert hybrid_cfg.get("enable", False), "Adaptive T1 requires hybrid_exploration.enable=true"
            assert config.actor_rollout_ref.rollout.n >= 2, "Adaptive T1 requires rollout.n >= 2"
            assert len(hybrid_cfg.temperatures) >= 2, "Adaptive T1 requires at least two hybrid temperatures"
            assert abs(float(hybrid_cfg.temperatures[0]) - float(per_config.entropy_t0)) < 1e-8, (
                "adaptive_t1 expects hybrid_exploration.temperatures[0] to match algorithm.per.entropy_t0"
            )
            assert abs(float(hybrid_cfg.temperatures[1]) - float(per_config.entropy_t1)) < 1e-8, (
                "adaptive_t1 expects hybrid_exploration.temperatures[1] to match algorithm.per.entropy_t1"
            )

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files, self.config.data, self.tokenizer, self.processor
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files, self.config.data, self.tokenizer, self.processor
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _decode_token_text_for_visualization(self, token_id: int) -> str:
        try:
            return self.tokenizer.decode(
                [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
        except TypeError:
            return self.tokenizer.decode([token_id], skip_special_tokens=False)

    def _maybe_dump_js_token_visualization(self, batch: DataProto, step_t0: float, step_t1: float):
        per_config = self.config.algorithm.per
        if per_config is None or not per_config.get("js_token_visualization_enable", False):
            return

        if self.global_steps < per_config.warmup_steps:
            return

        dump_freq = int(per_config.get("js_token_visualization_freq", 0) or 0)
        if dump_freq <= 0 or self.global_steps % dump_freq != 0:
            return

        required_tensors = {"prompts", "responses", "response_mask", "js_weights"}
        missing = required_tensors - set(batch.batch.keys())
        if missing:
            print(f"[JS Token Visualization] Skipped because required tensors are missing: {sorted(missing)}")
            return

        uid = batch.non_tensor_batch.get("uid", None)
        if uid is None:
            print("[JS Token Visualization] Skipped because uid is missing")
            return

        output_dir = per_config.get("js_token_visualization_output_dir", None)
        if not output_dir:
            output_dir = os.path.join(self.config.trainer.default_local_dir, "js_token_visualization")
        os.makedirs(output_dir, exist_ok=True)

        max_groups = int(per_config.get("js_token_visualization_max_groups", 2))
        max_tokens = int(per_config.get("js_token_visualization_max_tokens_per_sample", 256))

        response_mask = batch.batch["response_mask"]
        responses = batch.batch["responses"]
        prompts = batch.batch["prompts"]
        raw_js = batch.batch["js_weights"]
        final_weights = batch.batch.get("js_reweight_factors", None)
        token_level_rewards = batch.batch.get("token_level_rewards", None)
        sample_indices = self._get_batch_sample_indices(batch, response_mask.device).cpu().tolist()

        groups = []
        group_lookup = {}
        for i, group_uid in enumerate(uid):
            group_uid_str = str(group_uid)
            if group_uid_str not in group_lookup:
                if len(groups) >= max_groups:
                    continue
                prompt_ids = prompts[i].detach().cpu().tolist()
                prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
                group_lookup[group_uid_str] = {
                    "uid": group_uid_str,
                    "prompt_text": prompt_text,
                    "samples": [],
                }
                groups.append(group_lookup[group_uid_str])

            if group_uid_str not in group_lookup:
                continue

            mask_seq = response_mask[i].detach().cpu().bool()
            valid_token_count = int(mask_seq.sum().item())
            if valid_token_count == 0:
                continue

            response_ids = responses[i].detach().cpu()[mask_seq].tolist()
            raw_js_values = raw_js[i].detach().cpu()[mask_seq].float().tolist()
            if final_weights is not None:
                final_weight_values = final_weights[i].detach().cpu()[mask_seq].float().tolist()
            else:
                final_weight_values = [None] * len(response_ids)

            reward_values = None
            final_reward = None
            if token_level_rewards is not None:
                reward_values = token_level_rewards[i].detach().cpu()[mask_seq].float().tolist()
                if reward_values:
                    final_reward = float(reward_values[-1])

            display_count = min(len(response_ids), max_tokens)
            tokens = []
            for token_idx in range(display_count):
                token_id = int(response_ids[token_idx])
                token_entry = {
                    "index": token_idx,
                    "token_id": token_id,
                    "token_text": self._decode_token_text_for_visualization(token_id),
                    "raw_js": float(raw_js_values[token_idx]),
                    "final_weight": None if final_weight_values[token_idx] is None else float(final_weight_values[token_idx]),
                }
                if reward_values is not None:
                    token_entry["token_reward"] = float(reward_values[token_idx])
                tokens.append(token_entry)

            top_tokens = []
            top_indices = sorted(range(len(response_ids)), key=lambda idx: raw_js_values[idx], reverse=True)[:12]
            for token_idx in top_indices:
                token_id = int(response_ids[token_idx])
                top_tokens.append(
                    {
                        "index": token_idx,
                        "token_id": token_id,
                        "token_text": self._decode_token_text_for_visualization(token_id),
                        "raw_js": float(raw_js_values[token_idx]),
                        "final_weight": None if final_weight_values[token_idx] is None else float(final_weight_values[token_idx]),
                    }
                )
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=False)
            final_weight_mean = None
            if final_weight_values and all(value is not None for value in final_weight_values):
                final_weight_mean = float(np.mean(final_weight_values))

            baseline_count_per_group = self._get_hybrid_baseline_count_per_group()
            group_lookup[group_uid_str]["samples"].append(
                {
                    "sample_index": int(sample_indices[i]),
                    "is_baseline": bool(sample_indices[i] < baseline_count_per_group),
                    "final_reward": final_reward,
                    "raw_js_mean": float(np.mean(raw_js_values)) if raw_js_values else None,
                    "final_weight_mean": final_weight_mean,
                    "response_text": response_text,
                    "tokens": tokens,
                    "top_tokens": top_tokens,
                    "displayed_token_count": display_count,
                    "total_token_count": valid_token_count,
                }
            )

        if not groups:
            print("[JS Token Visualization] Skipped because no eligible groups were collected")
            return

        for group in groups:
            group["samples"].sort(key=lambda sample: sample["sample_index"])

        base_filename = os.path.join(output_dir, f"step_{self.global_steps:06d}")
        html_path = f"{base_filename}.html"
        json_path = f"{base_filename}.json"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "step": self.global_steps,
                    "t0": step_t0,
                    "t1": step_t1,
                    "groups": groups,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        with open(html_path, "w", encoding="utf-8") as f:
            f.write(
                render_js_token_visualization_html(
                    step=self.global_steps,
                    t0=step_t0,
                    t1=step_t1,
                    groups=groups,
                )
            )

        print(f"[JS Token Visualization] Dumped HTML to {html_path}")
        print(f"[JS Token Visualization] Dumped JSON to {json_path}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        sample_turns = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "interaction_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("interaction_kwargs")
            if "agent_name" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("agent_name")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            print(f"len reward_extra_infos_dict['reward']: {len(reward_extra_infos_dict['reward'])}")
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)
                    print(f"len reward_extra_infos_dict['{key}']: {len(reward_extra_infos_dict[key])}")

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        # Filter out incomplete fields from reward_extra_infos_dict
        # Some fields may not be present in all samples, so we only keep fields that are either empty or have the same length as sample_scores
        filtered_reward_extra_infos_dict = {}
        for key_info, lst in reward_extra_infos_dict.items():
            if len(lst) == 0 or len(lst) == len(sample_scores):
                filtered_reward_extra_infos_dict[key_info] = lst
            else:
                print(f"Warning: Skipping field '{key_info}' with length {len(lst)} (expected {len(sample_scores)})")
        reward_extra_infos_dict = filtered_reward_extra_infos_dict

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        # Calculate the sum of AIME 2024 and 2025 accuracy metrics
        aime2024_key = "val-core/aime2024/acc/mean@16"
        aime2025_key = "val-core/aime2025/acc/mean@16"

        if aime2024_key in metric_dict and aime2025_key in metric_dict:
            aime_score = metric_dict[aime2024_key] + metric_dict[aime2025_key]
            metric_dict["val-core/aime_combined_score"] = aime_score
            print(f"AIME combined score: {aime_score} (2024: {metric_dict[aime2024_key]}, 2025: {metric_dict[aime2025_key]})")
        else:
            aime_score = None
            if aime2024_key not in metric_dict:
                print(f"Warning: {aime2024_key} not found in metrics")
            if aime2025_key not in metric_dict:
                print(f"Warning: {aime2025_key} not found in metrics")

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
                profile_option=self.config.trainer.npu_profile.options,
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
                profile_option=self.config.trainer.npu_profile.options,
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.trainer, "profile_steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.trainer, "profile_steps")
            assert OmegaConf.select(self.config.trainer, "worker_nsight_options") is not None, (
                "worker_nsight_options must be set when profile_steps is set"
            )
            wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                OmegaConf.select(self.config.trainer, "worker_nsight_options")
            )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,
            )

    def _save_best_checkpoint(self, aime_score=None, livecodebench_score=None):
        """
        Save the checkpoint if it achieves a better score for either AIME or livecodebench metrics.

        Args:
            aime_score: The sum of AIME 2024 and 2025 accuracy metrics.
            livecodebench_score: The livecodebench/reward/mean@1 score.
        """
        from verl.utils.fs import local_mkdir_safe

        def save_staged_best_checkpoint(
            ckpt_dir_name: str,
            score_value: float,
            score_print_name: str,
            score_file_label: str,
        ) -> str:
            best_ckpt_dir = os.path.join(self.config.trainer.default_local_dir, ckpt_dir_name)
            staging_ckpt_dir = os.path.join(
                self.config.trainer.default_local_dir,
                f".{ckpt_dir_name}_staging_step_{self.global_steps}_{uuid.uuid4().hex[:8]}",
            )
            backup_ckpt_dir = None
            promoted = False

            local_mkdir_safe(staging_ckpt_dir)

            try:
                actor_local_path = os.path.join(staging_ckpt_dir, "actor")
                actor_remote_path = (
                    None
                    if self.config.trainer.default_hdfs_dir is None
                    else os.path.join(self.config.trainer.default_hdfs_dir, ckpt_dir_name, "actor")
                )

                print(f"Saving {score_print_name} checkpoint with score {score_value:.4f} at step {self.global_steps}")
                # Save to a fresh staging directory first so the previous best remains intact until the new checkpoint
                # is fully materialized across all ranks.
                self.actor_rollout_wg.save_checkpoint(
                    actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=None
                )

                if self.use_critic:
                    critic_local_path = os.path.join(staging_ckpt_dir, "critic")
                    critic_remote_path = (
                        None
                        if self.config.trainer.default_hdfs_dir is None
                        else os.path.join(self.config.trainer.default_hdfs_dir, ckpt_dir_name, "critic")
                    )
                    self.critic_wg.save_checkpoint(
                        critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=None
                    )

                metadata_path = os.path.join(staging_ckpt_dir, "best_checkpoint_info.txt")
                with open(metadata_path, "w") as f:
                    f.write(f"{score_file_label}: {score_value:.4f}\n")
                    f.write(f"Global Step: {self.global_steps}\n")

                if os.path.exists(best_ckpt_dir):
                    backup_ckpt_dir = f"{best_ckpt_dir}.backup_{uuid.uuid4().hex[:8]}"
                    os.rename(best_ckpt_dir, backup_ckpt_dir)

                os.rename(staging_ckpt_dir, best_ckpt_dir)
                promoted = True
                return best_ckpt_dir
            except Exception:
                if backup_ckpt_dir and os.path.exists(backup_ckpt_dir) and not os.path.exists(best_ckpt_dir):
                    try:
                        os.rename(backup_ckpt_dir, best_ckpt_dir)
                    except OSError:
                        pass

                if os.path.exists(staging_ckpt_dir):
                    shutil.rmtree(staging_ckpt_dir, ignore_errors=True)
                raise
            finally:
                if promoted and backup_ckpt_dir and os.path.exists(backup_ckpt_dir):
                    shutil.rmtree(backup_ckpt_dir, ignore_errors=True)

        # Handle AIME checkpoint saving
        if aime_score is not None and aime_score > self.best_aime_score:
            self.best_aime_score = aime_score
            self.best_aime_ckpt_path = save_staged_best_checkpoint(
                ckpt_dir_name="best_aime_ckpt",
                score_value=aime_score,
                score_print_name="best AIME",
                score_file_label="Best AIME Combined Score",
            )

        # Handle livecodebench checkpoint saving
        if livecodebench_score is not None and livecodebench_score > self.best_livecodebench_score:
            self.best_livecodebench_score = livecodebench_score
            self.best_livecodebench_ckpt_path = save_staged_best_checkpoint(
                ckpt_dir_name="best_livecodebench_ckpt",
                score_value=livecodebench_score,
                score_print_name="best livecodebench",
                score_file_label="Best livecodebench/reward/mean@1 Score",
            )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        trainer_state_local_path = os.path.join(local_global_step_folder, "trainer_state.pt")
        trainer_state = {
            "adaptive_t1": {
                "enabled": self.adaptive_t1_enabled,
                "current_t0": self.current_t0,
                "current_t1": self.current_t1,
                "controller_state": self.temp_controller.state_dict() if self.temp_controller is not None else None,
            },
            "tampo": {
                "enabled": self.tampo_enabled,
                "current_temperature": self.current_tampo_temperature,
                "current_temperature_index": self.current_tampo_temperature_index,
                "controller_state": self.tampo_controller.state_dict() if self.tampo_controller is not None else None,
            }
        }
        torch.save(trainer_state, trainer_state_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

        trainer_state_local_path = os.path.join(global_step_folder, "trainer_state.pt")
        if os.path.exists(trainer_state_local_path):
            trainer_state = torch.load(trainer_state_local_path, weights_only=False)
            adaptive_state = trainer_state.get("adaptive_t1", {})
            if adaptive_state:
                self.current_t0 = adaptive_state.get("current_t0", self.current_t0)
                self.current_t1 = float(adaptive_state.get("current_t1", self.current_t1))
                controller_state = adaptive_state.get("controller_state", None)
                if self.temp_controller is not None and controller_state is not None:
                    self.temp_controller.load_state_dict(controller_state)
                    self.current_t1 = float(self.temp_controller.value)
                print(
                    f"[Adaptive T1] Restored controller state: "
                    f"T0={self.current_t0}, T1={self.current_t1}, "
                    f"utility_signal={None if self.temp_controller is None else self.temp_controller.utility_signal}"
                )
            tampo_state = trainer_state.get("tampo", {})
            if tampo_state and self.tampo_controller is not None:
                controller_state = tampo_state.get("controller_state", None)
                if controller_state is not None:
                    self.tampo_controller.load_state_dict(controller_state)
                self.current_tampo_temperature = tampo_state.get("current_temperature", self.current_tampo_temperature)
                self.current_tampo_temperature_index = tampo_state.get(
                    "current_temperature_index",
                    self.current_tampo_temperature_index,
                )
                print(
                    f"[TAMPO] Restored controller state: "
                    f"current_temperature={self.current_tampo_temperature}, "
                    f"policy={self.tampo_controller.policy.tolist()}"
                )

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile()
            if self.use_critic:
                self.critic_wg.start_profile()
            if self.use_rm:
                self.rm_wg.start_profile()

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                do_profile = (
                    self.global_steps in self.config.trainer.profile_steps
                    if self.config.trainer.profile_steps is not None
                    else False
                )
                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(do_profile)

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
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
                if "index" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("index")
                if "agent_name" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("agent_name")

                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                # Pass global_steps and warmup status to rollout
                gen_batch.meta_info["global_steps"] = self.global_steps
                metrics.update(self._select_tampo_temperature_for_step())

                # Only ATJS-style features should activate rollout warmup behavior.
                in_warmup = False
                if self.config.algorithm.per is not None:
                    per_cfg = self.config.algorithm.per
                    warmup_active = bool(
                        per_cfg.enable_token_entropy_weighting or getattr(per_cfg, "adaptive_t1_enable", False)
                    )
                    if warmup_active:
                        warmup_steps = per_cfg.warmup_steps
                        in_warmup = self.global_steps < warmup_steps
                        gen_batch.meta_info["in_warmup"] = in_warmup
                        gen_batch.meta_info["warmup_steps"] = warmup_steps

                step_t0, step_t1 = self._get_runtime_temperatures()
                gen_batch.meta_info["runtime_hybrid_temperatures"] = [step_t0, step_t1]
                gen_batch.meta_info["temperature"] = step_t1
                if self.tampo_enabled:
                    gen_batch.meta_info["runtime_sampling_temperature"] = step_t1
                print(
                    f"[Adaptive T1] Step {self.global_steps}: using T0={step_t0:.6f}, "
                    f"T1={step_t1:.6f}, warmup={in_warmup}"
                )
                metrics.update(
                    {
                        "adaptive_t1/used_t0": step_t0,
                        "adaptive_t1/used_t1": step_t1,
                    }
                )

                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                    )
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)
                    self._ensure_runtime_sample_temperatures(
                        batch=batch,
                        step_t0=step_t0,
                        step_t1=step_t1,
                        in_warmup=in_warmup,
                    )

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        # Pass JS weighting config to worker via meta_info
                        if self.config.algorithm.per is not None and self.config.algorithm.per.enable_token_entropy_weighting:
                            js_enable = self.global_steps >= self.config.algorithm.per.warmup_steps
                            batch.meta_info["js_weighting_config"] = {
                                "enable": js_enable,
                                "T0": step_t0,
                                "T1": step_t1,
                            }
                            print(f"[DEBUG Trainer] Passing JS config to worker: T0={step_t0}, T1={step_t1}")
                        if self.tampo_enabled:
                            batch.meta_info["tampo_candidate_temperatures"] = self.tampo_candidate_temperatures

                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode

                        # Compute overall entropy (for backward compatibility)
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}

                        # If using token entropy weighting, compute separate entropy for baseline and exploration samples
                        if (self.config.algorithm.per is not None and
                            self.config.algorithm.per.enable_token_entropy_weighting and
                            self.global_steps >= self.config.algorithm.per.warmup_steps):

                            rollout_n = self.config.actor_rollout_ref.rollout.n
                            if rollout_n >= 2:
                                sample_indices = self._get_batch_sample_indices(batch, entropys.device)
                                baseline_count_per_group = self._get_hybrid_baseline_count_per_group()

                                # Identify sample types
                                is_baseline = sample_indices < baseline_count_per_group
                                is_exploration = ~is_baseline

                                # Compute entropy for baseline samples (T=T0)
                                if is_baseline.sum() > 0:
                                    baseline_mask = is_baseline.unsqueeze(-1) * response_masks  # [batch_size, response_length]
                                    entropy_baseline = agg_loss(loss_mat=entropys, loss_mask=baseline_mask, loss_agg_mode=loss_agg_mode)
                                    old_log_prob_metrics["actor/entropy_baseline"] = entropy_baseline.detach().item()

                                # Compute entropy for exploration samples (T=T1).
                                if is_exploration.sum() > 0:
                                    exploration_mask = is_exploration.unsqueeze(-1) * response_masks  # [batch_size, response_length]
                                    entropy_exploration = agg_loss(loss_mat=entropys, loss_mask=exploration_mask, loss_agg_mode=loss_agg_mode)
                                    old_log_prob_metrics["actor/entropy_exploration"] = entropy_exploration.detach().item()

                                    # Override actor/entropy with exploration-only entropy (more meaningful metric)
                                    old_log_prob_metrics["actor/entropy"] = entropy_exploration.detach().item()

                        metrics.update(old_log_prob_metrics)
                        # Keep entropys for entropy-based advantage correction (don't pop it)
                        # old_log_prob.batch.pop("entropys")  # Commented out
                        batch = batch.union(old_log_prob)
                        self._ensure_runtime_sample_temperatures(
                            batch=batch,
                            step_t0=step_t0,
                            step_t1=step_t1,
                            in_warmup=in_warmup,
                        )

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            rollout_old_log_probs = batch.batch["rollout_log_probs"]
                            actor_old_log_probs = batch.batch["old_log_probs"]
                            attention_mask = batch.batch["attention_mask"]
                            responses = batch.batch["responses"]
                            response_length = responses.size(1)
                            response_mask = attention_mask[:, -response_length:]

                            rollout_probs = torch.exp(rollout_old_log_probs)
                            actor_probs = torch.exp(actor_old_log_probs)
                            rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                            rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                            rollout_probs_diff_max = torch.max(rollout_probs_diff)
                            rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                            rollout_probs_diff_std = torch.std(rollout_probs_diff)
                            metrics.update(
                                {
                                    "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                    "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                    "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                }
                            )

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, color="olive"):
                            self._ensure_runtime_sample_temperatures(
                                batch=batch,
                                step_t0=step_t0,
                                step_t1=step_t1,
                                in_warmup=in_warmup,
                            )
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                            global_steps=self.global_steps,
                            total_training_steps=self.total_training_steps,
                        )

                        if self.tampo_enabled:
                            metrics.update(self._update_tampo_meta_policy(batch))

                        # Apply token-level JS weighting if enabled
                        if self.config.algorithm.per is not None:
                            batch = apply_token_js_weighting(
                                batch,
                                self.config.algorithm.per,
                                self.config.actor_rollout_ref.rollout,
                                global_steps=self.global_steps,
                                temperature_override=(step_t0, step_t1),
                            )
                            self._maybe_dump_js_token_visualization(batch, step_t0=step_t0, step_t1=step_t1)

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            self._ensure_runtime_sample_temperatures(
                                batch=batch,
                                step_t0=step_t0,
                                step_t1=step_t1,
                                in_warmup=in_warmup,
                            )
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    adaptive_temperature_metrics = self._update_adaptive_temperature(batch)
                    metrics.update(adaptive_temperature_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            if "request_id" in batch.non_tensor_batch:
                                reward_extra_infos_dict.setdefault(
                                    "request_id",
                                    batch.non_tensor_batch["request_id"].tolist(),
                                )
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("testing", timing_raw, color="green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                        # Save best checkpoint if applicable
                        aime_score = val_metrics.get("val-core/aime_combined_score", None)
                        livecodebench_score = val_metrics.get("val-core/livecodebench/reward/mean@1", None)

                        if aime_score is not None or livecodebench_score is not None:
                            self._save_best_checkpoint(aime_score=aime_score, livecodebench_score=livecodebench_score)

                    # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                    esi_close_to_expiration = should_save_ckpt_esi(
                        max_steps_duration=self.max_steps_duration,
                        redundant_time=self.config.trainer.esi_redundant_time,
                    )
                    # Check if the conditions for saving a checkpoint are met.
                    # The conditions include a mandatory condition (1) and
                    # one of the following optional conditions (2/3/4):
                    # 1. The save frequency is set to a positive value.
                    # 2. It's the last training step.
                    # 3. The current step number is a multiple of the save frequency.
                    # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                    if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                        or esi_close_to_expiration
                    ):
                        if esi_close_to_expiration:
                            print("Force saving checkpoint: ESI instance expiration approaching.")
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    self._stop_profiling(do_profile)

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                self._record_wallclock_cycle(
                    raw_cycle_step=self.global_steps,
                    is_effective_update=True,
                    num_gen_batches=1,
                    timing_raw=dict(timing_raw),
                    metrics=metrics,
                )

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
