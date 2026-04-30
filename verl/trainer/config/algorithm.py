# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from dataclasses import dataclass, field
from typing import Optional

from verl.base_config import BaseConfig


@dataclass
class KLControlConfig(BaseConfig):
    """Configuration for KL control.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        type (str): Type of KL control. Can be "fixed" or "adaptive".
        kl_coef (float): Initial coefficient for KL penalty.
        horizon (int): Horizon value for adaptive controller.
        target_kl (float): Target KL divergence for adaptive controller.
    """

    _frozen_fields = ["type", "kl_coef", "horizon", "target_kl"]
    type: str = "fixed"
    kl_coef: float = 0.001
    horizon: int = 10000
    target_kl: float = 0.1


@dataclass
class PFPPOConfig(BaseConfig):
    """Configuration for preference feedback PPO.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        reweight_method (str): Method for reweighting samples. Can be "pow", "max_min", or "max_random".
        weight_pow (float): Power used for weight scaling in "pow" method.
    """

    _frozen_fields = ["reweight_method", "weight_pow"]
    reweight_method: str = "pow"
    weight_pow: float = 2.0


@dataclass
class FilterGroupsConfig(BaseConfig):
    """Configuration for filter groups (used in DAPO and Entropy).

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        enable (bool): Whether to enable filter groups.
        metric (Optional[str]): Metric to use for filtering: "acc", "score", "seq_reward", "seq_final_reward", etc.
        max_num_gen_batches (int): Non-positive values mean no upper limit.
    """

    _frozen_fields = ["enable", "metric", "max_num_gen_batches"]

    enable: bool = False
    metric: Optional[str] = None
    max_num_gen_batches: int = 0


@dataclass
class PERConfig(BaseConfig):
    """Configuration for Token-level JS-Divergence-Weighted Advantage Reweighting.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Baseline + Exploration Strategy:
        When enable_token_entropy_weighting=True, advantages are reweighted as follows:

        - Sample 0 (T=T0, e.g., 0.3): Baseline, weight=0 (不参与梯度更新，仅提供baseline)
        - Sample 1+ (T=T1, e.g., 1.2): Exploration samples
            - ALL samples use JS-divergence weighting (no A≥0 vs A<0 distinction)

        Token-level JS-Divergence Weighting (for ALL exploration samples):
            JS[t] = 0.5 * (KL(P_T0 || M) + KL(P_T1 || M))
                where M = 0.5 * (P_T0 + P_T1)
            w[t] = JS[t] / mean(JS)  # Normalize to mean=1
            A_token[t] = A_traj * w[t]  # Apply weight

        This ensures:
            - High JS tokens (high temperature disagreement) get w[t] > 1 (more advantage)
            - Low JS tokens (low temperature disagreement) get w[t] < 1 (less advantage)
            - Mean weight = 1, so total advantage is approximately conserved

    Args:
        enable_token_entropy_weighting (bool): Enable baseline+exploration advantage reweighting. Default: False
        warmup_steps (int): Number of warmup steps using standard GRPO (all samples at T1). Default: 20
        entropy_t0 (float): Baseline temperature (T0) for Sample 0. Default: 0.3
        entropy_t1 (float): Exploratory temperature (T1) for Sample 1+. Default: 1.2
        entropy_adv_clip (float): Clip final token-level advantages to [-clip, clip]. Default: 10.0
        entropy_eps (float): Small constant for numerical stability. Default: 1e-8
        js_weight_lambda (float): Legacy compatibility field retained for old configs. The current ATJS1ctrl gate
            is fixed to ratio -> log1p -> quantile -> renorm and ignores this value.
        js_weight_quantile_clip (bool): Whether to apply row-wise quantile clipping before renormalization.
        js_weight_quantile_low (float): Lower row-wise quantile used for clipping token JS weights.
        js_weight_quantile_high (float): Upper row-wise quantile used for clipping token JS weights.
        js_weight_min (float): Legacy compatibility field retained for old configs; current gate no longer uses it.
        js_weight_max (float): Legacy compatibility field retained for old configs; current gate no longer uses it.
    """

    _frozen_fields = [
        "enable_token_entropy_weighting",
        "warmup_steps",
        "entropy_t0",
        "entropy_t1",
        "entropy_adv_clip",
        "entropy_eps",
        "js_weight_lambda",
        "js_weight_quantile_clip",
        "js_weight_quantile_low",
        "js_weight_quantile_high",
        "js_weight_min",
        "js_weight_max",
        "adaptive_t1_enable",
        "adaptive_t1_target_raw_js_mean",
        "adaptive_t1_ema_beta",
        "adaptive_t1_step_size",
        "adaptive_t1_max_delta",
        "adaptive_t1_min",
        "adaptive_t1_max",
        "tampo_enable",
        "tampo_candidate_temperatures",
        "tampo_warmup_steps",
        "tampo_warmup_ratio",
        "tampo_warmup_temperature",
        "tampo_ema_alpha",
        "tampo_top_p",
        "tampo_seed",
        "js_token_visualization_enable",
        "js_token_visualization_freq",
        "js_token_visualization_max_groups",
        "js_token_visualization_max_tokens_per_sample",
        "js_token_visualization_output_dir",
    ]

    # Token-level JS-divergence-weighted advantage reweighting configuration
    enable_token_entropy_weighting: bool = False
    warmup_steps: int = 20  # Number of warmup steps using standard GRPO
    entropy_t0: float = 0.3  # Baseline temperature for Sample 0
    entropy_t1: float = 1.2  # Exploratory temperature for Sample 1+
    entropy_adv_clip: float = 10.0
    entropy_eps: float = 1e-8
    js_weight_lambda: float = 0.0  # Legacy compatibility; current gate ignores this knob.
    js_weight_quantile_clip: bool = True
    js_weight_quantile_low: float = 0.05
    js_weight_quantile_high: float = 0.95
    js_weight_min: float = 0.2  # Legacy compatibility; current gate no longer uses clamp.
    js_weight_max: float = 5.0  # Legacy compatibility; current gate no longer uses clamp.
    adaptive_t1_enable: bool = False
    adaptive_t1_target_raw_js_mean: float = 0.03
    adaptive_t1_ema_beta: float = 0.9
    adaptive_t1_step_size: float = 0.5
    adaptive_t1_max_delta: float = 0.05
    adaptive_t1_min: float = 0.6
    adaptive_t1_max: float = 1.8
    tampo_enable: bool = False
    tampo_candidate_temperatures: list[float] = field(
        default_factory=lambda: [round(0.6 + 0.1 * i, 1) for i in range(10)]
    )
    tampo_warmup_steps: int = 0  # 0 means use tampo_warmup_ratio * total_training_steps.
    tampo_warmup_ratio: float = 0.1
    tampo_warmup_temperature: float = 1.0
    tampo_ema_alpha: float = 0.05
    tampo_top_p: float = 0.7
    tampo_seed: int = 0
    js_token_visualization_enable: bool = False
    js_token_visualization_freq: int = 0
    js_token_visualization_max_groups: int = 2
    js_token_visualization_max_tokens_per_sample: int = 256
    js_token_visualization_output_dir: Optional[str] = None


@dataclass
class AlgoConfig(BaseConfig):
    """Configuration for the algorithm.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        gamma (float): Discount factor for future rewards.
        lam (float): Trade-off between bias and variance in the GAE estimator.
        adv_estimator (str): Advantage estimator type: "gae", "grpo", "reinforce_plus_plus", etc.
        norm_adv_by_std_in_grpo (bool): Whether to normalize advantages by std (specific to GRPO).
        use_kl_in_reward (bool): Whether to enable in-reward KL penalty.
        kl_penalty (str): How to estimate KL divergence: "kl", "abs", "mse", "low_var_kl", or "full".
        kl_ctrl (KLControlConfig): KL control configuration.
        use_pf_ppo (bool): Whether to enable preference feedback PPO.
        pf_ppo (Optional[PFPPOConfig]): Preference feedback PPO settings.
        filter_groups (Optional[FilterGroupsConfig]): Filter groups configuration, used in DAPO and Entropy
        per (Optional[PERConfig]): Prioritized Experience Replay configuration for CoT exploration
    """

    _frozen_fields = [
        "gamma",
        "lam",
        "adv_estimator",
        "norm_adv_by_std_in_grpo",
        "use_kl_in_reward",
        "kl_penalty",
        "use_pf_ppo",
    ]

    gamma: float = 1.0
    lam: float = 1.0
    adv_estimator: str = "gae"
    norm_adv_by_std_in_grpo: bool = True
    use_kl_in_reward: bool = False
    kl_penalty: str = "kl"
    kl_ctrl: KLControlConfig = field(default_factory=KLControlConfig)
    use_pf_ppo: bool = False
    pf_ppo: Optional[PFPPOConfig] = None
    filter_groups: Optional[FilterGroupsConfig] = None
    per: Optional[PERConfig] = None
