#!/bin/bash
# ============================================================
# DAPO + JS-Divergence Token-Level Weighting Training Script
# ============================================================
# This script runs DAPO with JS-divergence token-level advantage weighting
#
# Key Features:
# 1. Mixed Temperature Sampling:
#    - Sample 0: T=0.1 (baseline, deterministic)
#    - Sample 1-3: T=1.2 (exploration, stochastic)
#
# 2. Token-Level JS-Divergence Weighting (after 20-step warmup):
#    - Sample 0: weight=0 (不参与梯度更新，仅提供baseline)
#    - Sample 1+: weight=JS[t]/mean(JS) (JS散度加权)
#
# 3. Adaptive Degradation:
#    - Early training: High JS variance → Strong token-level weighting
#    - Late training: Low JS variance → Weights approach 1.0 → Degrades to DAPO
#
# 4. 20-Step Warmup:
#    - First 20 steps: All samples use T=1.2, standard DAPO
#    - After step 20: Enable mixed temperature + JS weighting
# ============================================================


PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data}"
MODEL_DIR="${MODEL_DIR:-${PROJECT_ROOT}/models}"
CKPT_DIR="${CKPT_DIR:-${PROJECT_ROOT}/checkpoints}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${PROJECT_ROOT}/tensorboard}"

cd "${PROJECT_ROOT}"
PRETRAINED_MODEL=${MODEL_DIR}/Qwen3-8B
n_nodes=1
n_gpus_per_node=8
tensor_model_parallel_size=8
save_freq=200

dapo_train_path=${DATA_DIR}/deepscaler.parquet
r1_test_path=${DATA_DIR}/validation.parquet

experiment_name="dapo-js-r4-8B-$(date +%Y%m%d_%H%M)"
max_prompt_length=2048
max_response_length=8192
OUTPUT_DIR=${CKPT_DIR%/}/${experiment_name}
LOG_DIR=${LOG_DIR%/}/${experiment_name}
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train.log"

# 设置使用本地修改的 vLLM
LOCAL_VLLM_PATH=${LOCAL_VLLM_PATH:-}
export PYTHONPATH="${LOCAL_VLLM_PATH}:${PYTHONPATH}"
echo "使用本地 vLLM: ${LOCAL_VLLM_PATH}"

export VLLM_SOFT_THINK_DEBUG=1
export PYTHONUNBUFFERED=1
exec > >(tee -a "$LOG_FILE") 2>&1

set -x

# ============================================================
# DAPO Configuration
# ============================================================
rollout_n=4  # Number of samples per prompt (1 baseline + 3 exploration)

# DAPO Asymmetric Clipping
clip_ratio_low=0.2
clip_ratio_high=0.28

# Loss Aggregation (DAPO feature)
loss_agg_mode="token-mean"

# Dynamic Sampling Configuration (DAPO feature)
enable_filter_groups=True
filter_groups_metric=acc
max_num_gen_batches=10

# ============================================================
# JS-Divergence Token-Level Weighting Configuration
# ============================================================
USE_TOKEN_ENTROPY_WEIGHTING=true  # Enable JS weighting
USE_BASELINE_ZERO_WEIGHT=true     # Sample 0 weight=0 (不参与梯度更新)
USE_HYBRID_EXPLORATION=true       # Enable mixed temperature sampling

# Warmup Configuration
warmup_steps=20  # First 20 steps use standard DAPO

# Temperature Configuration
entropy_t0=0.3   # Baseline temperature (Sample 0)
entropy_t1=1.2   # Exploration temperature (Sample 1+)

# JS Weighting Parameters
entropy_adv_clip=10.0  # Advantage clipping threshold
entropy_eps=1e-8       # Numerical stability constant

# Sampling Configuration (Mixed Temperature)
# Note: During warmup, all samples use T=1.2
# After warmup, hybrid_exploration takes effect
temperature=1.2  # Default rollout temperature (used during warmup)
top_p=1.0
top_k=-1

# ============================================================
# Qwen3 Thinking Mode Control
# ============================================================
ENABLE_THINKING_MODE=true

# ============================================================
# Length Penalty Configuration
# ============================================================
export VERL_ENABLE_LENGTH_PENALTY=false
export MAX_RESPONSE_LENGTH=8192

# ============================================================
# Run DAPO + JS-Divergence Training
# ============================================================
python3 -m recipe.dapo.main_dapo_js \
    algorithm.adv_estimator=grpo \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.per.enable_token_entropy_weighting=${USE_TOKEN_ENTROPY_WEIGHTING} \
    algorithm.per.warmup_steps=${warmup_steps} \
    algorithm.per.entropy_t0=${entropy_t0} \
    algorithm.per.entropy_t1=${entropy_t1} \
    algorithm.per.entropy_adv_clip=${entropy_adv_clip} \
    algorithm.per.entropy_eps=${entropy_eps} \
    actor_rollout_ref.rollout.hybrid_exploration.enable=${USE_HYBRID_EXPLORATION} \
    actor_rollout_ref.rollout.hybrid_exploration.temperatures=[${entropy_t0},${entropy_t1}] \
    actor_rollout_ref.rollout.hybrid_exploration.top_k_values=[-1,-1] \
    actor_rollout_ref.rollout.hybrid_exploration.use_baseline_zero_weight=${USE_BASELINE_ZERO_WEIGHT} \
    data.train_files=$dapo_train_path \
    data.val_files=$r1_test_path \
    data.train_batch_size=128 \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.truncation='left' \
    +data.apply_chat_template_kwargs.enable_thinking=$ENABLE_THINKING_MODE \
    +actor_rollout_ref.actor.tb_type=tempered_important_sampling \
    actor_rollout_ref.model.path=$PRETRAINED_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=2 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tensor_model_parallel_size \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.n=$rollout_n \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name='cceRL' \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$n_nodes \
    trainer.resume_mode=auto \
    trainer.save_freq=$save_freq \
    trainer.val_before_train=False \
    trainer.default_local_dir=$OUTPUT_DIR \
    trainer.test_freq=5 \
    trainer.total_epochs=2 \
    $@

