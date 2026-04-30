#!/usr/bin/env bash

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data}"
MODEL_DIR="${MODEL_DIR:-${PROJECT_ROOT}/models}"
CKPT_DIR="${CKPT_DIR:-${PROJECT_ROOT}/checkpoints}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${PROJECT_ROOT}/tensorboard}"

set -euo pipefail

cd "${PROJECT_ROOT}"
ROOT_DIR=${PROJECT_ROOT}
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-cpl-ATJS1ctrl-r4-14B-${RUN_TAG}}"
OUTPUT_DIR="${ROOT_DIR}/ckpt/${EXPERIMENT_NAME}"

PRETRAINED_MODEL=${MODEL_DIR}/Qwen3-14B
TRAIN_DATA=${DATA_DIR}/deepscaler.parquet
VAL_DATA=${DATA_DIR}/validation.parquet

N_NODES="${N_NODES:-4}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
TP_SIZE="${TP_SIZE:-8}"
ROLLOUT_N="${ROLLOUT_N:-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}"

ENTROPY_T0="${ENTROPY_T0:-0.3}"
ENTROPY_T1="${ENTROPY_T1:-1.2}"
WARMUP_STEPS="${WARMUP_STEPS:-20}"
ADAPTIVE_T1_MIN="${ADAPTIVE_T1_MIN:-0.6}"
ADAPTIVE_T1_MAX="${ADAPTIVE_T1_MAX:-1.8}"
ADAPTIVE_T1_EMA_BETA="${ADAPTIVE_T1_EMA_BETA:-0.9}"
ADAPTIVE_T1_STEP_SIZE="${ADAPTIVE_T1_STEP_SIZE:-0.5}"
ADAPTIVE_T1_MAX_DELTA="${ADAPTIVE_T1_MAX_DELTA:-0.05}"

LOCAL_VLLM_PATH=${LOCAL_VLLM_PATH:-}
export PYTHONPATH="${LOCAL_VLLM_PATH}:${PYTHONPATH:-}"
export RAY_ADDRESS="${RAY_ADDRESS:-auto}"
export PYTHONUNBUFFERED=1
export VERL_ENABLE_LENGTH_PENALTY=false

echo "Experiment name : ${EXPERIMENT_NAME}"
echo "Output dir      : ${OUTPUT_DIR}"
echo "Adaptive T1     : ATJS1 EMA controller"
echo "EMA beta        : ${ADAPTIVE_T1_EMA_BETA}"
echo "Step size       : ${ADAPTIVE_T1_STEP_SIZE}"
echo "Max delta       : ${ADAPTIVE_T1_MAX_DELTA}"
echo "Ray address     : ${RAY_ADDRESS}"

set -x
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.per.enable_token_entropy_weighting=true \
    algorithm.per.warmup_steps="${WARMUP_STEPS}" \
    algorithm.per.entropy_t0="${ENTROPY_T0}" \
    algorithm.per.entropy_t1="${ENTROPY_T1}" \
    algorithm.per.entropy_adv_clip=10.0 \
    algorithm.per.adaptive_t1_enable=true \
    algorithm.per.adaptive_t1_ema_beta="${ADAPTIVE_T1_EMA_BETA}" \
    algorithm.per.adaptive_t1_step_size="${ADAPTIVE_T1_STEP_SIZE}" \
    algorithm.per.adaptive_t1_max_delta="${ADAPTIVE_T1_MAX_DELTA}" \
    algorithm.per.adaptive_t1_min="${ADAPTIVE_T1_MIN}" \
    algorithm.per.adaptive_t1_max="${ADAPTIVE_T1_MAX}" \
    actor_rollout_ref.rollout.hybrid_exploration.enable=true \
    actor_rollout_ref.rollout.hybrid_exploration.use_baseline_zero_weight=true \
    "actor_rollout_ref.rollout.hybrid_exploration.temperatures=[${ENTROPY_T0},${ENTROPY_T1}]" \
    'actor_rollout_ref.rollout.hybrid_exploration.top_k_values=[-1,-1]' \
    data.train_files="${TRAIN_DATA}" \
    data.val_files="${VAL_DATA}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.truncation=left \
    +data.apply_chat_template_kwargs.enable_thinking=true \
    +actor_rollout_ref.actor.tb_type=tempered_important_sampling \
    actor_rollout_ref.model.path="${PRETRAINED_MODEL}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.val_kwargs.n=2 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${TP_SIZE}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.temperature="${ENTROPY_T1}" \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    'trainer.logger=[console,tensorboard]' \
    trainer.project_name=cceRL \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    trainer.nnodes="${N_NODES}" \
    trainer.resume_mode=auto \
    trainer.save_freq=200 \
    trainer.val_before_train=False \
    trainer.default_local_dir="${OUTPUT_DIR}" \
    trainer.test_freq=5 \
    trainer.total_epochs=2 \
    "$@"
set +x
