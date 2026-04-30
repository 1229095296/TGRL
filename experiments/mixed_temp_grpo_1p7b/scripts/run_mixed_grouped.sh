#!/bin/bash


PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data}"
MODEL_DIR="${MODEL_DIR:-${PROJECT_ROOT}/models}"
CKPT_DIR="${CKPT_DIR:-${PROJECT_ROOT}/checkpoints}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${PROJECT_ROOT}/tensorboard}"

set -euo pipefail

REPO_ROOT=${PROJECT_ROOT}
cd "${REPO_ROOT}"

MODEL_PATH=${MODEL_PATH:-${MODEL_DIR}/Qwen3-1.7B}
RAW_GSM8K_PATH=${RAW_GSM8K_PATH:-${DATA_DIR}/gsm8k.parquet}
TRAIN_INPUT_PATH=${TRAIN_INPUT_PATH:-${DATA_DIR}/gsm8k.parquet}
VAL_INPUT_PATH=${VAL_INPUT_PATH:-${DATA_DIR}/gsm8k_test.parquet}

N_NODES=${N_NODES:-1}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-4}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}
LOGPROB_MICRO_BATCH_SIZE_PER_GPU=${LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-4}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-1024}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}
SAVE_FREQ=${SAVE_FREQ:-200}
TEST_FREQ=${TEST_FREQ:-5}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-2}
SEED=${SEED:-1}
TRAIN_LIMIT=${TRAIN_LIMIT:-}
VAL_LIMIT=${VAL_LIMIT:-}
ROLLOUT_N=${ROLLOUT_N:-8}
VAL_N=${VAL_N:-8}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.85}
ENABLE_THINKING_MODE=${ENABLE_THINKING_MODE:-true}
PROBE_BACKEND=${PROBE_BACKEND:-}
PROBE_MODEL_PATH=${PROBE_MODEL_PATH:-${MODEL_PATH}}
PROBE_TOKENIZER_PATH=${PROBE_TOKENIZER_PATH:-}
PROBE_DEVICE=${PROBE_DEVICE:-auto}
PROBE_N=${PROBE_N:-8}
PROBE_BATCH_SIZE=${PROBE_BATCH_SIZE:-4}
PROBE_MAX_PROMPTS=${PROBE_MAX_PROMPTS:-}
PROBE_MAX_NEW_TOKENS=${PROBE_MAX_NEW_TOKENS:-256}
PROBE_SEED=${PROBE_SEED:-${SEED}}
PROBE_LOW_TEMPERATURE=${PROBE_LOW_TEMPERATURE:-0.3}
PROBE_HIGH_TEMPERATURE=${PROBE_HIGH_TEMPERATURE:-1.2}
PROBE_LOW_TEMPERATURES=${PROBE_LOW_TEMPERATURES:-}
PROBE_HIGH_TEMPERATURES=${PROBE_HIGH_TEMPERATURES:-}
AUTO_SELECT_PAIR=${AUTO_SELECT_PAIR:-false}

METHOD_NAME=${METHOD_NAME:-mixed_grouped}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-mixed-temp-grpo-1p7b-${METHOD_NAME}-$(date +%Y%m%d_%H%M)}
CKPT_ROOT=${CKPT_DIR}
RUN_LOG_ROOT=${LOG_DIR}
OUTPUT_ROOT=${OUTPUT_ROOT:-${REPO_ROOT}/experiments/mixed_temp_grpo_1p7b/outputs}
DATA_DIR=${DATA_DIR:-${OUTPUT_ROOT}/data}
ANALYSIS_DIR=${ANALYSIS_DIR:-${OUTPUT_ROOT}/analysis/${EXPERIMENT_NAME}}
CONFIG_PATH=${CONFIG_PATH:-${REPO_ROOT}/experiments/mixed_temp_grpo_1p7b/configs/mixed_grouped.yaml}
PREPARE_DATA=${PREPARE_DATA:-true}

mkdir -p "${DATA_DIR}" "${ANALYSIS_DIR}" "${RUN_LOG_ROOT}/${EXPERIMENT_NAME}"
LOG_FILE="${RUN_LOG_ROOT}/${EXPERIMENT_NAME}/train.log"

LOCAL_VLLM_PATH=${LOCAL_VLLM_PATH:-${REPO_ROOT}/vllm-0.8.5}
export PYTHONPATH="${LOCAL_VLLM_PATH}:${PYTHONPATH:-}"
echo "使用本地 vLLM: ${LOCAL_VLLM_PATH}"
export VLLM_SOFT_THINK_DEBUG=1
export PYTHONUNBUFFERED=1
export VERL_ENABLE_LENGTH_PENALTY=false
export MAX_RESPONSE_LENGTH

exec > >(tee -a "${LOG_FILE}") 2>&1

set -x

if (( ROLLOUT_MAX_NUM_BATCHED_TOKENS < ROLLOUT_MAX_NUM_SEQS )); then
    ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_SEQS}
fi

if [[ "${PREPARE_DATA}" == "true" ]]; then
    prepare_args=(--output-dir "${DATA_DIR}")
    if [[ -n "${TRAIN_INPUT_PATH}" && -n "${VAL_INPUT_PATH}" ]]; then
        prepare_args+=(--train-input "${TRAIN_INPUT_PATH}" --val-input "${VAL_INPUT_PATH}")
    else
        prepare_args+=(--input "${RAW_GSM8K_PATH}")
    fi
    if [[ -n "${TRAIN_LIMIT}" ]]; then
        prepare_args+=(--train-limit "${TRAIN_LIMIT}")
    fi
    if [[ -n "${VAL_LIMIT}" ]]; then
        prepare_args+=(--val-limit "${VAL_LIMIT}")
    fi
    if [[ -n "${PROBE_BACKEND}" ]]; then
        prepare_args+=(
            --probe-backend "${PROBE_BACKEND}"
            --probe-model-path "${PROBE_MODEL_PATH}"
            --probe-device "${PROBE_DEVICE}"
            --probe-n "${PROBE_N}"
            --probe-batch-size "${PROBE_BATCH_SIZE}"
            --probe-max-new-tokens "${PROBE_MAX_NEW_TOKENS}"
            --probe-seed "${PROBE_SEED}"
            --probe-low-temperature "${PROBE_LOW_TEMPERATURE}"
            --probe-high-temperature "${PROBE_HIGH_TEMPERATURE}"
        )
        if [[ -n "${PROBE_TOKENIZER_PATH}" ]]; then
            prepare_args+=(--probe-tokenizer-path "${PROBE_TOKENIZER_PATH}")
        fi
        if [[ -n "${PROBE_MAX_PROMPTS}" ]]; then
            prepare_args+=(--probe-max-prompts "${PROBE_MAX_PROMPTS}")
        fi
        if [[ -n "${PROBE_LOW_TEMPERATURES}" ]]; then
            prepare_args+=(--probe-low-temperatures ${PROBE_LOW_TEMPERATURES})
        fi
        if [[ -n "${PROBE_HIGH_TEMPERATURES}" ]]; then
            prepare_args+=(--probe-high-temperatures ${PROBE_HIGH_TEMPERATURES})
        fi
        if [[ "${AUTO_SELECT_PAIR}" == "true" ]]; then
            prepare_args+=(--auto-select-pair)
        fi
    fi

    python3 experiments/mixed_temp_grpo_1p7b/scripts/prepare_gsm8k_medium.py "${prepare_args[@]}"
fi

python3 -m experiments.mixed_temp_grpo_1p7b.trainer.main_mixed_grpo \
    --experiment-config "${CONFIG_PATH}" \
    +data.apply_chat_template_kwargs.enable_thinking="${ENABLE_THINKING_MODE}" \
    +actor_rollout_ref.actor.tb_type=tempered_important_sampling \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${TENSOR_MODEL_PARALLEL_SIZE}" \
    data.train_files="${DATA_DIR}/train_medium.parquet" \
    data.val_files="${DATA_DIR}/val_full.parquet" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.seed="${SEED}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.truncation=left \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=true \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOGPROB_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOGPROB_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))" \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.val_kwargs.n="${VAL_N}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.ref.fsdp_config.param_offload=false \
    algorithm.use_kl_in_reward=false \
    trainer.critic_warmup=0 \
    trainer.project_name=cceRL \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    trainer.nnodes="${N_NODES}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.resume_mode=auto \
    trainer.val_before_train=false \
    trainer.default_local_dir="${CKPT_ROOT}/${EXPERIMENT_NAME}" \
    experiment.analysis_dir="${ANALYSIS_DIR}" \
    "$@"
