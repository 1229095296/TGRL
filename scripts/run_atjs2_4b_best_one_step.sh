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
CKPT_ROOT="${ROOT_DIR}/ckpt"
RUNLOG_ROOT="${ROOT_DIR}/run_log"

SOURCE_EXPERIMENT="cpl-ATJS2-r4-8B"
BEST_DIR="${CKPT_ROOT}/${SOURCE_EXPERIMENT}/best_aime_ckpt"
BEST_INFO="${BEST_DIR}/best_checkpoint_info.txt"
SOURCE_TRAIN_LOG="${RUNLOG_ROOT}/${SOURCE_EXPERIMENT}/train.log"

PRETRAINED_MODEL=${MODEL_DIR}/Qwen3-4B
TRAIN_DATA=${DATA_DIR}/deepscaler.parquet
VAL_DATA=${DATA_DIR}/validation.parquet

N_NODES=1
N_GPUS_PER_NODE=8
TP_SIZE=8
ROLLOUT_N=4
TRAIN_BATCH_SIZE=8
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=8192

if [[ ! -f "${BEST_INFO}" ]]; then
  echo "Missing best checkpoint metadata: ${BEST_INFO}" >&2
  exit 1
fi

if [[ ! -d "${BEST_DIR}/actor" ]]; then
  echo "Missing actor checkpoint: ${BEST_DIR}/actor" >&2
  exit 1
fi

BEST_STEP="$(awk -F': ' '/^Global Step:/{print $2}' "${BEST_INFO}" | tail -n 1)"
if [[ -z "${BEST_STEP}" ]]; then
  echo "Failed to parse Global Step from ${BEST_INFO}" >&2
  exit 1
fi

BEST_T1="$(awk -v step="${BEST_STEP}" '
  $0 ~ ("\\[Adaptive T1\\] Step " step ":") && /next_T1=/ {
    if (match($0, /next_T1=[0-9.]+/)) {
      val = substr($0, RSTART + 8, RLENGTH - 8)
    }
  }
  END {
    if (val != "") {
      print val
    }
  }
' "${SOURCE_TRAIN_LOG}")"
BEST_T1="${BEST_T1:-1.2}"

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
EXPERIMENT_NAME="${SOURCE_EXPERIMENT}-best1step-jsviz-${RUN_TAG}"
OUTPUT_DIR="${CKPT_ROOT}/${EXPERIMENT_NAME}"
LOG_DIR="${RUNLOG_ROOT}/${EXPERIMENT_NAME}"
LOG_FILE="${LOG_DIR}/train.log"
ROLLOUT_DUMP_DIR="${OUTPUT_DIR}/rollout_data"
JSVIZ_DIR="${OUTPUT_DIR}/js_token_visualization"
RESUME_WRAP_DIR="${OUTPUT_DIR}/resume_stub/global_step_${BEST_STEP}"

mkdir -p "${LOG_DIR}" "${ROLLOUT_DUMP_DIR}" "${JSVIZ_DIR}" "${RESUME_WRAP_DIR}"
ln -sfn "${BEST_DIR}/actor" "${RESUME_WRAP_DIR}/actor"
if [[ -d "${BEST_DIR}/critic" ]]; then
  ln -sfn "${BEST_DIR}/critic" "${RESUME_WRAP_DIR}/critic"
  HAS_CRITIC=true
else
  HAS_CRITIC=false
fi
cp "${BEST_INFO}" "${RESUME_WRAP_DIR}/best_checkpoint_info.txt"

LOCAL_VLLM_PATH=${LOCAL_VLLM_PATH:-}
export PYTHONPATH="${LOCAL_VLLM_PATH}:${PYTHONPATH:-}"
export VLLM_SOFT_THINK_DEBUG=1
export PYTHONUNBUFFERED=1
export VERL_ENABLE_LENGTH_PENALTY=false
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH}"

echo "Source experiment : ${SOURCE_EXPERIMENT}"
echo "Best checkpoint   : ${BEST_DIR}"
echo "Best step         : ${BEST_STEP}"
echo "Recovered T1      : ${BEST_T1}"
echo "Has critic ckpt   : ${HAS_CRITIC}"
echo "Adaptive T1       : disabled for this one-step probe"
echo "New experiment    : ${EXPERIMENT_NAME}"
echo "Resume wrapper    : ${RESUME_WRAP_DIR}"
echo "Rollout dump dir  : ${ROLLOUT_DUMP_DIR}"
echo "JS viz dir        : ${JSVIZ_DIR}"

exec > >(tee -a "${LOG_FILE}") 2>&1

set -x
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.per.enable_token_entropy_weighting=true \
    algorithm.per.warmup_steps=0 \
    algorithm.per.entropy_t0=0.3 \
    algorithm.per.entropy_t1="${BEST_T1}" \
    algorithm.per.js_token_visualization_enable=true \
    algorithm.per.js_token_visualization_freq=1 \
    algorithm.per.js_token_visualization_max_groups=4 \
    algorithm.per.js_token_visualization_max_tokens_per_sample=8192 \
    algorithm.per.js_token_visualization_output_dir="${JSVIZ_DIR}" \
    actor_rollout_ref.rollout.hybrid_exploration.enable=true \
    actor_rollout_ref.rollout.hybrid_exploration.use_baseline_zero_weight=true \
    "actor_rollout_ref.rollout.hybrid_exploration.temperatures=[0.3,${BEST_T1}]" \
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
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${TP_SIZE}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.temperature="${BEST_T1}" \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    'trainer.logger=[console,tensorboard]' \
    trainer.project_name=cceRL \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    trainer.nnodes="${N_NODES}" \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path="${RESUME_WRAP_DIR}" \
    trainer.save_freq=-1 \
    trainer.val_before_train=False \
    trainer.default_local_dir="${OUTPUT_DIR}" \
    trainer.rollout_data_dir="${ROLLOUT_DUMP_DIR}" \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps="$((BEST_STEP + 1))" \
    "$@"
set +x

STEP_JSONL="${ROLLOUT_DUMP_DIR}/$((BEST_STEP + 1)).jsonl"
if [[ -f "${STEP_JSONL}" ]]; then
  echo
  echo "First rollout samples from ${STEP_JSONL}:"
  python3 - "${STEP_JSONL}" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 6:
            break
        obj = json.loads(line)
        input_text = obj.get("input", "").replace("\n", " ")
        output_text = obj.get("output", "").replace("\n", " ")
        print(f"[sample {i}] score={obj.get('score')}")
        print(f"  input : {input_text[:180]}")
        print(f"  output: {output_text[:280]}")
PY
else
  echo "Rollout dump not found: ${STEP_JSONL}" >&2
fi

echo
echo "Done."
echo "Log file        : ${LOG_FILE}"
echo "Rollout dump dir: ${ROLLOUT_DUMP_DIR}"
echo "JS viz dir      : ${JSVIZ_DIR}"
