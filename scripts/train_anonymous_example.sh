#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data}"
MODEL_DIR="${MODEL_DIR:-${PROJECT_ROOT}/models}"
CKPT_DIR="${CKPT_DIR:-${PROJECT_ROOT}/checkpoints}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${PROJECT_ROOT}/tensorboard}"

MODEL_PATH="${MODEL_PATH:-${MODEL_DIR}/Qwen3-4B}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/deepscaler.parquet}"
VAL_FILE="${VAL_FILE:-${DATA_DIR}/validation.parquet}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-anonymous-method-4b}"

mkdir -p "${CKPT_DIR}/${EXPERIMENT_NAME}" "${LOG_DIR}/${EXPERIMENT_NAME}" "${TENSORBOARD_DIR}"

python3 -m verl.trainer.main_ppo \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  trainer.project_name="anonymous_review" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.default_local_dir="${CKPT_DIR}/${EXPERIMENT_NAME}" \
  trainer.logger='["console","tensorboard"]' \
  "$@"
