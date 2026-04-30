#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data}"
MODEL_DIR="${MODEL_DIR:-${PROJECT_ROOT}/models}"
CKPT_DIR="${CKPT_DIR:-${PROJECT_ROOT}/checkpoints}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"

MODEL_PATH="${MODEL_PATH:-${MODEL_DIR}/Qwen3-4B}"
VAL_FILE="${VAL_FILE:-${DATA_DIR}/validation.parquet}"
EVAL_CKPT="${EVAL_CKPT:-${CKPT_DIR}/anonymous-method-4b}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-anonymous-eval}"

mkdir -p "${LOG_DIR}/${EXPERIMENT_NAME}"

python3 -m verl.trainer.main_ppo \
  data.train_files="${VAL_FILE}" \
  data.val_files="${VAL_FILE}" \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  trainer.project_name="anonymous_review" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.resume_mode=auto \
  trainer.default_local_dir="${EVAL_CKPT}" \
  trainer.logger='["console"]' \
  trainer.total_epochs=0 \
  "$@"
