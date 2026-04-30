#!/bin/bash


PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data}"
MODEL_DIR="${MODEL_DIR:-${PROJECT_ROOT}/models}"
CKPT_DIR="${CKPT_DIR:-${PROJECT_ROOT}/checkpoints}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${PROJECT_ROOT}/tensorboard}"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONFIG_PATH=${CONFIG_PATH:-${PROJECT_ROOT}/experiments/mixed_temp_grpo_1p7b/configs/single_low.yaml}
METHOD_NAME=single_low

CONFIG_PATH="${CONFIG_PATH}" METHOD_NAME="${METHOD_NAME}" bash "${SCRIPT_DIR}/run_mixed_grouped.sh" "$@"
