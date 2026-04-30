#!/bin/bash

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data}"
MODEL_DIR="${MODEL_DIR:-${PROJECT_ROOT}/models}"
CKPT_DIR="${CKPT_DIR:-${PROJECT_ROOT}/checkpoints}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${PROJECT_ROOT}/tensorboard}"

set -xeuo pipefail

# Get the configuration name and engine name from arguments
CONFIG_NAME="$1"
ENGINE="${2:-vllm}"

# Download model if needed
huggingface-cli download Qwen/Qwen2.5-0.5B --local-dir "$HOME/models/Qwen/Qwen2.5-0.5B"

# Run the training with the specified configuration
python3 -m verl.trainer.main_ppo \
    --config-name "$CONFIG_NAME" "$@" 