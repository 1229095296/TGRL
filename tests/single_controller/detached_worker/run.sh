#!/bin/bash

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data}"
MODEL_DIR="${MODEL_DIR:-${PROJECT_ROOT}/models}"
CKPT_DIR="${CKPT_DIR:-${PROJECT_ROOT}/checkpoints}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${PROJECT_ROOT}/tensorboard}"

ray start --head --port=6379
python3 server.py
python3 client.py
ray stop --force