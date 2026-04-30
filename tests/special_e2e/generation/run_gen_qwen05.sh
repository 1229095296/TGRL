#!/usr/bin/env bash
# Tested with 1 & 4 GPUs

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data}"
MODEL_DIR="${MODEL_DIR:-${PROJECT_ROOT}/models}"
CKPT_DIR="${CKPT_DIR:-${PROJECT_ROOT}/checkpoints}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${PROJECT_ROOT}/tensorboard}"

set -xeuo pipefail

MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}

NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
OUTPUT_PATH=${OUTPUT_PATH:-$HOME${DATA_DIR}/qwen_05_gen_test.parquet}
GEN_TP=${GEN_TP:-2}  # Default tensor parallel size to 2

python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    data.path="${HOME}${DATA_DIR}/test.parquet" \
    data.prompt_key=prompt \
    data.n_samples=1 \
    data.output_path="${OUTPUT_PATH}" \
    model.path="${MODEL_ID}" \
    +model.trust_remote_code=True \
    rollout.temperature=1.0 \
    rollout.top_k=50 \
    rollout.top_p=0.7 \
    rollout.prompt_length=2048 \
    rollout.response_length=1024 \
    rollout.tensor_model_parallel_size="${GEN_TP}" \
    rollout.gpu_memory_utilization=0.8
