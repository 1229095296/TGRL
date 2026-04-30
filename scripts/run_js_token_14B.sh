#!/usr/bin/env bash

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data}"
MODEL_DIR="${MODEL_DIR:-${PROJECT_ROOT}/models}"
CKPT_DIR="${CKPT_DIR:-${PROJECT_ROOT}/checkpoints}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${PROJECT_ROOT}/tensorboard}"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

NUM_WORKERS="${NUM_WORKERS:-2}"
GPUS_PER_WORKER="${GPUS_PER_WORKER:-8}"
CPUS_PER_WORKER="${CPUS_PER_WORKER:-8}"
RAY_ADDRESS="${RAY_ADDRESS:-auto}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
INTERVENTION_MODEL_DEVICE="${INTERVENTION_MODEL_DEVICE:-auto}"
INTERVENTION_MODE="${INTERVENTION_MODE:-sequential}"
INTERVENTION_GRANULARITY="${INTERVENTION_GRANULARITY:-block}"
BLOCK_SIZE="${BLOCK_SIZE:-5}"
SEQUENTIAL_ROUNDS="${SEQUENTIAL_ROUNDS:-3}"
SEQUENTIAL_GAP="${SEQUENTIAL_GAP:-2}"
ROLLOUTS_PER_PROMPT="${ROLLOUTS_PER_PROMPT:-8}"
RESUME="${RESUME:-true}"
GPU_KEEPALIVE="${GPU_KEEPALIVE:-true}"
GPU_KEEPALIVE_SIZE="${GPU_KEEPALIVE_SIZE:-4096}"
GPU_KEEPALIVE_INTERVAL="${GPU_KEEPALIVE_INTERVAL:-0.0}"
GPU_KEEPALIVE_DTYPE="${GPU_KEEPALIVE_DTYPE:-float16}"
RUN_TAG="${RUN_TAG:-${INTERVENTION_MODE}_${INTERVENTION_GRANULARITY}${BLOCK_SIZE}_rounds${SEQUENTIAL_ROUNDS}_gap${SEQUENTIAL_GAP}_roll${ROLLOUTS_PER_PROMPT}}"

MODEL_PATH="${MODEL_DIR}/Qwen3-14B"
DATA_PATH="${DATA_DIR}/validation.parquet"
ROOT_DIR="${PROJECT_ROOT}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/analysis/js_token_intervention/14b_${RUN_TAG}}"

mkdir -p "${OUTPUT_DIR}"

EXTRA_ARGS=()
if [[ "${RESUME}" == "1" || "${RESUME}" == "true" || "${RESUME}" == "TRUE" ]]; then
    EXTRA_ARGS+=(--resume)
fi
if [[ "${GPU_KEEPALIVE}" == "1" || "${GPU_KEEPALIVE}" == "true" || "${GPU_KEEPALIVE}" == "TRUE" ]]; then
    EXTRA_ARGS+=(
        --gpu-keepalive
        --gpu-keepalive-size "${GPU_KEEPALIVE_SIZE}"
        --gpu-keepalive-interval "${GPU_KEEPALIVE_INTERVAL}"
        --gpu-keepalive-dtype "${GPU_KEEPALIVE_DTYPE}"
    )
fi

python3 "${SCRIPT_DIR}/js_token_intervention_ray.py" \
    --ray-address "${RAY_ADDRESS}" \
    --num-workers "${NUM_WORKERS}" \
    --gpus-per-worker "${GPUS_PER_WORKER}" \
    --cpus-per-worker "${CPUS_PER_WORKER}" \
    --script-path "${SCRIPT_DIR}/js_token_intervention_vllm.py" \
    --model-path "${MODEL_PATH}" \
    --data-path "${DATA_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --max-examples 1024 \
    --target-successes 320 \
    --max-new-tokens 8192 \
    --t0 0.3 \
    --t1 1.2 \
    --top-p 1.0 \
    --top-k -1 \
    --rollouts-per-prompt "${ROLLOUTS_PER_PROMPT}" \
    --analysis-batch-size 32 \
    --positions-per-sample 1 \
    --intervention-mode "${INTERVENTION_MODE}" \
    --intervention-granularity "${INTERVENTION_GRANULARITY}" \
    --block-size "${BLOCK_SIZE}" \
    --sequential-rounds "${SEQUENTIAL_ROUNDS}" \
    --sequential-gap "${SEQUENTIAL_GAP}" \
    --matched-random-repeats 5 \
    --selectors "js,matched_random,entropy,low_margin" \
    --replacement-strategy "t0_top1_else_t0_top2" \
    --enable-thinking \
    --dtype bfloat16 \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --vllm-gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
    --vllm-max-model-len "${VLLM_MAX_MODEL_LEN}" \
    --intervention-model-device "${INTERVENTION_MODEL_DEVICE}" \
    --trust-remote-code \
    --seed 42 \
    "${EXTRA_ARGS[@]}" \
    "$@"
