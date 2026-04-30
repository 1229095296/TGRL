#!/usr/bin/env bash

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data}"
MODEL_DIR="${MODEL_DIR:-${PROJECT_ROOT}/models}"
CKPT_DIR="${CKPT_DIR:-${PROJECT_ROOT}/checkpoints}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${PROJECT_ROOT}/tensorboard}"

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CKPT_ROOT="${CKPT_DIR}"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/wallclock_14b"

resolve_latest_run_dir() {
    local pattern="$1"
    local latest=""
    latest=$(ls -td "${CKPT_ROOT}"/${pattern} 2>/dev/null | head -n1 || true)
    echo "${latest}"
}

ATJS_RUN_DIR="${ATJS_RUN_DIR:-$(resolve_latest_run_dir 'cpl-ATJS1ctrl-r4-14B*')}"
GRPO_RUN_DIR="${GRPO_RUN_DIR:-$(resolve_latest_run_dir 'grpo-r4-14B-*')}"
DAPO_RUN_DIR="${DAPO_RUN_DIR:-$(resolve_latest_run_dir 'dapo-T1.2r4-14B-*')}"

METRIC_KEY="${METRIC_KEY:-val-core/aime_combined_score}"
BUDGET_HOURS="${BUDGET_HOURS:-12}"
TARGET_REFERENCE="${TARGET_REFERENCE:-DAPO}"
TARGET_FRACTION="${TARGET_FRACTION:-0.95}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/$(date +%Y%m%d_%H%M%S)}"

if [[ -z "${ATJS_RUN_DIR}" || -z "${GRPO_RUN_DIR}" || -z "${DAPO_RUN_DIR}" ]]; then
    cat <<EOF
无法自动解析 14B wall-clock 对比所需的 run 目录。
请显式指定：
  ATJS_RUN_DIR=/path/to/atjs_run \
  GRPO_RUN_DIR=/path/to/grpo_run \
  DAPO_RUN_DIR=/path/to/dapo_run \
  OUTPUT_DIR=/path/to/output \
  bash scripts/run_14b_wallclock_efficiency.sh

默认查找目录：
  ATJS: ${CKPT_ROOT}/cpl-ATJS1ctrl-r4-14B*
  GRPO: ${CKPT_ROOT}/grpo-r4-14B-*
  DAPO: ${CKPT_ROOT}/dapo-T1.2r4-14B-*

每个 run 目录下都必须存在 wallclock_study/events.jsonl。
EOF
    exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

echo "ATJS_RUN_DIR=${ATJS_RUN_DIR}"
echo "GRPO_RUN_DIR=${GRPO_RUN_DIR}"
echo "DAPO_RUN_DIR=${DAPO_RUN_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

python3 "${ROOT_DIR}/scripts/analyze_wallclock_efficiency.py" \
    --run "ATJS1ctrl=${ATJS_RUN_DIR}" \
    --run "GRPO=${GRPO_RUN_DIR}" \
    --run "DAPO=${DAPO_RUN_DIR}" \
    --metric-key "${METRIC_KEY}" \
    --budget-hours "${BUDGET_HOURS}" \
    --target-reference "${TARGET_REFERENCE}" \
    --target-fraction "${TARGET_FRACTION}" \
    --output-dir "${OUTPUT_DIR}"

cat <<EOF

已生成 wall-clock 论文指标文件：
  ${OUTPUT_DIR}/runtime_summary.csv
  ${OUTPUT_DIR}/fixed_budget_scores.csv
  ${OUTPUT_DIR}/time_to_target.csv
  ${OUTPUT_DIR}/validation_over_time.csv
  ${OUTPUT_DIR}/summary.json
EOF
