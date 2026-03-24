#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"

MODEL_PATH="${MODEL_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/results/logicvista}"
CACHE_DIR="${CACHE_DIR:-${PROJECT_ROOT}/.cache/hf/logicvista}"
PARALLEL_MODE="${PARALLEL_MODE:-dp}"
BON_N="${BON_N:-8}"
TOP_P="${TOP_P:-1.0}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MODEL_TYPE="${MODEL_TYPE:-auto}"
MAX_NUM_PROBLEMS="${MAX_NUM_PROBLEMS:--1}"
KEEP_MULTI_CHAR_ANSWERS="${KEEP_MULTI_CHAR_ANSWERS:-false}"

require_existing_path "${MODEL_PATH}" "MODEL_PATH"
mkdir -p "${OUTPUT_DIR}" "${CACHE_DIR}"

extra_args=()
if [[ "${KEEP_MULTI_CHAR_ANSWERS}" == "true" ]]; then
  extra_args+=(--keep-multi-char-answers)
fi

cmd=(
  python -m pepo.evaluation.evaluate_logicvista
  --checkpoint "${MODEL_PATH}"
  --cache_dir "${CACHE_DIR}"
  --out_dir "${OUTPUT_DIR}"
  --parallel-mode "${PARALLEL_MODE}"
  --bon_n "${BON_N}"
  --top_p "${TOP_P}"
  --max_tokens "${MAX_TOKENS}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max_num_problems "${MAX_NUM_PROBLEMS}"
  --model_type "${MODEL_TYPE}"
)
if ((${#extra_args[@]})); then
  cmd+=("${extra_args[@]}")
fi
"${cmd[@]}"
