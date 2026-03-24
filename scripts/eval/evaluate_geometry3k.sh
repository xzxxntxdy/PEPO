#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"

GEOMETRY_IMAGE_ROOT="${GEOMETRY_IMAGE_ROOT:-${DEFAULT_IMAGE_ROOT}}"
DATA_FILE="${DATA_FILE:-${PROJECT_ROOT}/data/geometry3k/test.jsonl}"
MODEL_PATH="${MODEL_PATH:-}"
PROCESSOR_PATH="${PROCESSOR_PATH:-${MODEL_PATH}}"
OUTPUT_PATH="${OUTPUT_PATH:-${PROJECT_ROOT}/results/geometry3k/results_test.json}"
BON_N="${BON_N:-8}"
BON_TEMPERATURE="${BON_TEMPERATURE:-1.0}"
BON_TOP_P="${BON_TOP_P:-1.0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
GROUP_SIZE="${GROUP_SIZE:-0}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MODEL_TYPE="${MODEL_TYPE:-auto}"
LIMIT="${LIMIT:--1}"

RESOLVED_DIR="${PROJECT_ROOT}/.cache/datasets/geometry3k"
RESOLVED_DATA="${RESOLVED_DIR}/test.abs.jsonl"

require_existing_path "${MODEL_PATH}" "MODEL_PATH"
require_existing_path "${PROCESSOR_PATH}" "PROCESSOR_PATH"
resolve_dataset_paths "${DATA_FILE}" "${RESOLVED_DATA}" "${GEOMETRY_IMAGE_ROOT}"
mkdir -p "$(dirname "${OUTPUT_PATH}")"

python -m pepo.evaluation.evaluate_geometry3k \
  --data "${RESOLVED_DATA}" \
  --model_path "${MODEL_PATH}" \
  --processor_path "${PROCESSOR_PATH}" \
  --output "${OUTPUT_PATH}" \
  --bon_n "${BON_N}" \
  --bon_temperature "${BON_TEMPERATURE}" \
  --bon_top_p "${BON_TOP_P}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --group_size "${GROUP_SIZE}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --model-type "${MODEL_TYPE}" \
  --limit "${LIMIT}"
