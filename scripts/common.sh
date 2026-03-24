#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/third_party/ms-swift:${PYTHONPATH:-}"
DEFAULT_IMAGE_ROOT="${DEFAULT_IMAGE_ROOT:-${PROJECT_ROOT}/data/raw}"

require_existing_path() {
  local path_value="$1"
  local path_name="$2"
  if [[ -z "${path_value}" ]]; then
    echo "Missing required path: ${path_name}" >&2
    exit 1
  fi
  if [[ ! -e "${path_value}" ]]; then
    echo "Path does not exist for ${path_name}: ${path_value}" >&2
    exit 1
  fi
}

require_nonempty_value() {
  local value="$1"
  local value_name="$2"
  if [[ -z "${value}" ]]; then
    echo "${value_name} must not be empty" >&2
    exit 1
  fi
}

resolve_dataset_paths() {
  local input_path="$1"
  local output_path="$2"
  local image_root="$3"
  mkdir -p "$(dirname "${output_path}")"
  python -m pepo.data.resolve_image_paths \
    --input "${input_path}" \
    --output "${output_path}" \
    --image_root "${image_root}"
}
