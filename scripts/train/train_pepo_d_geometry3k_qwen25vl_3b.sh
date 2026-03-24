#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-VL-3B-Instruct}"
GEOMETRY_IMAGE_ROOT="${GEOMETRY_IMAGE_ROOT:-${DEFAULT_IMAGE_ROOT}}"
MASTER_PORT="${MASTER_PORT:-12349}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/pepo_d/geometry3k/qwen2_5_vl_3b}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-pepo_d_geometry3k_qwen2_5_vl_3b}"

TRAIN_DATA="${PROJECT_ROOT}/data/geometry3k/train.jsonl"
VAL_DATA="${PROJECT_ROOT}/data/geometry3k/val.jsonl"
RESOLVED_DIR="${PROJECT_ROOT}/.cache/datasets/geometry3k"
RESOLVED_TRAIN="${RESOLVED_DIR}/train.abs.jsonl"
RESOLVED_VAL="${RESOLVED_DIR}/val.abs.jsonl"

require_nonempty_value "${MODEL_ID}" "MODEL_ID"
resolve_dataset_paths "${TRAIN_DATA}" "${RESOLVED_TRAIN}" "${GEOMETRY_IMAGE_ROOT}"
resolve_dataset_paths "${VAL_DATA}" "${RESOLVED_VAL}" "${GEOMETRY_IMAGE_ROOT}"

extra_args=()
if [[ -n "${SWANLAB_TOKEN:-}" ]]; then
  extra_args+=(--report_to swanlab --swanlab_token "${SWANLAB_TOKEN}" --swanlab_exp_name "${EXPERIMENT_NAME}")
fi

if [[ -n "${HF_ENDPOINT:-}" ]]; then
  export HF_ENDPOINT
fi

export MASTER_PORT
cmd=(
  python -m pepo.train.rlhf
  --rlhf_type grpo
  --model "${MODEL_ID}"
  --use_hf true
  --external_plugins "${PROJECT_ROOT}/src/pepo/rewards/plugin.py"
  --reward_funcs external_format external_qa_acc
  --use_vllm true
  --vllm_mode colocate
  --sleep_level 0
  --vllm_gpu_memory_utilization 0.6
  --vllm_max_model_len 2048
  --vllm_enable_prefix_caching true
  --train_type full
  --gradient_checkpointing true
  --torch_dtype bfloat16
  --dataset "${RESOLVED_TRAIN}"
  --val_dataset "${RESOLVED_VAL}"
  --per_device_eval_batch_size 8
  --max_completion_length 1024
  --per_device_train_batch_size 2
  --learning_rate 1e-6
  --gradient_accumulation_steps 4
  --num_train_epochs 1
  --save_strategy epoch
  --eval_strategy steps
  --eval_steps 25
  --save_total_limit 5
  --logging_steps 1
  --dataloader_num_workers 4
  --num_generations 8
  --temperature 1.0
  --top_p 1.0
  --deepspeed zero2
  --beta 0.001
  --log_completions true
  --log_entropy true
  --num_iterations 1
  --async_generate false
  --attn_impl flash_attn
  --loss_type bnpo
  --dynamic_sample true
  --overlong_filter true
  --img_token '<|image_pad|>'
  --gate_alpha 0.10
  --gate_temperature 1.0
  --use_vision_weights true
  --output_dir "${OUTPUT_DIR}"
)
if ((${#extra_args[@]})); then
  cmd+=("${extra_args[@]}")
fi
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
"${cmd[@]}"
