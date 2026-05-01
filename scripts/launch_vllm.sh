#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=6,7
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

MODEL_NAME="google/gemma-4-31B"

trl vllm-serve \
  --model "${MODEL_NAME}" \
  --host 0.0.0.0 \
  --port 8000 \
  --vllm_tensor_parallel_size 2 \
  --vllm_gpu_memory_utilization 0.90 \
  --vllm_max_model_length 16384 \
  --dtype bfloat16