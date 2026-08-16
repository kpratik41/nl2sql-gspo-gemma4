#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ec2-user/consensus/nl2sql-gspo-gemma4"
CKPT="outputs/sft_rft_sftckpt70-2/gemma4_31b_sftckpt70_dapo12_lr2e6/checkpoint-70"
RUN_TAG="temp0_olddev_schema_tool_unpatched_vllm_async_tp2_ctx43k"
LOG="${CKPT}/${RUN_TAG}.log"

cd "$ROOT"

INFERENCE_CUDA_VISIBLE_DEVICES="0,1" \
PYTHON_BIN="/home/ec2-user/miniconda3/envs/nl2sql312/bin/python" \
PYTHONUNBUFFERED=1 \
INFERENCE_BACKEND="vllm_async" \
MODEL_PATH="${CKPT}" \
INPUT_FILE="outputs/old-dev-schema-tool-unpatched.jsonl" \
OUTPUT_DIR="${CKPT}/${RUN_TAG}" \
APPEND_OUTPUT_TIMESTAMP=0 \
NUM_EXAMPLES=-1 \
MAX_PROMPT_LENGTH=34000 \
MAX_NEW_TOKENS=8000 \
MAX_TOOL_ROUNDS=8 \
TEMPERATURE=0.0 \
TOP_P=1.0 \
VLLM_TENSOR_PARALLEL_SIZE=2 \
VLLM_DATA_PARALLEL_SIZE=1 \
VLLM_ASYNC_CONCURRENCY=16 \
VLLM_GPU_MEMORY_UTILIZATION=0.93 \
VLLM_MAX_MODEL_LEN=43000 \
bash scripts/launch_inference.sh > "$LOG" 2>&1

echo "[run] ckpt-70 finished rc=$?"
