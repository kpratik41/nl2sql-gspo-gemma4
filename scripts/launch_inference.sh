#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${INFERENCE_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

INFERENCE_BACKEND="${INFERENCE_BACKEND:-transformers}"
MODEL_PATH="${MODEL_PATH:-outputs/gemma4_31b_gspo_bird}"
INPUT_FILE="${INPUT_FILE:-outputs/dev-20251106-schema.jsonl}"
DATABASE_DIR="${DATABASE_DIR:-databases/dev_databases}"
DIFF_JSON_PATH="${DIFF_JSON_PATH:-data/bird_dev_data/raw/dev_20251106.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/bird_dev_inference}"
NUM_EXAMPLES="${NUM_EXAMPLES:--1}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-30000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-4}"
VLLM_DATA_PARALLEL_SIZE="${VLLM_DATA_PARALLEL_SIZE:-2}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-}"
APPEND_OUTPUT_TIMESTAMP="${APPEND_OUTPUT_TIMESTAMP:-1}"
OUTPUT_TIMESTAMP="${OUTPUT_TIMESTAMP:-}"

if [[ "${APPEND_OUTPUT_TIMESTAMP}" == "1" ]]; then
  if [[ -z "${OUTPUT_TIMESTAMP}" ]]; then
    OUTPUT_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
  fi

  if [[ ! "${OUTPUT_DIR}" =~ _[0-9]{8}_[0-9]{6}$ ]]; then
    OUTPUT_DIR="${OUTPUT_DIR}_${OUTPUT_TIMESTAMP}"
  fi
fi

echo "[launcher] output_dir=${OUTPUT_DIR}"

cmd=(
  "${PYTHON_BIN}"
  scripts/run_inference_bird.py
  --inference_backend "${INFERENCE_BACKEND}"
  --model_name_or_path "${MODEL_PATH}"
  --input_file "${INPUT_FILE}"
  --database_dir "${DATABASE_DIR}"
  --diff_json_path "${DIFF_JSON_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --num_examples "${NUM_EXAMPLES}"
  --max_prompt_length "${MAX_PROMPT_LENGTH}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --overwrite
)

if [[ -n "${VLLM_TENSOR_PARALLEL_SIZE}" ]]; then
  cmd+=(--vllm_tensor_parallel_size "${VLLM_TENSOR_PARALLEL_SIZE}")
fi

if [[ -n "${VLLM_DATA_PARALLEL_SIZE}" ]]; then
  cmd+=(--vllm_data_parallel_size "${VLLM_DATA_PARALLEL_SIZE}")
fi

if [[ -n "${VLLM_MAX_MODEL_LEN}" ]]; then
  cmd+=(--vllm_max_model_len "${VLLM_MAX_MODEL_LEN}")
fi

if [[ "${INFERENCE_BACKEND}" == "vllm" ]]; then
  cmd+=(--vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}")
fi

"${cmd[@]}"