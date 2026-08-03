#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${INFERENCE_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

INFERENCE_BACKEND="${INFERENCE_BACKEND:-vllm}"
MODEL_PATH="${MODEL_PATH:-outputs/gemma4_31b_gspo_bird}"
INPUT_FILE="${INPUT_FILE:-outputs/old-dev-schema-tool-unpatched.jsonl}"
DATABASE_DIR="${DATABASE_DIR:-databases/dev_databases}"
DIFF_JSON_PATH="${DIFF_JSON_PATH:-data/bird_dev_data/raw/dev_20251106.json}"
USER_OUTPUT_DIR="${OUTPUT_DIR:-}"
NUM_EXAMPLES="${NUM_EXAMPLES:--1}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-34000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8000}"
MAX_TOOL_ROUNDS="${MAX_TOOL_ROUNDS:-8}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-60}"
EVAL_WORKERS="${EVAL_WORKERS:-16}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.93}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-43000}"
VLLM_ASYNC_CONCURRENCY="${VLLM_ASYNC_CONCURRENCY:-16}"
APPEND_OUTPUT_TIMESTAMP="${APPEND_OUTPUT_TIMESTAMP:-1}"
OUTPUT_TIMESTAMP="${OUTPUT_TIMESTAMP:-}"

case "${INFERENCE_BACKEND}" in
  vllm)
    VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-2}"
    VLLM_DATA_PARALLEL_SIZE="${VLLM_DATA_PARALLEL_SIZE:-4}"
    ;;
  vllm_async)
    VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-8}"
    VLLM_DATA_PARALLEL_SIZE="${VLLM_DATA_PARALLEL_SIZE:-1}"
    ;;
  *)
    echo "[launcher] unsupported INFERENCE_BACKEND=${INFERENCE_BACKEND}; use vllm or vllm_async" >&2
    exit 2
    ;;
esac

sanitize_path_part() {
  local value="$1"
  value="${value##*/}"
  value="${value%.*}"
  value="$(printf '%s' "${value}" | tr -cs '[:alnum:]_.-' '_')"
  value="${value##_}"
  value="${value%%_}"
  if [[ -z "${value}" ]]; then
    value="unknown"
  fi
  printf '%s' "${value}"
}

infer_split_name() {
  local input_name
  input_name="$(basename "${INPUT_FILE}")"

  case "${input_name}" in
    train*) printf 'train' ;;
    dev*) printf 'dev' ;;
    test*) printf 'test' ;;
    *) printf 'unknown' ;;
  esac
}

build_default_output_dir() {
  local split_name input_tag model_tag run_tag prompt_k output_k context_k

  split_name="$(infer_split_name)"
  input_tag="$(sanitize_path_part "${INPUT_FILE}")"
  model_tag="$(sanitize_path_part "${MODEL_PATH}")"
  prompt_k="$(( (MAX_PROMPT_LENGTH + 999) / 1000 ))k"
  output_k="$(( (MAX_NEW_TOKENS + 999) / 1000 ))k"
  context_k="$(( (VLLM_MAX_MODEL_LEN + 999) / 1000 ))k"

  run_tag="${INFERENCE_BACKEND}_tp${VLLM_TENSOR_PARALLEL_SIZE}_dp${VLLM_DATA_PARALLEL_SIZE}"
  if [[ "${INFERENCE_BACKEND}" == "vllm_async" ]]; then
    run_tag="${run_tag}_c${VLLM_ASYNC_CONCURRENCY}"
  fi
  run_tag="${run_tag}_ctx${context_k}_p${prompt_k}_o${output_k}_r${MAX_TOOL_ROUNDS}"

  printf 'outputs/inference/%s/%s/%s/%s' "${split_name}" "${input_tag}" "${model_tag}" "${run_tag}"
}

if [[ -n "${USER_OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${USER_OUTPUT_DIR}"
else
  OUTPUT_DIR="$(build_default_output_dir)"
fi

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
  --max_tool_rounds "${MAX_TOOL_ROUNDS}"
  --temperature "${TEMPERATURE}"
  --top_p "${TOP_P}"
  --eval_timeout "${EVAL_TIMEOUT}"
  --eval_workers "${EVAL_WORKERS}"
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

if [[ "${INFERENCE_BACKEND}" == "vllm" || "${INFERENCE_BACKEND}" == "vllm_async" ]]; then
  cmd+=(--vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}")
fi

if [[ "${INFERENCE_BACKEND}" == "vllm_async" ]]; then
  cmd+=(--vllm_async_concurrency "${VLLM_ASYNC_CONCURRENCY}")
fi

"${cmd[@]}"
