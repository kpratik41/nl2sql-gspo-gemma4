#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${INFERENCE_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

MODEL_PATH="${MODEL_PATH:-pratikkakkar/gemma-4-31b-it-bird-rl}"
INPUT_FILE="${INPUT_FILE:-outputs/bird_dev-schema.jsonl}"
DATABASE_DIR="${DATABASE_DIR:-databases/dev_databases}"
DIFF_JSON_PATH="${DIFF_JSON_PATH:-data/bird_dev_data/raw/bird_dev.json}"
USER_OUTPUT_DIR="${OUTPUT_DIR:-}"
NUM_EXAMPLES="${NUM_EXAMPLES:--1}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-44000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8000}"
MAX_TOOL_ROUNDS="${MAX_TOOL_ROUNDS:-8}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
# 30s matches the official BIRD evaluator (--meta_time_out default).
EVAL_TIMEOUT="${EVAL_TIMEOUT:-30}"
# Tool calls during generation keep the more generous budget; this is the
# model exploring the database, not the graded query.
TOOL_TIMEOUT="${TOOL_TIMEOUT:-60}"
EVAL_WORKERS="${EVAL_WORKERS:-16}"
SHARD_INDEX="${SHARD_INDEX:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
NO_APPEND_SHARD_TO_OUTPUT_DIR="${NO_APPEND_SHARD_TO_OUTPUT_DIR:-0}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.96}"
# Covers MAX_PROMPT_LENGTH + MAX_NEW_TOKENS with headroom for the tool loop,
# which appends each tool response to the running context.
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-53000}"
VLLM_ASYNC_CONCURRENCY="${VLLM_ASYNC_CONCURRENCY:-8}"
NO_RESUME="${NO_RESUME:-0}"
FALLBACK_SQL="${FALLBACK_SQL:-}"
# 1 = a rollout that exhausts MAX_TOOL_ROUNDS gets one non-tool turn to commit to
# SQL; 0 = cut it off at the cap and return no SQL.
FORCE_FINALIZE="${FORCE_FINALIZE:-1}"
APPEND_OUTPUT_TIMESTAMP="${APPEND_OUTPUT_TIMESTAMP:-1}"
OUTPUT_TIMESTAMP="${OUTPUT_TIMESTAMP:-}"
# tp=2: a 31B model in bf16 is ~62 GB of weights, so it does not fit on one
# 80 GB card at this context length. Extra GPUs are better spent on shards
# (data parallelism) than on widening tensor parallelism.
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-2}"

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

  run_tag="vllm_async_tp${VLLM_TENSOR_PARALLEL_SIZE}_c${VLLM_ASYNC_CONCURRENCY}"
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
  --tool_timeout "${TOOL_TIMEOUT}"
  --eval_workers "${EVAL_WORKERS}"
  --shard_index "${SHARD_INDEX}"
  --num_shards "${NUM_SHARDS}"
  --overwrite
)

if [[ "${NO_APPEND_SHARD_TO_OUTPUT_DIR}" == "1" ]]; then
  cmd+=(--no_append_shard_to_output_dir)
fi

if [[ -n "${VLLM_TENSOR_PARALLEL_SIZE}" ]]; then
  cmd+=(--vllm_tensor_parallel_size "${VLLM_TENSOR_PARALLEL_SIZE}")
fi

if [[ -n "${VLLM_MAX_MODEL_LEN}" ]]; then
  cmd+=(--vllm_max_model_len "${VLLM_MAX_MODEL_LEN}")
fi

if [[ "${NO_RESUME}" == "1" ]]; then
  cmd+=(--no_resume)
fi

if [[ "${FORCE_FINALIZE}" != "1" ]]; then
  cmd+=(--no_force_finalize)
fi

if [[ -n "${FALLBACK_SQL}" ]]; then
  cmd+=(--fallback_sql "${FALLBACK_SQL}")
fi

cmd+=(--vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}")
cmd+=(--vllm_async_concurrency "${VLLM_ASYNC_CONCURRENCY}")

"${cmd[@]}"
