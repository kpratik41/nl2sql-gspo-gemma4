#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/ubuntu/sft_rft/nl2sql-gspo-gemma4}"
cd "${REPO_ROOT}"

mkdir -p logs
QUEUE_LOG="${QUEUE_LOG:-logs/sft_temp0_eval_queue_$(date +%Y%m%d_%H%M%S).log}"
exec > >(tee -a "${QUEUE_LOG}") 2>&1

SFT_OUTPUT_DIR="${SFT_OUTPUT_DIR:-outputs/sft/gemma4_31b_rft_sft_consensus_sft}"
INPUT_FILE="${INPUT_FILE:-outputs/old-dev-schema-tool-unpatched.jsonl}"
DATABASE_DIR="${DATABASE_DIR:-databases/dev_databases}"
DIFF_JSON_PATH="${DIFF_JSON_PATH:-data/bird_dev_data/raw/dev_20251106.json}"
CKPTS="${CKPTS:-10 20 30}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
INFERENCE_CUDA_VISIBLE_DEVICES="${INFERENCE_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
INFERENCE_BACKEND="${INFERENCE_BACKEND:-vllm_async}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-2}"
VLLM_DATA_PARALLEL_SIZE="${VLLM_DATA_PARALLEL_SIZE:-4}"
VLLM_ASYNC_CONCURRENCY="${VLLM_ASYNC_CONCURRENCY:-16}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.93}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-43000}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-34000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8000}"
MAX_TOOL_ROUNDS="${MAX_TOOL_ROUNDS:-8}"
EVAL_WORKERS="${EVAL_WORKERS:-16}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-60}"

echo "[queue] started at $(date -Is)"
echo "[queue] log=${QUEUE_LOG}"
echo "[queue] waiting for SFT output_dir=${SFT_OUTPUT_DIR}"
echo "[queue] ckpts=${CKPTS}"
echo "[queue] input_file=${INPUT_FILE}"
echo "[queue] backend=${INFERENCE_BACKEND} temp=${TEMPERATURE} tp=${VLLM_TENSOR_PARALLEL_SIZE} dp=${VLLM_DATA_PARALLEL_SIZE} concurrency=${VLLM_ASYNC_CONCURRENCY}"

while pgrep -f "scripts/sft/train_sft.py.*${SFT_OUTPUT_DIR}" >/dev/null; do
  latest_log="$(ls -t logs/a4_sft_*.log 2>/dev/null | head -n 1 || true)"
  if [[ -n "${latest_log}" ]]; then
    progress="$(grep -aoE '[0-9]+%\\|[^\\r\\n]*\\|[[:space:]]*[0-9]+/56[^\\r\\n]*' "${latest_log}" | tail -n 1 || true)"
    echo "[queue] $(date -Is) SFT still running ${progress}"
  else
    echo "[queue] $(date -Is) SFT still running"
  fi
  sleep 60
done

echo "[queue] SFT process ended at $(date -Is); starting queued temp0 evals"

for ckpt in ${CKPTS}; do
  model_dir="${SFT_OUTPUT_DIR}/checkpoint-${ckpt}"
  output_dir="${model_dir}/temp0_olddev_schema_tool_unpatched_vllm_async_tp${VLLM_TENSOR_PARALLEL_SIZE}_dp${VLLM_DATA_PARALLEL_SIZE}"

  if [[ ! -d "${model_dir}" ]]; then
    echo "[queue] missing checkpoint: ${model_dir}" >&2
    exit 1
  fi

  echo "[queue] starting ckpt=${ckpt} at $(date -Is)"
  echo "[queue] model_dir=${model_dir}"
  echo "[queue] output_dir=${output_dir}"

  APPEND_OUTPUT_TIMESTAMP=0 \
  DATABASE_DIR="${DATABASE_DIR}" \
  DIFF_JSON_PATH="${DIFF_JSON_PATH}" \
  EVAL_TIMEOUT="${EVAL_TIMEOUT}" \
  EVAL_WORKERS="${EVAL_WORKERS}" \
  INFERENCE_BACKEND="${INFERENCE_BACKEND}" \
  INFERENCE_CUDA_VISIBLE_DEVICES="${INFERENCE_CUDA_VISIBLE_DEVICES}" \
  INPUT_FILE="${INPUT_FILE}" \
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
  MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH}" \
  MAX_TOOL_ROUNDS="${MAX_TOOL_ROUNDS}" \
  MODEL_PATH="${model_dir}" \
  NUM_EXAMPLES=-1 \
  OUTPUT_DIR="${output_dir}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  TEMPERATURE="${TEMPERATURE}" \
  TOP_P="${TOP_P}" \
  VLLM_ASYNC_CONCURRENCY="${VLLM_ASYNC_CONCURRENCY}" \
  VLLM_DATA_PARALLEL_SIZE="${VLLM_DATA_PARALLEL_SIZE}" \
  VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION}" \
  VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN}" \
  VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE}" \
    bash scripts/launch_inference.sh

  echo "[queue] finished ckpt=${ckpt} at $(date -Is)"
done

echo "[queue] all queued temp0 evals finished at $(date -Is)"
