#!/usr/bin/env bash
# Run 3 A5: temp-0 async vLLM dev eval, one checkpoint per GPU pair (tp=2, dp=1).
#
# Four checkpoints run concurrently on 8 GPUs:
#   ckpt-10 -> 0,1   ckpt-20 -> 2,3   ckpt-30 -> 4,5   ckpt-40 -> 6,7
#
# Results are written INSIDE each checkpoint directory.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/ubuntu/sft-rl/nl2sql-gspo-gemma4}"
cd "${REPO_ROOT}"

SFT_OUTPUT_DIR="${SFT_OUTPUT_DIR:-outputs/sft/run3_gemma4_31b_rft_sft}"
INPUT_FILE="${INPUT_FILE:-outputs/old-dev-schema-tool-unpatched.jsonl}"
DATABASE_DIR="${DATABASE_DIR:-databases/dev_databases}"
DIFF_JSON_PATH="${DIFF_JSON_PATH:-data/bird_dev_data/raw/dev_20251106.json}"
CKPTS="${CKPTS:-10 20 30 40}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
INFERENCE_BACKEND="${INFERENCE_BACKEND:-vllm_async}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
TP="${TP:-2}"
DP="${DP:-1}"
VLLM_ASYNC_CONCURRENCY="${VLLM_ASYNC_CONCURRENCY:-16}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.93}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-43000}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-34000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8000}"
MAX_TOOL_ROUNDS="${MAX_TOOL_ROUNDS:-8}"
EVAL_WORKERS="${EVAL_WORKERS:-16}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-60}"

mkdir -p logs
QUEUE_LOG="${QUEUE_LOG:-logs/run3_temp0_eval_parallel.log}"
exec > >(tee -a "${QUEUE_LOG}") 2>&1

echo "[a5] started at $(date -Is)"
echo "[a5] sft_output_dir=${SFT_OUTPUT_DIR}"
echo "[a5] ckpts=${CKPTS}"
echo "[a5] input_file=${INPUT_FILE}"
echo "[a5] backend=${INFERENCE_BACKEND} temp=${TEMPERATURE} tp=${TP} dp=${DP}"

pids=()
tags=()
i=0
for ckpt in ${CKPTS}; do
  model_dir="${SFT_OUTPUT_DIR}/checkpoint-${ckpt}"
  if [[ ! -d "${model_dir}" ]]; then
    echo "[a5] missing checkpoint: ${model_dir}" >&2
    exit 1
  fi
  if [[ ! -f "${model_dir}/processor_config.json" ]]; then
    install -m 0644 gemma-4-31b-it-local/processor_config.json "${model_dir}/processor_config.json"
    echo "[a5] patched processor_config.json into ${model_dir}"
  fi

  group="$(( i * TP )),$(( i * TP + 1 ))"
  # results live inside the checkpoint directory
  output_dir="${model_dir}/temp0_olddev_schema_tool_unpatched_vllm_async_tp${TP}_dp${DP}"
  mkdir -p "${output_dir}"

  echo "[a5] launching ckpt=${ckpt} on GPUs ${group} -> ${output_dir}"

  APPEND_OUTPUT_TIMESTAMP=0 \
  DATABASE_DIR="${DATABASE_DIR}" \
  DIFF_JSON_PATH="${DIFF_JSON_PATH}" \
  EVAL_TIMEOUT="${EVAL_TIMEOUT}" \
  EVAL_WORKERS="${EVAL_WORKERS}" \
  INFERENCE_BACKEND="${INFERENCE_BACKEND}" \
  INFERENCE_CUDA_VISIBLE_DEVICES="${group}" \
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
  VLLM_DATA_PARALLEL_SIZE="${DP}" \
  VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION}" \
  VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN}" \
  VLLM_TENSOR_PARALLEL_SIZE="${TP}" \
    bash scripts/launch_inference.sh > "${model_dir}/temp0_eval.log" 2>&1 &

  pids+=($!)
  tags+=("${ckpt}")
  i=$(( i + 1 ))
done

failed=0
for j in "${!pids[@]}"; do
  if wait "${pids[$j]}"; then
    echo "[a5] ckpt-${tags[$j]} OK at $(date -Is)"
  else
    echo "[a5] ckpt-${tags[$j]} FAILED; see ${SFT_OUTPUT_DIR}/checkpoint-${tags[$j]}/temp0_eval.log" >&2
    failed=1
  fi
done

echo "[a5] all evals finished at $(date -Is) failed=${failed}"
exit "${failed}"
