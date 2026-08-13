#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/ubuntu/sft_rft/nl2sql-gspo-gemma4}"
cd "${REPO_ROOT}"

mkdir -p logs
QUEUE_LOG="${QUEUE_LOG:-logs/sft_passk16_olddev_queue_$(date +%Y%m%d_%H%M%S).log}"
exec > >(tee -a "${QUEUE_LOG}") 2>&1

SFT_OUTPUT_DIR="${SFT_OUTPUT_DIR:-outputs/sft/gemma4_31b_rft_sft_consensus_sft}"
INPUT_FILE="${INPUT_FILE:-outputs/old-dev-schema-tool-unpatched.jsonl}"
DATABASE_DIR="${DATABASE_DIR:-databases/dev_databases}"
DIFF_JSON_PATH="${DIFF_JSON_PATH:-data/bird_dev_data/raw/dev_20251106.json}"
CKPTS="${CKPTS:-10}"
WAIT_FOR_IDLE="${WAIT_FOR_IDLE:-0}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
TP="${TP:-2}"
NUM_SHARDS="${NUM_SHARDS:-4}"
TEMPERATURE="${TEMPERATURE:-1.2}"
NUM_GENERATIONS="${NUM_GENERATIONS:-16}"
LIMIT="${LIMIT:--1}"
VLLM_ASYNC_CONCURRENCY="${VLLM_ASYNC_CONCURRENCY:-16}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.93}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-43000}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-30000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8000}"
MAX_TOOL_ROUNDS="${MAX_TOOL_ROUNDS:-8}"
EVAL_WORKERS="${EVAL_WORKERS:-16}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-60}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

echo "[queue] started at $(date -Is)"
echo "[queue] log=${QUEUE_LOG}"
echo "[queue] ckpts=${CKPTS}"
echo "[queue] input_file=${INPUT_FILE}"
echo "[queue] pass@${NUM_GENERATIONS} temp=${TEMPERATURE} tp=${TP} shards=${NUM_SHARDS} concurrency=${VLLM_ASYNC_CONCURRENCY}"

if [[ "${WAIT_FOR_IDLE}" == "1" ]]; then
  while pgrep -f "scripts/run_passk_bird.py.*${SFT_OUTPUT_DIR}/checkpoint-" >/dev/null; do
    echo "[queue] $(date -Is) waiting for active pass@k checkpoint run to finish"
    sleep 60
  done
fi

TOTAL_GPUS=$(( TP * NUM_SHARDS ))
VISIBLE="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if [[ "${TOTAL_GPUS}" -gt "${VISIBLE}" ]]; then
  echo "[queue] need TP*NUM_SHARDS=${TOTAL_GPUS} GPUs but only ${VISIBLE} visible" >&2
  exit 1
fi

for ckpt in ${CKPTS}; do
  model_dir="${SFT_OUTPUT_DIR}/checkpoint-${ckpt}"
  base_out="${model_dir}/passk16_olddev_schema_tool_unpatched_temp1p2_tp${TP}_shards${NUM_SHARDS}"

  if [[ ! -d "${model_dir}" ]]; then
    echo "[queue] missing checkpoint: ${model_dir}" >&2
    exit 1
  fi
  if [[ ! -f "${model_dir}/processor_config.json" ]]; then
    install -m 0644 gemma-4-31b-it-local/processor_config.json "${model_dir}/processor_config.json"
    echo "[queue] patched processor_config.json into ${model_dir}"
  fi

  mkdir -p "${base_out}"
  echo "[queue] starting ckpt=${ckpt} at $(date -Is)"
  echo "[queue] model_dir=${model_dir}"
  echo "[queue] output_dir=${base_out}"

  pids=()
  for (( i=0; i<NUM_SHARDS; i++ )); do
    group=""
    for (( g=0; g<TP; g++ )); do
      gpu=$(( i * TP + g ))
      group="${group:+${group},}${gpu}"
    done

    echo "[queue] launching ckpt=${ckpt} shard=${i}/${NUM_SHARDS} gpus=${group}"
    CUDA_VISIBLE_DEVICES="${group}" \
    "${PYTHON_BIN}" scripts/run_passk_bird.py \
      --model_name_or_path "${model_dir}" \
      --input_file "${INPUT_FILE}" \
      --database_dir "${DATABASE_DIR}" \
      --diff_json_path "${DIFF_JSON_PATH}" \
      --output_dir "${base_out}" \
      --limit "${LIMIT}" \
      --shard_index "${i}" \
      --num_shards "${NUM_SHARDS}" \
      --num_generations "${NUM_GENERATIONS}" \
      --temperature "${TEMPERATURE}" \
      --top_p 1.0 \
      --max_prompt_length "${MAX_PROMPT_LENGTH}" \
      --max_new_tokens "${MAX_NEW_TOKENS}" \
      --max_tool_rounds "${MAX_TOOL_ROUNDS}" \
      --eval_timeout "${EVAL_TIMEOUT}" \
      --eval_workers "${EVAL_WORKERS}" \
      --vllm_tensor_parallel_size "${TP}" \
      --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
      --vllm_max_model_len "${VLLM_MAX_MODEL_LEN}" \
      --vllm_async_concurrency "${VLLM_ASYNC_CONCURRENCY}" \
      --overwrite \
      > "${base_out}/shard${i}.log" 2>&1 &
    pids+=($!)
  done

  failed=0
  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
      echo "[queue] ckpt=${ckpt} shard=${i} finished OK at $(date -Is)"
    else
      echo "[queue] ckpt=${ckpt} shard=${i} FAILED; see ${base_out}/shard${i}.log" >&2
      failed=1
    fi
  done

  if [[ "${failed}" -ne 0 ]]; then
    echo "[queue] ckpt=${ckpt} failed; skipping merge" >&2
    exit 1
  fi

  echo "[queue] ckpt=${ckpt} all shards done; merging at $(date -Is)"
  merge_dirs=()
  for (( i=0; i<NUM_SHARDS; i++ )); do
    merge_dirs+=( "$(printf '%s/shard-%05d-of-%05d' "${base_out}" "${i}" "${NUM_SHARDS}")" )
  done

  "${PYTHON_BIN}" scripts/run_passk_bird.py \
    --merge_shard_dirs "${merge_dirs[@]}" \
    --merge_output_dir "${base_out}/merged" \
    --input_file "${INPUT_FILE}" \
    --database_dir "${DATABASE_DIR}" \
    --diff_json_path "${DIFF_JSON_PATH}" \
    --limit "${LIMIT}" \
    --num_generations "${NUM_GENERATIONS}" \
    --overwrite

  echo "[queue] finished ckpt=${ckpt} at $(date -Is)"
  echo "[queue] summary=${base_out}/merged/passk_summary.md"
done

echo "[queue] all pass@k runs finished at $(date -Is)"
