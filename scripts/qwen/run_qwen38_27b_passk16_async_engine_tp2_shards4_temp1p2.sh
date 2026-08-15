#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

PYTHON_BIN="${PYTHON_BIN:-/home/ec2-user/miniconda3/envs/nl2sql312/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/ec2-user/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}"
INPUT_FILE="${INPUT_FILE:-outputs/old-dev-schema-tool-unpatched.jsonl}"
DATABASE_DIR="${DATABASE_DIR:-databases/dev_databases}"
DIFF_JSON_PATH="${DIFF_JSON_PATH:-data/bird_dev_data/raw/bird_dev_unpatched.json}"
BASE_OUT="${BASE_OUT:-outputs/passk/qwen3p8_27b_olddev_schema_tool_passk16_temp1p2_tp2_shards4_c16_async_engine}"
LOG_DIR="${LOG_DIR:-logs/qwen38_27b_passk16_async_engine_tp2_shards4_temp1p2_c16}"

TOTAL="${TOTAL:-1534}"
TP="${TP:-2}"
SHARDS="${SHARDS:-4}"
NUM_GENERATIONS="${NUM_GENERATIONS:-16}"
TEMPERATURE="${TEMPERATURE:-1.2}"
CONCURRENCY_PER_SHARD="${CONCURRENCY_PER_SHARD:-16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-43000}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-34000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8000}"
MAX_TOOL_ROUNDS="${MAX_TOOL_ROUNDS:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.93}"
IDLE_MEMORY_MB="${IDLE_MEMORY_MB:-5000}"

GPU_GROUPS=("0,1" "2,3" "4,5" "6,7")

mkdir -p "${BASE_OUT}" "${LOG_DIR}"
exec > >(tee -a "${BASE_OUT}/queue_and_run.log") 2>&1

max_gpu_memory_used_mb() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk 'BEGIN{m=0} {if ($1+0>m) m=$1+0} END{print m}'
}

echo "[$(date -Is)] waiting for GPUs to be idle; threshold=${IDLE_MEMORY_MB} MiB"
while true; do
  used="$(max_gpu_memory_used_mb)"
  if [[ "${used}" -lt "${IDLE_MEMORY_MB}" ]]; then
    echo "[$(date -Is)] GPUs idle enough; max_used=${used} MiB"
    break
  fi
  echo "[$(date -Is)] GPUs busy; max_used=${used} MiB"
  sleep 120
done

echo "[$(date -Is)] starting Qwen3.8 async-engine pass@${NUM_GENERATIONS}"
echo "[$(date -Is)] model=${MODEL_PATH}"
echo "[$(date -Is)] input=${INPUT_FILE}"
echo "[$(date -Is)] output=${BASE_OUT}"
echo "[$(date -Is)] tp=${TP} shards=${SHARDS} concurrency=${CONCURRENCY_PER_SHARD} max_model_len=${MAX_MODEL_LEN}"

pids=()
for shard in $(seq 0 $((SHARDS - 1))); do
  shard_log="${LOG_DIR}/shard${shard}.log"
  echo "[$(date -Is)] launching shard=${shard}/${SHARDS} gpus=${GPU_GROUPS[$shard]} log=${shard_log}"
  CUDA_VISIBLE_DEVICES="${GPU_GROUPS[$shard]}" \
  PYTHONUNBUFFERED=1 \
  VLLM_CACHE_ROOT="${BASE_OUT}/shard-${shard}-vllm-cache" \
  TORCHINDUCTOR_CACHE_DIR="${BASE_OUT}/shard-${shard}-torchinductor-cache" \
  PYTHONPATH="${PYTHONPATH:-src:.}" "${PYTHON_BIN}" scripts/run_passk_bird_qwen_async.py \
    --model_name_or_path "${MODEL_PATH}" \
    --input_file "${INPUT_FILE}" \
    --database_dir "${DATABASE_DIR}" \
    --diff_json_path "${DIFF_JSON_PATH}" \
    --output_dir "${BASE_OUT}" \
    --limit "${TOTAL}" \
    --shard_index "${shard}" \
    --num_shards "${SHARDS}" \
    --num_generations "${NUM_GENERATIONS}" \
    --temperature "${TEMPERATURE}" \
    --top_p 1.0 \
    --top_k 20 \
    --max_prompt_length "${MAX_PROMPT_LENGTH}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --max_tool_rounds "${MAX_TOOL_ROUNDS}" \
    --eval_timeout 60 \
    --eval_workers 16 \
    --vllm_tensor_parallel_size "${TP}" \
    --vllm_gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
    --vllm_max_model_len "${MAX_MODEL_LEN}" \
    --vllm_async_concurrency "${CONCURRENCY_PER_SHARD}" \
    --tool_choice_policy required_first \
    --empty_tool_retries 1 \
    --overwrite \
    >"${shard_log}" 2>&1 &
  pids+=("$!")
done

failed=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "[$(date -Is)] shard ${i} finished OK"
  else
    echo "[$(date -Is)] shard ${i} FAILED"
    failed=1
  fi
done

if [[ "${failed}" -ne 0 ]]; then
  echo "[$(date -Is)] one or more shards failed; skipping merge"
  exit 1
fi

merge_dirs=()
for shard in $(seq 0 $((SHARDS - 1))); do
  merge_dirs+=( "$(printf '%s/shard-%05d-of-%05d' "${BASE_OUT}" "${shard}" "${SHARDS}")" )
done

echo "[$(date -Is)] all shards done, merging"
PYTHONPATH="${PYTHONPATH:-src:.}" "${PYTHON_BIN}" scripts/run_passk_bird_qwen_async.py \
  --merge_shard_dirs "${merge_dirs[@]}" \
  --merge_output_dir "${BASE_OUT}/merged" \
  --model_name_or_path "${MODEL_PATH}" \
  --input_file "${INPUT_FILE}" \
  --database_dir "${DATABASE_DIR}" \
  --diff_json_path "${DIFF_JSON_PATH}" \
  --limit "${TOTAL}" \
  --num_generations "${NUM_GENERATIONS}" \
  --overwrite

echo "[$(date -Is)] complete"
echo "[$(date -Is)] merged summary=${BASE_OUT}/merged/passk_summary.md"
