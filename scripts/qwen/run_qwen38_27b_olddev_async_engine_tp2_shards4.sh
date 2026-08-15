#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

PYTHON_BIN="${PYTHON_BIN:-/home/ec2-user/miniconda3/envs/nl2sql312/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/ec2-user/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}"
INPUT_FILE="${INPUT_FILE:-outputs/old-dev-schema-tool-unpatched.jsonl}"
DATABASE_DIR="${DATABASE_DIR:-databases/dev_databases}"
DIFF_JSON_PATH="${DIFF_JSON_PATH:-data/bird_dev_data/raw/bird_dev_unpatched.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/inference/dev/old-dev-schema-tool-unpatched/Qwen3.8-27B/full1534_tp2_shards4_temp0_async_engine_qwen_native}"

NUM_EXAMPLES="${NUM_EXAMPLES:--1}"
TP="${TP:-2}"
SHARDS="${SHARDS:-4}"
TEMPERATURE="${TEMPERATURE:-0.0}"
CONCURRENCY_PER_SHARD="${CONCURRENCY_PER_SHARD:-16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-43000}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-34000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8000}"
MAX_TOOL_ROUNDS="${MAX_TOOL_ROUNDS:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.93}"
IDLE_MEMORY_MB="${IDLE_MEMORY_MB:-5000}"

GPU_GROUPS=("0,1" "2,3" "4,5" "6,7")

mkdir -p "${OUTPUT_DIR}"
exec > >(tee -a "${OUTPUT_DIR}.log") 2>&1

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

echo "[$(date -Is)] starting Qwen3.8 async-engine temp0"
echo "[$(date -Is)] model=${MODEL_PATH}"
echo "[$(date -Is)] input=${INPUT_FILE}"
echo "[$(date -Is)] output=${OUTPUT_DIR}"
echo "[$(date -Is)] tp=${TP} shards=${SHARDS} concurrency=${CONCURRENCY_PER_SHARD} max_model_len=${MAX_MODEL_LEN}"

PYTHONPATH="${PYTHONPATH:-src:.}" "${PYTHON_BIN}" scripts/run_inference_bird_qwen_async.py \
  --model_name_or_path "${MODEL_PATH}" \
  --input_file "${INPUT_FILE}" \
  --database_dir "${DATABASE_DIR}" \
  --diff_json_path "${DIFF_JSON_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --num_examples "${NUM_EXAMPLES}" \
  --num_shards "${SHARDS}" \
  --gpu_groups "${GPU_GROUPS[@]}" \
  --max_prompt_length "${MAX_PROMPT_LENGTH}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --max_tool_rounds "${MAX_TOOL_ROUNDS}" \
  --temperature "${TEMPERATURE}" \
  --top_p 1.0 \
  --top_k 20 \
  --eval_timeout 60 \
  --eval_workers 16 \
  --vllm_tensor_parallel_size "${TP}" \
  --vllm_gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
  --vllm_max_model_len "${MAX_MODEL_LEN}" \
  --vllm_async_concurrency "${CONCURRENCY_PER_SHARD}" \
  --tool_choice_policy required_first \
  --empty_tool_retries 1 \
  --overwrite

echo "[$(date -Is)] complete"
