#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ec2-user/consensus/nl2sql-gspo-gemma4-consensus"
CKPT="/home/ec2-user/consensus/nl2sql-gspo-gemma4/outputs/sft_rft3/gemma4_31b_sftckpt56_dapo12_lr2e6_beta_0p005_0p001_0/checkpoint-35"
INPUT_FILE="/home/ec2-user/consensus/nl2sql-gspo-gemma4/outputs/old-dev-schema-tool-unpatched.jsonl"
DATABASE_DIR="/home/ec2-user/consensus/nl2sql-gspo-gemma4/databases/dev_databases"
DIFF_JSON_PATH="${ROOT}/data/bird_dev_data/raw/bird_dev_unpatched.json"
PROCESSOR_CONFIG="${ROOT}/gemma-4-31b-it-local/processor_config.json"
RUN_TAG="temp0_olddev_schema_tool_unpatched_vllm_async_tp2_shards4_ctx43k"
OUTPUT_DIR="${CKPT}/${RUN_TAG}"
LOG="${CKPT}/${RUN_TAG}.log"
IDLE_MEMORY_MB="${IDLE_MEMORY_MB:-5000}"

cd "${ROOT}"
mkdir -p "${OUTPUT_DIR}"
install -m 0644 "${PROCESSOR_CONFIG}" "${CKPT}/processor_config.json"

max_gpu_memory_used_mb() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk 'BEGIN{m=0} {if ($1+0>m) m=$1+0} END{print m}'
}

{
  echo "[$(date -Is)] queued Gemma checkpoint-35 temp0 inference"
  echo "[$(date -Is)] branch=$(git branch --show-current) commit=$(git rev-parse --short HEAD)"
  echo "[$(date -Is)] ckpt=${CKPT}"
  echo "[$(date -Is)] input=${INPUT_FILE}"
  echo "[$(date -Is)] output=${OUTPUT_DIR}"
  echo "[$(date -Is)] patched processor_config=${CKPT}/processor_config.json"
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

  export TOKENIZERS_PARALLELISM=false
  export PYTHONUNBUFFERED=1
  export PYTHONPATH="${ROOT}/src:${ROOT}:${PYTHONPATH:-}"

  echo "[$(date -Is)] starting async vLLM sharded inference"
  /home/ec2-user/miniconda3/envs/nl2sql312/bin/python scripts/run_inference_bird_async_sharded.py \
    --model_name_or_path "${CKPT}" \
    --input_file "${INPUT_FILE}" \
    --database_dir "${DATABASE_DIR}" \
    --diff_json_path "${DIFF_JSON_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_examples -1 \
    --num_shards 4 \
    --gpu_groups 0,1 2,3 4,5 6,7 \
    --max_prompt_length 34000 \
    --max_new_tokens 8000 \
    --max_tool_rounds 8 \
    --temperature 0.0 \
    --top_p 1.0 \
    --eval_timeout 60 \
    --eval_workers 16 \
    --vllm_tensor_parallel_size 2 \
    --vllm_gpu_memory_utilization 0.93 \
    --vllm_max_model_len 43000 \
    --vllm_async_concurrency 16 \
    --overwrite
  echo "[$(date -Is)] complete; summary=${OUTPUT_DIR}/eval_summary.md"
} 2>&1 | tee -a "${LOG}"
