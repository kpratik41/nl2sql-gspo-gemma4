#!/usr/bin/env bash
# pass@16 over the full BIRD training file for outputs/checkpoint-90.
# 4 shards x tensor-parallel-2 = all 8 GPUs, then merge into <base>/merged.
set -euo pipefail

cd /home/ubuntu/nl2sql-gspo-gemma4

PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/nl2sql-gspo-gemma4/.venv/bin/python}"
CKPT_DIR="${CKPT_DIR:-outputs/checkpoint-90}"
INPUT_FILE="${INPUT_FILE:-outputs/train-6601-schema-bare-tool.jsonl}"
BASE_OUT="${BASE_OUT:-outputs/passk/train6601_bare_tool_ckpt90_temp1p2_tp2_shards4}"
LIMIT="${LIMIT:--1}"
TEMPERATURE="${TEMPERATURE:-1.2}"
NUM_GENERATIONS="${NUM_GENERATIONS:-16}"

mkdir -p "${BASE_OUT}"
exec > >(tee -a "${BASE_OUT}/run_passk16_tp2_shards4.log") 2>&1

# Unbuffered so per-shard progress is visible live instead of sitting in a pipe
# buffer until the worker exits.
export PYTHONUNBUFFERED=1
export PYTHONPATH=".:src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

echo "[$(date -Is)] starting pass@${NUM_GENERATIONS} on training data"
echo "[$(date -Is)] checkpoint=${CKPT_DIR}"
echo "[$(date -Is)] input=${INPUT_FILE} limit=${LIMIT}"
echo "[$(date -Is)] output=${BASE_OUT}"
echo "[$(date -Is)] temperature=${TEMPERATURE} tp=2 shards=4"

GPU_GROUPS=("0,1" "2,3" "4,5" "6,7")
pids=()

for i in 0 1 2 3; do
  echo "[$(date -Is)] launching shard ${i}/4 on GPUs ${GPU_GROUPS[$i]}"
  CUDA_VISIBLE_DEVICES="${GPU_GROUPS[$i]}" \
  "${PYTHON_BIN}" scripts/run_passk_bird.py \
    --model_name_or_path "${CKPT_DIR}" \
    --input_file "${INPUT_FILE}" \
    --database_dir databases/train_databases \
    --diff_json_path data/bird_train_data/raw/train.json \
    --output_dir "${BASE_OUT}" \
    --limit "${LIMIT}" \
    --shard_index "${i}" \
    --num_shards 4 \
    --num_generations "${NUM_GENERATIONS}" \
    --temperature "${TEMPERATURE}" \
    --top_p 1.0 \
    --max_prompt_length 30000 \
    --max_new_tokens 8000 \
    --max_tool_rounds 8 \
    --eval_timeout 60 \
    --eval_workers 16 \
    --vllm_tensor_parallel_size 2 \
    --vllm_gpu_memory_utilization 0.93 \
    --vllm_max_model_len 43000 \
    --vllm_async_concurrency 16 \
    --overwrite \
    > "${BASE_OUT}/shard${i}.log" 2>&1 &
  pids+=($!)
done

failed=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "[$(date -Is)] shard ${i} finished OK"
  else
    echo "[$(date -Is)] shard ${i} FAILED (see ${BASE_OUT}/shard${i}.log)"
    failed=1
  fi
done

if [[ "${failed}" -ne 0 ]]; then
  echo "[$(date -Is)] one or more shards failed; skipping merge"
  exit 1
fi

echo "[$(date -Is)] all shards done, merging"
"${PYTHON_BIN}" scripts/run_passk_bird.py \
  --merge_shard_dirs \
    "${BASE_OUT}/shard-00000-of-00004" \
    "${BASE_OUT}/shard-00001-of-00004" \
    "${BASE_OUT}/shard-00002-of-00004" \
    "${BASE_OUT}/shard-00003-of-00004" \
  --merge_output_dir "${BASE_OUT}/merged" \
  --input_file "${INPUT_FILE}" \
  --database_dir databases/train_databases \
  --diff_json_path data/bird_train_data/raw/train.json \
  --limit "${LIMIT}" \
  --num_generations "${NUM_GENERATIONS}" \
  --overwrite

echo "[$(date -Is)] complete"
echo "[$(date -Is)] merged summary=${BASE_OUT}/merged/passk_summary.md"
