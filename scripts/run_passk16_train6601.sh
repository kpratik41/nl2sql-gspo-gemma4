#!/usr/bin/env bash
# pass@16 over the full BIRD training file.
#
# Shards the examples across NUM_SHARDS independent async-vLLM processes, each
# holding a tensor-parallel-TP engine on its own GPU group, then merges into
# <BASE_OUT>/merged.
#
# Defaults: base google/gemma-4-31B-it, temperature 1.2, tp=1, 8 shards.
#
#   bash scripts/run_passk16_train6601.sh
#   MODEL=outputs/checkpoint-90 TP=2 NUM_SHARDS=4 bash scripts/run_passk16_train6601.sh
set -euo pipefail

cd /home/ubuntu/nl2sql-gspo-gemma4

PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/nl2sql-gspo-gemma4/.venv/bin/python}"
MODEL="${MODEL:-google/gemma-4-31B-it}"
INPUT_FILE="${INPUT_FILE:-outputs/train-6601-schema-bare-tool.jsonl}"
DATABASE_DIR="${DATABASE_DIR:-databases/train_databases}"
DIFF_JSON="${DIFF_JSON:-data/bird_train_data/raw/train.json}"
TP="${TP:-1}"
NUM_SHARDS="${NUM_SHARDS:-8}"
TEMPERATURE="${TEMPERATURE:-1.2}"
NUM_GENERATIONS="${NUM_GENERATIONS:-16}"
LIMIT="${LIMIT:--1}"
TAG="${TAG:-gemma4-31b-it}"
BASE_OUT="${BASE_OUT:-outputs/passk/train6601_bare_tool_${TAG}_temp1p2_tp${TP}_shards${NUM_SHARDS}}"

TOTAL_GPUS=$(( TP * NUM_SHARDS ))
VISIBLE=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if [[ "${TOTAL_GPUS}" -gt "${VISIBLE}" ]]; then
  echo "[passk] need TP*NUM_SHARDS=${TOTAL_GPUS} GPUs but only ${VISIBLE} present" >&2
  exit 1
fi

mkdir -p "${BASE_OUT}"
exec > >(tee -a "${BASE_OUT}/run_passk16.log") 2>&1

export PYTHONUNBUFFERED=1
export PYTHONPATH=".:src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

echo "[$(date -Is)] starting pass@${NUM_GENERATIONS} on training data"
echo "[$(date -Is)] model=${MODEL}"
echo "[$(date -Is)] input=${INPUT_FILE} limit=${LIMIT}"
echo "[$(date -Is)] database_dir=${DATABASE_DIR}"
echo "[$(date -Is)] output=${BASE_OUT}"
echo "[$(date -Is)] temperature=${TEMPERATURE} tp=${TP} shards=${NUM_SHARDS}"

pids=()
for (( i=0; i<NUM_SHARDS; i++ )); do
  # Contiguous GPU group of size TP for this shard.
  group=""
  for (( g=0; g<TP; g++ )); do
    gpu=$(( i * TP + g ))
    group="${group:+${group},}${gpu}"
  done
  echo "[$(date -Is)] launching shard ${i}/${NUM_SHARDS} on GPUs ${group}"
  CUDA_VISIBLE_DEVICES="${group}" \
  "${PYTHON_BIN}" scripts/run_passk_bird.py \
    --model_name_or_path "${MODEL}" \
    --input_file "${INPUT_FILE}" \
    --database_dir "${DATABASE_DIR}" \
    --diff_json_path "${DIFF_JSON}" \
    --output_dir "${BASE_OUT}" \
    --limit "${LIMIT}" \
    --shard_index "${i}" \
    --num_shards "${NUM_SHARDS}" \
    --num_generations "${NUM_GENERATIONS}" \
    --temperature "${TEMPERATURE}" \
    --top_p 1.0 \
    --max_prompt_length 30000 \
    --max_new_tokens 8000 \
    --max_tool_rounds 8 \
    --eval_timeout 60 \
    --eval_workers 16 \
    --vllm_tensor_parallel_size "${TP}" \
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
merge_dirs=()
for (( i=0; i<NUM_SHARDS; i++ )); do
  merge_dirs+=( "$(printf '%s/shard-%05d-of-%05d' "${BASE_OUT}" "${i}" "${NUM_SHARDS}")" )
done

"${PYTHON_BIN}" scripts/run_passk_bird.py \
  --merge_shard_dirs "${merge_dirs[@]}" \
  --merge_output_dir "${BASE_OUT}/merged" \
  --input_file "${INPUT_FILE}" \
  --database_dir "${DATABASE_DIR}" \
  --diff_json_path "${DIFF_JSON}" \
  --limit "${LIMIT}" \
  --num_generations "${NUM_GENERATIONS}" \
  --overwrite

echo "[$(date -Is)] complete"
echo "[$(date -Is)] merged summary=${BASE_OUT}/merged/passk_summary.md"
