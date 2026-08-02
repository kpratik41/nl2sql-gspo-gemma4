#!/usr/bin/env bash
# Stage A2b — sampled gold-conditioned teacher traces over the all-wrong band.
# Sharded: NUM_SHARDS processes x TP GPUs each, then merged.
#
#   bash scripts/teacher/run_a2b_sampled.sh                 # tp=1, 8 shards
#   TP=2 NUM_SHARDS=4 bash scripts/teacher/run_a2b_sampled.sh
set -euo pipefail
cd /home/ubuntu/nl2sql-gspo-gemma4

export PYTHONUNBUFFERED=1
export PYTHONPATH=".:src:scripts:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

PY="${PY:-.venv/bin/python}"
TP="${TP:-1}"
NUM_SHARDS="${NUM_SHARDS:-8}"
NUM_SAMPLES="${NUM_SAMPLES:-8}"
TEMPERATURE="${TEMPERATURE:-0.7}"
OUT="${OUT:-outputs/teacher/a2b_sampled}"
TARGETS="${TARGETS:-outputs/teacher/target_idx_all_wrong.txt}"
HINT="${HINT:-full_sql}"

TOTAL=$(( TP * NUM_SHARDS ))
VISIBLE=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if [[ "${TOTAL}" -gt "${VISIBLE}" ]]; then
  echo "[a2b] need TP*NUM_SHARDS=${TOTAL} GPUs, only ${VISIBLE} visible" >&2; exit 1
fi

mkdir -p "${OUT}"
exec > >(tee -a "${OUT}/run.log") 2>&1
echo "[$(date -Is)] Stage A2b: hint=${HINT} samples=${NUM_SAMPLES} temp=${TEMPERATURE} tp=${TP} shards=${NUM_SHARDS}"

pids=()
for (( i=0; i<NUM_SHARDS; i++ )); do
  group=""
  for (( g=0; g<TP; g++ )); do gpu=$(( i * TP + g )); group="${group:+${group},}${gpu}"; done
  echo "[$(date -Is)] launching shard ${i}/${NUM_SHARDS} on GPUs ${group}"
  CUDA_VISIBLE_DEVICES="${group}" "${PY}" scripts/teacher/gen_teacher_traces.py \
    --model_name_or_path google/gemma-4-31B-it \
    --input_file outputs/train-6601-schema-bare-tool.jsonl \
    --database_dir databases/train_databases \
    --target_idx_file "${TARGETS}" \
    --output_dir "${OUT}" \
    --hint_strategy "${HINT}" \
    --num_samples "${NUM_SAMPLES}" \
    --temperature "${TEMPERATURE}" --top_p 1.0 \
    --max_prompt_length 34000 --max_new_tokens 8000 --max_tool_rounds 8 \
    --eval_timeout 60 \
    --shard_index "${i}" --num_shards "${NUM_SHARDS}" \
    --vllm_tensor_parallel_size "${TP}" \
    --vllm_gpu_memory_utilization 0.93 \
    --vllm_max_model_len 43000 \
    --vllm_async_concurrency 16 \
    --overwrite > "${OUT}/shard${i}.log" 2>&1 &
  pids+=($!)
done

failed=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then echo "[$(date -Is)] shard ${i} OK"; else echo "[$(date -Is)] shard ${i} FAILED (${OUT}/shard${i}.log)"; failed=1; fi
done
[[ "${failed}" -eq 0 ]] || { echo "[$(date -Is)] shard failure; skipping merge"; exit 1; }

echo "[$(date -Is)] merging"
dirs=()
for (( i=0; i<NUM_SHARDS; i++ )); do dirs+=( "$(printf '%s/shard-%05d-of-%05d' "${OUT}" "${i}" "${NUM_SHARDS}")" ); done
"${PY}" scripts/teacher/gen_teacher_traces.py \
  --merge_shard_dirs "${dirs[@]}" \
  --merge_output_dir "${OUT}/merged" \
  --hint_strategy "${HINT}" --num_samples "${NUM_SAMPLES}" --temperature "${TEMPERATURE}" \
  --output_dir "${OUT}" --overwrite
echo "[$(date -Is)] complete"
