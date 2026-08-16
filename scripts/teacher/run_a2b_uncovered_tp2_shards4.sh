#!/usr/bin/env bash
# Stage A2b — sampled gold-conditioned teacher traces over A2-uncovered all-wrong ids.
set -euo pipefail

cd /home/ec2-user/consensus/nl2sql-gspo-gemma4

export PYTHONUNBUFFERED=1
export PYTHONPATH=".:src:scripts:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

PY="${PY:-/home/ec2-user/miniconda3/envs/nl2sql312/bin/python}"
MODEL="${MODEL:-/home/ec2-user/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/518276fb130dc81caf9a4f772e65e63ef2526493}"
TP="${TP:-2}"
NUM_SHARDS="${NUM_SHARDS:-4}"
NUM_SAMPLES="${NUM_SAMPLES:-8}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-1.0}"
HINT="${HINT:-full_sql}"
OUT="${OUT:-outputs/teacher/a2b_uncovered_tp2_shards4}"
TARGETS="${TARGETS:-outputs/teacher/target_idx_all_wrong_a2_uncovered.txt}"

TOTAL=$(( TP * NUM_SHARDS ))
VISIBLE=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if [[ "${TOTAL}" -gt "${VISIBLE}" ]]; then
  echo "[a2b] need TP*NUM_SHARDS=${TOTAL} GPUs, only ${VISIBLE} visible" >&2
  exit 1
fi

mkdir -p "${OUT}"
exec > >(tee -a "${OUT}/run.log") 2>&1

echo "[$(date -Is)] Stage A2b uncovered: hint=${HINT} samples=${NUM_SAMPLES} temp=${TEMPERATURE} top_p=${TOP_P} tp=${TP} shards=${NUM_SHARDS}"
echo "[$(date -Is)] targets=${TARGETS} out=${OUT}"

pids=()
for (( i=0; i<NUM_SHARDS; i++ )); do
  group=""
  for (( g=0; g<TP; g++ )); do
    gpu=$(( i * TP + g ))
    group="${group:+${group},}${gpu}"
  done
  echo "[$(date -Is)] launching shard ${i}/${NUM_SHARDS} on GPUs ${group}"
  CUDA_VISIBLE_DEVICES="${group}" "${PY}" scripts/teacher/gen_teacher_traces.py \
    --model_name_or_path "${MODEL}" \
    --input_file outputs/train-6601-schema-bare-tool.jsonl \
    --database_dir databases/train_databases \
    --target_idx_file "${TARGETS}" \
    --output_dir "${OUT}" \
    --hint_strategy "${HINT}" \
    --num_samples "${NUM_SAMPLES}" \
    --temperature "${TEMPERATURE}" --top_p "${TOP_P}" \
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
  if wait "${pids[$i]}"; then
    echo "[$(date -Is)] shard ${i} OK"
  else
    echo "[$(date -Is)] shard ${i} FAILED (${OUT}/shard${i}.log)"
    failed=1
  fi
done

if [[ "${failed}" -ne 0 ]]; then
  echo "[$(date -Is)] shard failure; skipping merge"
  exit 1
fi

echo "[$(date -Is)] merging shards"
dirs=()
for (( i=0; i<NUM_SHARDS; i++ )); do
  dirs+=( "$(printf '%s/shard-%05d-of-%05d' "${OUT}" "${i}" "${NUM_SHARDS}")" )
done

"${PY}" scripts/teacher/gen_teacher_traces.py \
  --merge_shard_dirs "${dirs[@]}" \
  --merge_output_dir "${OUT}/merged" \
  --hint_strategy "${HINT}" --num_samples "${NUM_SAMPLES}" --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}" \
  --output_dir "${OUT}" --overwrite

echo "[$(date -Is)] complete"
