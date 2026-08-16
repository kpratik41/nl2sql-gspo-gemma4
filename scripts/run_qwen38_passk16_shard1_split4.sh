#!/usr/bin/env bash
# Recover shard 1 of the Qwen3.8-27B pass@16 dev run by splitting it four ways
# across all 8 GPUs (4 sub-shards x tp=2) instead of re-running it on one pair.
#
# The original run sharded by `source_idx % 4 == shard_index`, so shard 1 is
# {idx : idx % 4 == 1}. Running --num_shards 16 with shard indices 1, 5, 9, 13
# reproduces exactly that set (96 examples each, 384 total) because
# {r : r % 16 in (1,5,9,13)} == {r : r % 4 == 1}. source_idx values are
# preserved, so the (idx, sample_id) merge key stays valid.
#
# Then: merge the 4 sub-shards -> shard-00001-of-00004, and merge all four
# top-level shards -> merged/.
#
#   bash scripts/run_qwen38_passk16_shard1_split4.sh
#   IDLE_MEMORY_MB=5000 bash scripts/run_qwen38_passk16_shard1_split4.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

BASE_OUT="${BASE_OUT:-outputs/passk/qwen3p8_27b_olddev_schema_tool_passk16_temp1p2_tp2_shards4_c16_async_engine_full1534}"
LOG_DIR="${LOG_DIR:-logs/qwen38_27b_passk16_async_engine_full1534_tp2_shards4_temp1p2_c16}"
RUN_LOG="${BASE_OUT}/recover_shard1_split4.log"
PYTHON_BIN="${PYTHON_BIN:-/home/ec2-user/miniconda3/envs/nl2sql312/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/ec2-user/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}"
INPUT_FILE="${INPUT_FILE:-outputs/old-dev-schema-tool-unpatched.jsonl}"
DATABASE_DIR="${DATABASE_DIR:-databases/dev_databases}"
DIFF_JSON_PATH="${DIFF_JSON_PATH:-data/bird_dev_data/raw/bird_dev_unpatched.json}"
IDLE_MEMORY_MB="${IDLE_MEMORY_MB:-5000}"

# Sub-shard index -> GPU pair. Indices are shard 1 of 4, split 16 ways.
SUB_SHARDS=(1 5 9 13)
GPU_GROUPS=("0,1" "2,3" "4,5" "6,7")
SUB_NUM_SHARDS=16

mkdir -p "${BASE_OUT}" "${LOG_DIR}"

max_gpu_memory_used_mb() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
    | awk 'BEGIN{m=0} {if ($1+0>m) m=$1+0} END{print m}'
}

{
echo "[$(date -Is)] shard1 split-4 recovery: sub-shards ${SUB_SHARDS[*]} of ${SUB_NUM_SHARDS}, tp=2 each, 8 GPUs"
echo "[$(date -Is)] waiting for GPUs idle; threshold=${IDLE_MEMORY_MB} MiB"
while true; do
  used=$(max_gpu_memory_used_mb)
  if [[ "${used}" -lt "${IDLE_MEMORY_MB}" ]]; then
    echo "[$(date -Is)] GPUs idle enough; max_used=${used} MiB"
    break
  fi
  echo "[$(date -Is)] GPUs busy; max_used=${used} MiB"
  sleep 120
done

pids=()
for i in "${!SUB_SHARDS[@]}"; do
  idx="${SUB_SHARDS[$i]}"
  gpus="${GPU_GROUPS[$i]}"
  echo "[$(date -Is)] launching sub-shard ${idx}/${SUB_NUM_SHARDS} on GPUs ${gpus}"
  CUDA_VISIBLE_DEVICES="${gpus}" \
  PYTHONUNBUFFERED=1 \
  VLLM_CACHE_ROOT="${BASE_OUT}/shard1sub-${idx}-vllm-cache" \
  TORCHINDUCTOR_CACHE_DIR="${BASE_OUT}/shard1sub-${idx}-torchinductor-cache" \
  PYTHONPATH="src:.:${PYTHONPATH:-}" "${PYTHON_BIN}" scripts/run_passk_bird_qwen_async.py \
    --model_name_or_path "${MODEL_PATH}" \
    --input_file "${INPUT_FILE}" \
    --database_dir "${DATABASE_DIR}" \
    --diff_json_path "${DIFF_JSON_PATH}" \
    --output_dir "${BASE_OUT}" \
    --limit 1534 \
    --shard_index "${idx}" \
    --num_shards "${SUB_NUM_SHARDS}" \
    --num_generations 16 \
    --temperature 1.2 \
    --top_p 1.0 \
    --top_k 20 \
    --max_prompt_length 34000 \
    --max_new_tokens 8000 \
    --max_tool_rounds 8 \
    --eval_timeout 60 \
    --eval_workers 16 \
    --vllm_tensor_parallel_size 2 \
    --vllm_gpu_memory_utilization 0.93 \
    --vllm_max_model_len 43000 \
    --vllm_async_concurrency 16 \
    --tool_choice_policy required_first \
    --empty_tool_retries 1 \
    --overwrite \
    > "${LOG_DIR}/shard1sub${idx}.log" 2>&1 &
  pids+=($!)
done

failed=0
for i in "${!pids[@]}"; do
  idx="${SUB_SHARDS[$i]}"
  if wait "${pids[$i]}"; then
    echo "[$(date -Is)] sub-shard ${idx} finished OK"
  else
    echo "[$(date -Is)] sub-shard ${idx} FAILED (see ${LOG_DIR}/shard1sub${idx}.log)"
    failed=1
  fi
done

if [[ "${failed}" -ne 0 ]]; then
  echo "[$(date -Is)] one or more sub-shards failed; skipping merges"
  exit 1
fi

# Stage 1: 4 sub-shards -> a drop-in replacement for shard-00001-of-00004.
# The summary written here is over all 1534 rows and so understates pass@k;
# only passk_candidates.jsonl from this directory is consumed downstream.
echo "[$(date -Is)] merging 4 sub-shards into shard-00001-of-00004"
sub_dirs=()
for idx in "${SUB_SHARDS[@]}"; do
  sub_dirs+=( "$(printf '%s/shard-%05d-of-%05d' "${BASE_OUT}" "${idx}" "${SUB_NUM_SHARDS}")" )
done

PYTHONPATH="src:.:${PYTHONPATH:-}" "${PYTHON_BIN}" scripts/run_passk_bird_qwen_async.py \
  --merge_shard_dirs "${sub_dirs[@]}" \
  --merge_output_dir "${BASE_OUT}/shard-00001-of-00004" \
  --model_name_or_path "${MODEL_PATH}" \
  --input_file "${INPUT_FILE}" \
  --database_dir "${DATABASE_DIR}" \
  --diff_json_path "${DIFF_JSON_PATH}" \
  --limit 1534 \
  --num_generations 16 \
  --temperature 1.2 \
  --top_p 1.0 \
  --top_k 20 \
  --max_new_tokens 8000 \
  --max_tool_rounds 8 \
  --vllm_tensor_parallel_size 2 \
  --vllm_async_concurrency 16 \
  --overwrite

# Stage 2: all four top-level shards -> the real 1534-example result.
echo "[$(date -Is)] merging all 4 shards into merged/"
PYTHONPATH="src:.:${PYTHONPATH:-}" "${PYTHON_BIN}" scripts/run_passk_bird_qwen_async.py \
  --merge_shard_dirs \
    "${BASE_OUT}/shard-00000-of-00004" \
    "${BASE_OUT}/shard-00001-of-00004" \
    "${BASE_OUT}/shard-00002-of-00004" \
    "${BASE_OUT}/shard-00003-of-00004" \
  --merge_output_dir "${BASE_OUT}/merged" \
  --model_name_or_path "${MODEL_PATH}" \
  --input_file "${INPUT_FILE}" \
  --database_dir "${DATABASE_DIR}" \
  --diff_json_path "${DIFF_JSON_PATH}" \
  --limit 1534 \
  --num_generations 16 \
  --temperature 1.2 \
  --top_p 1.0 \
  --top_k 20 \
  --max_new_tokens 8000 \
  --max_tool_rounds 8 \
  --vllm_tensor_parallel_size 2 \
  --vllm_async_concurrency 16 \
  --overwrite

echo "[$(date -Is)] complete; summary=${BASE_OUT}/merged/passk_summary.md"
} 2>&1 | tee -a "${RUN_LOG}"
