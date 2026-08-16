#!/usr/bin/env bash
# pass@16 on BIRD dev (1534, unpatched) for the sft_rft3 DAPO checkpoint-10.
#
# Runs from the consensus-branch checkout (this file's repo) because the model is a
# Gemma checkpoint. Data, databases and the model itself live in the sibling
# nl2sql-gspo-gemma4 checkout, so those are passed as absolute paths.
#
# 4 shards x tp=2 = 8 GPUs, async vLLM engine, temperature 1.2, 16 generations.
#
#   bash scripts/run_gemma31b_sftrft3_ckpt10_passk16_dev.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

MAIN_REPO="${MAIN_REPO:-/home/ec2-user/consensus/nl2sql-gspo-gemma4}"
PYTHON_BIN="${PYTHON_BIN:-/home/ec2-user/miniconda3/envs/nl2sql312/bin/python}"
MODEL="${MODEL:-${MAIN_REPO}/outputs/sft_rft3/gemma4_31b_sftckpt56_dapo12_lr2e6_beta_0p005_0p001_0/checkpoint-10}"
INPUT_FILE="${INPUT_FILE:-${MAIN_REPO}/outputs/old-dev-schema-tool-unpatched.jsonl}"
DATABASE_DIR="${DATABASE_DIR:-${MAIN_REPO}/databases/dev_databases}"
DIFF_JSON="${DIFF_JSON:-$(pwd)/data/bird_dev_data/raw/bird_dev_unpatched.json}"

TP="${TP:-2}"
NUM_SHARDS="${NUM_SHARDS:-4}"
TEMPERATURE="${TEMPERATURE:-1.2}"
NUM_GENERATIONS="${NUM_GENERATIONS:-16}"
LIMIT="${LIMIT:-1534}"
IDLE_MEMORY_MB="${IDLE_MEMORY_MB:-5000}"

BASE_OUT="${BASE_OUT:-${MAIN_REPO}/outputs/passk/gemma4_31b_sftrft3_ckpt10_olddev_schema_tool_passk16_temp1p2_tp${TP}_shards${NUM_SHARDS}}"
RUN_LOG="${BASE_OUT}/run_passk16.log"
STATUS_FILE="${BASE_OUT}/PROGRESS.md"

mkdir -p "${BASE_OUT}"

max_gpu_memory_used_mb() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
    | awk 'BEGIN{m=0} {if ($1+0>m) m=$1+0} END{print m}'
}

# Background progress reporter: rewrites PROGRESS.md every 60s so the run's state can
# be read at any time from one file without grepping four shard logs.
write_progress() {
  local expected_per_shard=$(( (LIMIT / NUM_SHARDS + 1) * NUM_GENERATIONS ))
  while true; do
    {
      echo "# pass@16 progress — gemma4-31b sft_rft3 ckpt-10"
      echo
      echo "- updated: \`$(date -Is)\`"
      echo "- model: \`${MODEL}\`"
      echo "- output: \`${BASE_OUT}\`"
      echo "- config: \`tp=${TP} shards=${NUM_SHARDS} k=${NUM_GENERATIONS} temp=${TEMPERATURE} limit=${LIMIT}\`"
      echo
      echo "| shard | phase | progress | last update |"
      echo "| ---: | :-- | :-- | :-- |"
      for (( i=0; i<NUM_SHARDS; i++ )); do
        local log="${BASE_OUT}/shard${i}.log"
        local dir
        dir="$(printf '%s/shard-%05d-of-%05d' "${BASE_OUT}" "${i}" "${NUM_SHARDS}")"
        local phase="starting" prog="-" stamp="-"
        if [[ -f "${log}" ]]; then
          stamp="$(date -Is -r "${log}")"
          local gen ev
          gen="$(grep -c '^\[passk\] generated ' "${log}" 2>/dev/null || true)"
          ev="$(grep -o '\[passk\] evaluated [0-9]*/[0-9]*' "${log}" 2>/dev/null | tail -1 || true)"
          local lastgen
          lastgen="$(grep -o '\[passk\] generated [0-9]*/[0-9]*' "${log}" 2>/dev/null | tail -1 || true)"
          if [[ -f "${dir}/passk_summary.md" ]]; then
            phase="done"
            prog="$(grep -m1 '^| 16 |' "${dir}/passk_summary.md" 2>/dev/null || echo 'complete')"
          elif [[ -n "${ev}" ]]; then
            phase="evaluating"; prog="${ev#\[passk\] evaluated }"
          elif [[ -n "${lastgen}" ]]; then
            phase="generating"; prog="${lastgen#\[passk\] generated } of ~${expected_per_shard}"
          elif grep -q 'loading' "${log}" 2>/dev/null; then
            phase="loading engine"
          fi
        fi
        echo "| ${i} | ${phase} | ${prog} | ${stamp} |"
      done
      echo
      if [[ -f "${BASE_OUT}/merged/passk_summary.md" ]]; then
        echo "## MERGED RESULT READY"
        echo
        echo "\`${BASE_OUT}/merged/passk_summary.md\`"
      fi
      echo
      echo "GPU: \`$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ';')\`"
    } > "${STATUS_FILE}.tmp" 2>/dev/null
    mv -f "${STATUS_FILE}.tmp" "${STATUS_FILE}" 2>/dev/null || true
    sleep 60
  done
}

{
echo "[$(date -Is)] pass@16 dev for sft_rft3 ckpt-10 (consensus checkout)"
echo "[$(date -Is)] repo=$(pwd) branch=$(git rev-parse --abbrev-ref HEAD)"
echo "[$(date -Is)] model=${MODEL}"
echo "[$(date -Is)] input=${INPUT_FILE}"
echo "[$(date -Is)] output=${BASE_OUT}"
echo "[$(date -Is)] tp=${TP} shards=${NUM_SHARDS} k=${NUM_GENERATIONS} temp=${TEMPERATURE}"

TOTAL_GPUS=$(( TP * NUM_SHARDS ))
VISIBLE=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if [[ "${TOTAL_GPUS}" -gt "${VISIBLE}" ]]; then
  echo "[$(date -Is)] need ${TOTAL_GPUS} GPUs but only ${VISIBLE} present" >&2
  exit 1
fi

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

write_progress &
progress_pid=$!
trap 'kill "${progress_pid}" 2>/dev/null || true' EXIT

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$(pwd):$(pwd)/src:${PYTHONPATH:-}"

pids=()
for (( i=0; i<NUM_SHARDS; i++ )); do
  group=""
  for (( g=0; g<TP; g++ )); do
    gpu=$(( i * TP + g ))
    group="${group:+${group},}${gpu}"
  done
  echo "[$(date -Is)] launching shard ${i}/${NUM_SHARDS} on GPUs ${group}"
  CUDA_VISIBLE_DEVICES="${group}" \
  VLLM_CACHE_ROOT="${BASE_OUT}/shard-${i}-vllm-cache" \
  TORCHINDUCTOR_CACHE_DIR="${BASE_OUT}/shard-${i}-torchinductor-cache" \
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
    --max_prompt_length 34000 \
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
  echo "[$(date -Is)] raw generations are preserved at shard-*/passk_candidates_raw.jsonl"
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
  --model_name_or_path "${MODEL}" \
  --input_file "${INPUT_FILE}" \
  --database_dir "${DATABASE_DIR}" \
  --diff_json_path "${DIFF_JSON}" \
  --limit "${LIMIT}" \
  --num_generations "${NUM_GENERATIONS}" \
  --overwrite

echo "[$(date -Is)] complete; summary=${BASE_OUT}/merged/passk_summary.md"
} 2>&1 | tee -a "${RUN_LOG}"
