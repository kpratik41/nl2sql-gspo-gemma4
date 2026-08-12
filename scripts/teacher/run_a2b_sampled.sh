#!/usr/bin/env bash
# Stage A2b — sampled gold-conditioned teacher traces.
#
# Same generator as A2, but higher temperature and multiple trajectories per
# target. By default this uses an A2-uncovered target file when present, then
# falls back to the all-wrong target ids.
#
#   bash scripts/teacher/run_a2b_sampled.sh
#   TARGETS=outputs/teacher/target_idx_all_wrong_a2_uncovered.json bash scripts/teacher/run_a2b_sampled.sh
#   TEMPERATURE=0.8 NUM_SAMPLES=16 TP=2 NUM_SHARDS=4 bash scripts/teacher/run_a2b_sampled.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

if [[ -z "${PY:-}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PY=".venv/bin/python"
  else
    PY="python3"
  fi
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH=".:src:scripts:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

MODEL="${MODEL:-google/gemma-4-31B-it}"
INPUT_FILE="${INPUT_FILE:-outputs/train-6601-schema-bare-tool.jsonl}"
DATABASE_DIR="${DATABASE_DIR:-databases/train_databases}"
A2_TRACES="${A2_TRACES:-outputs/teacher/a2_greedy_tp2_shards4/merged/teacher_traces.jsonl}"
ALL_WRONG_IDS="${ALL_WRONG_IDS:-outputs/teacher/target_idx_all_wrong.json}"
UNCOVERED_PREFIX="${UNCOVERED_PREFIX:-outputs/teacher/target_idx_all_wrong_a2_uncovered}"
if [[ -z "${TARGETS:-}" ]]; then
  if [[ ! -f "${UNCOVERED_PREFIX}.json" && -f "${A2_TRACES}" && -f "${ALL_WRONG_IDS}" ]]; then
    "${PY}" scripts/teacher/build_a2_uncovered.py \
      --all-wrong-ids "${ALL_WRONG_IDS}" \
      --traces "${A2_TRACES}" \
      --output-prefix "${UNCOVERED_PREFIX}" \
      --overwrite
  fi
  if [[ -f "outputs/teacher/target_idx_all_wrong_a2_uncovered.json" ]]; then
    TARGETS="outputs/teacher/target_idx_all_wrong_a2_uncovered.json"
  elif [[ -f "outputs/teacher/target_idx_all_wrong_a2_uncovered.txt" ]]; then
    TARGETS="outputs/teacher/target_idx_all_wrong_a2_uncovered.txt"
  else
    TARGETS="outputs/teacher/target_idx_all_wrong.json"
  fi
fi
TP="${TP:-2}"
NUM_SHARDS="${NUM_SHARDS:-4}"
NUM_SAMPLES="${NUM_SAMPLES:-16}"
TEMPERATURE="${TEMPERATURE:-0.7}"
OUT="${OUT:-outputs/teacher/a2b_sampled_temp${TEMPERATURE}_samples${NUM_SAMPLES}_tp${TP}_shards${NUM_SHARDS}}"
LIMIT="${LIMIT:--1}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-34000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8000}"
MAX_TOOL_ROUNDS="${MAX_TOOL_ROUNDS:-8}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-43000}"
VLLM_ASYNC_CONCURRENCY="${VLLM_ASYNC_CONCURRENCY:-16}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.93}"

if [[ ! -f "${TARGETS}" ]]; then
  echo "[a2b] missing target idx file: ${TARGETS}" >&2
  echo "[a2b] run scripts/teacher/build_passk_distribution_and_bands.py first" >&2
  exit 1
fi

TOTAL_GPUS=$(( TP * NUM_SHARDS ))
VISIBLE=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if [[ "${TOTAL_GPUS}" -gt "${VISIBLE}" ]]; then
  echo "[a2b] need TP*NUM_SHARDS=${TOTAL_GPUS} GPUs but only ${VISIBLE} visible" >&2
  exit 1
fi

mkdir -p "${OUT}"
exec > >(tee -a "${OUT}/run_a2b_sampled.log") 2>&1

echo "[$(date -Is)] Stage A2b sampled teacher traces"
echo "[$(date -Is)] model=${MODEL}"
echo "[$(date -Is)] input=${INPUT_FILE}"
echo "[$(date -Is)] targets=${TARGETS}"
echo "[$(date -Is)] output=${OUT}"
echo "[$(date -Is)] temp=${TEMPERATURE} samples=${NUM_SAMPLES} tp=${TP} shards=${NUM_SHARDS}"

pids=()
for (( i=0; i<NUM_SHARDS; i++ )); do
  group=""
  for (( g=0; g<TP; g++ )); do
    gpu=$(( i * TP + g ))
    group="${group:+${group},}${gpu}"
  done
  shard_dir="$(printf '%s/shard-%05d-of-%05d' "${OUT}" "${i}" "${NUM_SHARDS}")"
  echo "[$(date -Is)] launching shard ${i}/${NUM_SHARDS} on GPUs ${group} -> ${shard_dir}"
  CUDA_VISIBLE_DEVICES="${group}" "${PY}" scripts/teacher/gen_gold_conditioned_traces.py \
    --model_name_or_path "${MODEL}" \
    --input_file "${INPUT_FILE}" \
    --database_dir "${DATABASE_DIR}" \
    --target_idx_file "${TARGETS}" \
    --output_dir "${shard_dir}" \
    --hint_strategy full_sql \
    --num_samples "${NUM_SAMPLES}" \
    --temperature "${TEMPERATURE}" \
    --top_p 1.0 \
    --limit "${LIMIT}" \
    --shard_index "${i}" \
    --num_shards "${NUM_SHARDS}" \
    --max_prompt_length "${MAX_PROMPT_LENGTH}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --max_tool_rounds "${MAX_TOOL_ROUNDS}" \
    --eval_timeout 60 \
    --vllm_tensor_parallel_size "${TP}" \
    --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
    --vllm_max_model_len "${VLLM_MAX_MODEL_LEN}" \
    --vllm_async_concurrency "${VLLM_ASYNC_CONCURRENCY}" \
    --overwrite > "${OUT}/shard${i}.log" 2>&1 &
  pids+=($!)
done

failed=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "[$(date -Is)] shard ${i} OK"
  else
    echo "[$(date -Is)] shard ${i} FAILED; see ${OUT}/shard${i}.log" >&2
    failed=1
  fi
done

if [[ "${failed}" -ne 0 ]]; then
  echo "[$(date -Is)] shard failure; skipping merge" >&2
  exit 1
fi

merge_dirs=()
for (( i=0; i<NUM_SHARDS; i++ )); do
  merge_dirs+=( "$(printf '%s/shard-%05d-of-%05d' "${OUT}" "${i}" "${NUM_SHARDS}")" )
done

echo "[$(date -Is)] merging shards -> ${OUT}/merged"
"${PY}" scripts/teacher/gen_gold_conditioned_traces.py \
  --merge_shard_dirs "${merge_dirs[@]}" \
  --merge_output_dir "${OUT}/merged" \
  --overwrite

echo "[$(date -Is)] complete"
echo "[$(date -Is)] traces=${OUT}/merged/teacher_traces.jsonl"
