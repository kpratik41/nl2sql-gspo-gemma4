#!/usr/bin/env bash
# Stage A3a — hint-free self-trace trajectories over the self-trace band.
#
# A3a is generation only: it writes every sampled trajectory with its verified /
# kept flags.  The later A3 packing stage chooses the best verified trajectory
# per idx and converts traces into SFT rows.
#
# Defaults:
#   - mixed/sometimes band + capped all-correct band
#   - all-correct cap per DB = 12 when targets are rebuilt from pass@16
#   - hint_strategy = none, temperature = 0.7, num_samples = 2
#   - TP=2, NUM_SHARDS=4, i.e. all 8 GPUs on an 8xGPU node
#
# Examples:
#   DRY_RUN=1 bash scripts/teacher/run_a3a_selftrace.sh
#   TARGETS=outputs/teacher/target_idx_selftrace.txt bash scripts/teacher/run_a3a_selftrace.sh
#   PASSK_DIR=/path/to/passk/merged bash scripts/teacher/run_a3a_selftrace.sh
#   PASSK_DIR=/path/to/passk/merged BUILD_TARGETS_ONLY=1 bash scripts/teacher/run_a3a_selftrace.sh
#   TEMPERATURE=0.7 NUM_SAMPLES=2 TP=2 NUM_SHARDS=4 bash scripts/teacher/run_a3a_selftrace.sh
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
TP="${TP:-2}"
NUM_SHARDS="${NUM_SHARDS:-4}"
NUM_SAMPLES="${NUM_SAMPLES:-2}"
TEMPERATURE="${TEMPERATURE:-0.7}"
ALL_CORRECT_CAP_PER_DB="${ALL_CORRECT_CAP_PER_DB:-12}"
SEED="${SEED:-0}"
TARGET_DIR="${TARGET_DIR:-outputs/teacher/a3a_targets_cap${ALL_CORRECT_CAP_PER_DB}}"
OUT="${OUT:-outputs/teacher/a3a_selftrace_temp${TEMPERATURE}_samples${NUM_SAMPLES}_tp${TP}_shards${NUM_SHARDS}}"
LIMIT="${LIMIT:--1}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-34000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8000}"
MAX_TOOL_ROUNDS="${MAX_TOOL_ROUNDS:-8}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-43000}"
VLLM_ASYNC_CONCURRENCY="${VLLM_ASYNC_CONCURRENCY:-16}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.93}"
DRY_RUN="${DRY_RUN:-0}"
BUILD_TARGETS_ONLY="${BUILD_TARGETS_ONLY:-0}"
REBUILD_TARGETS="${REBUILD_TARGETS:-0}"

build_targets_from_passk() {
  if [[ -z "${PASSK_DIR:-}" ]]; then
    echo "[a3a] PASSK_DIR is required to build cap-${ALL_CORRECT_CAP_PER_DB} targets." >&2
    exit 1
  fi
  "${PY}" scripts/teacher/build_passk_distribution_and_bands.py \
    --passk-dir "${PASSK_DIR}" \
    --output-dir "${TARGET_DIR}" \
    --all-correct-cap-per-db "${ALL_CORRECT_CAP_PER_DB}" \
    --seed "${SEED}" \
    --overwrite
}

if [[ -z "${TARGETS:-}" ]]; then
  if [[ "${REBUILD_TARGETS}" == "1" ]]; then
    build_targets_from_passk
    TARGETS="${TARGET_DIR}/target_idx_selftrace.json"
  elif [[ -f "${TARGET_DIR}/target_idx_selftrace.json" ]]; then
    TARGETS="${TARGET_DIR}/target_idx_selftrace.json"
  elif [[ -f "${TARGET_DIR}/target_idx_selftrace.txt" ]]; then
    TARGETS="${TARGET_DIR}/target_idx_selftrace.txt"
  elif [[ -n "${PASSK_DIR:-}" ]]; then
    build_targets_from_passk
    TARGETS="${TARGET_DIR}/target_idx_selftrace.json"
  elif [[ -f "outputs/teacher/target_idx_selftrace.json" ]]; then
    TARGETS="outputs/teacher/target_idx_selftrace.json"
  elif [[ -f "outputs/teacher/target_idx_selftrace.txt" ]]; then
    TARGETS="outputs/teacher/target_idx_selftrace.txt"
  else
    echo "[a3a] missing self-trace target ids." >&2
    echo "[a3a] Set TARGETS=... or PASSK_DIR=... so cap-${ALL_CORRECT_CAP_PER_DB} targets can be built." >&2
    exit 1
  fi
fi

if [[ ! -f "${TARGETS}" ]]; then
  echo "[a3a] missing target idx file: ${TARGETS}" >&2
  exit 1
fi

TARGET_COUNT="$("${PY}" - "${TARGETS}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if path.suffix == ".json":
    print(len(json.loads(path.read_text(encoding="utf-8"))))
else:
    with path.open(encoding="utf-8") as handle:
        print(sum(1 for line in handle if line.strip()))
PY
)"

if [[ "${BUILD_TARGETS_ONLY}" == "1" ]]; then
  echo "[a3a] targets ready: ${TARGETS} count=${TARGET_COUNT}"
  echo "[a3a] target_dir=${TARGET_DIR} cap_per_db=${ALL_CORRECT_CAP_PER_DB} seed=${SEED}"
  exit 0
fi

TOTAL_GPUS=$(( TP * NUM_SHARDS ))
VISIBLE=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if [[ "${TOTAL_GPUS}" -gt "${VISIBLE}" ]]; then
  echo "[a3a] need TP*NUM_SHARDS=${TOTAL_GPUS} GPUs but only ${VISIBLE} visible" >&2
  exit 1
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[a3a] dry run"
  echo "[a3a] model=${MODEL}"
  echo "[a3a] input=${INPUT_FILE}"
  echo "[a3a] targets=${TARGETS} count=${TARGET_COUNT}"
  echo "[a3a] output=${OUT}"
  echo "[a3a] hint_strategy=none temp=${TEMPERATURE} samples=${NUM_SAMPLES} tp=${TP} shards=${NUM_SHARDS}"
  echo "[a3a] max_prompt_length=${MAX_PROMPT_LENGTH} max_new_tokens=${MAX_NEW_TOKENS} max_tool_rounds=${MAX_TOOL_ROUNDS}"
  echo "[a3a] vllm_max_model_len=${VLLM_MAX_MODEL_LEN} async_concurrency=${VLLM_ASYNC_CONCURRENCY} gpu_mem_util=${VLLM_GPU_MEMORY_UTILIZATION}"
  exit 0
fi

mkdir -p "${OUT}"
exec > >(tee -a "${OUT}/run_a3a_selftrace.log") 2>&1

echo "[$(date -Is)] Stage A3a hint-free self-trace trajectories"
echo "[$(date -Is)] model=${MODEL}"
echo "[$(date -Is)] input=${INPUT_FILE}"
echo "[$(date -Is)] targets=${TARGETS} count=${TARGET_COUNT}"
echo "[$(date -Is)] output=${OUT}"
echo "[$(date -Is)] hint_strategy=none temp=${TEMPERATURE} samples=${NUM_SAMPLES} tp=${TP} shards=${NUM_SHARDS}"

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
    --hint_strategy none \
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
