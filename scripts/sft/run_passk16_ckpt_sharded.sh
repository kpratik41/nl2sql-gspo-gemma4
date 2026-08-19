#!/usr/bin/env bash
# pass@16 on a single SFT checkpoint, sharded across GPU groups, then merged.
#
# Results are written INSIDE the checkpoint directory. Each shard gets its own
# GPU group of size TP, starting at GPU_OFFSET.
#
#   CKPT_DIR=outputs/sft/.../checkpoint-25 TP=2 NUM_SHARDS=3 LIMIT=500 \
#     bash scripts/sft/run_passk16_ckpt_sharded.sh
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/ubuntu/sft-rl/nl2sql-gspo-gemma4}"
cd "${REPO_ROOT}"

CKPT_DIR="${CKPT_DIR:?set CKPT_DIR to the checkpoint directory}"
INPUT_FILE="${INPUT_FILE:-outputs/old-dev-schema-tool-unpatched.jsonl}"
DATABASE_DIR="${DATABASE_DIR:-databases/dev_databases}"
DIFF_JSON_PATH="${DIFF_JSON_PATH:-data/bird_dev_data/raw/dev_20251106.json}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
TP="${TP:-2}"
NUM_SHARDS="${NUM_SHARDS:-3}"
GPU_OFFSET="${GPU_OFFSET:-0}"
LIMIT="${LIMIT:-500}"
NUM_GENERATIONS="${NUM_GENERATIONS:-16}"
TEMPERATURE="${TEMPERATURE:-1.2}"
TOP_P="${TOP_P:-1.0}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-30000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8000}"
MAX_TOOL_ROUNDS="${MAX_TOOL_ROUNDS:-8}"
EVAL_WORKERS="${EVAL_WORKERS:-16}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-60}"
VLLM_ASYNC_CONCURRENCY="${VLLM_ASYNC_CONCURRENCY:-16}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.93}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-43000}"

TAG="${TAG:-passk${NUM_GENERATIONS}_olddev_first${LIMIT}_temp1p2_tp${TP}_shards${NUM_SHARDS}}"
BASE_OUT="${BASE_OUT:-${CKPT_DIR}/${TAG}}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

if [[ ! -f "${CKPT_DIR}/processor_config.json" ]]; then
  install -m 0644 gemma-4-31b-it-local/processor_config.json "${CKPT_DIR}/processor_config.json"
  echo "[passk] patched processor_config.json into ${CKPT_DIR}"
fi

mkdir -p "${BASE_OUT}"
LOG="${BASE_OUT}/run_passk16.log"
exec > >(tee -a "${LOG}") 2>&1

echo "[passk] started at $(date -Is)"
echo "[passk] ckpt=${CKPT_DIR}"
echo "[passk] input=${INPUT_FILE} limit=${LIMIT}"
echo "[passk] k=${NUM_GENERATIONS} temp=${TEMPERATURE} tp=${TP} shards=${NUM_SHARDS} gpu_offset=${GPU_OFFSET}"
echo "[passk] output=${BASE_OUT}"

pids=()
for (( i=0; i<NUM_SHARDS; i++ )); do
  group=""
  for (( g=0; g<TP; g++ )); do
    gpu=$(( GPU_OFFSET + i * TP + g ))
    group="${group:+${group},}${gpu}"
  done
  echo "[passk] launching shard ${i}/${NUM_SHARDS} on GPUs ${group}"
  CUDA_VISIBLE_DEVICES="${group}" "${PYTHON_BIN}" -u scripts/run_passk_bird.py \
    --model_name_or_path "${CKPT_DIR}" \
    --input_file "${INPUT_FILE}" \
    --database_dir "${DATABASE_DIR}" \
    --diff_json_path "${DIFF_JSON_PATH}" \
    --output_dir "${BASE_OUT}" \
    --limit "${LIMIT}" \
    --shard_index "${i}" \
    --num_shards "${NUM_SHARDS}" \
    --num_generations "${NUM_GENERATIONS}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --max_prompt_length "${MAX_PROMPT_LENGTH}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --max_tool_rounds "${MAX_TOOL_ROUNDS}" \
    --eval_timeout "${EVAL_TIMEOUT}" \
    --eval_workers "${EVAL_WORKERS}" \
    --vllm_tensor_parallel_size "${TP}" \
    --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
    --vllm_max_model_len "${VLLM_MAX_MODEL_LEN}" \
    --vllm_async_concurrency "${VLLM_ASYNC_CONCURRENCY}" \
    --overwrite > "${BASE_OUT}/shard${i}.log" 2>&1 &
  pids+=($!)
done

failed=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "[passk] shard ${i} OK at $(date -Is)"
  else
    echo "[passk] shard ${i} FAILED; see ${BASE_OUT}/shard${i}.log" >&2
    failed=1
  fi
done

if [[ "${failed}" -ne 0 ]]; then
  echo "[passk] one or more shards failed; skipping merge" >&2
  exit 1
fi

echo "[passk] all shards done, merging"
merge_dirs=()
for (( i=0; i<NUM_SHARDS; i++ )); do
  merge_dirs+=( "$(printf '%s/shard-%05d-of-%05d' "${BASE_OUT}" "${i}" "${NUM_SHARDS}")" )
done

"${PYTHON_BIN}" -u scripts/run_passk_bird.py \
  --merge_shard_dirs "${merge_dirs[@]}" \
  --merge_output_dir "${BASE_OUT}/merged" \
  --input_file "${INPUT_FILE}" \
  --database_dir "${DATABASE_DIR}" \
  --diff_json_path "${DIFF_JSON_PATH}" \
  --limit "${LIMIT}" \
  --num_generations "${NUM_GENERATIONS}" \
  --overwrite

echo "[passk] complete at $(date -Is)"
echo "[passk] merged summary=${BASE_OUT}/merged/passk_summary.md"
