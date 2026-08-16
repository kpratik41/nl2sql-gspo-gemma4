#!/usr/bin/env bash
# Qwen3.8-27B pass@16 on BIRD dev, temp 1.2, in-memory AsyncLLMEngine.
#
# tp=1 with 8 shards: one engine per H200, no cross-GPU communication at all.
# At 27B bf16 the weights are ~54 GiB, so a 143 GiB H200 at 0.93 utilization
# leaves ~79 GiB per shard for KV cache -- ample for 16 concurrent sequences at
# max_model_len 43000.
#
# The tp=2 / 4-shard variant of this run lives next to it in
# run_qwen38_27b_passk16_async_engine_tp2_shards4_temp1p2.sh. This one doubles
# the number of independent engines, which is the better shape for pass@k:
# every shard is embarrassingly parallel and there is no TP all-reduce on the
# critical path.
#
# Progress is queryable while it runs:
#   bash scripts/qwen/passk_progress.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# CRITICAL: .venv/bin must be on PATH. FlashInfer's GDN prefill kernel is
# JIT-compiled with ninja; invoking .venv/bin/python directly leaves .venv/bin
# off PATH, the JIT dies with "[Errno 2] No such file or directory: 'ninja'",
# and every sample returns generation_error -- while the process still exits 0.
export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
export PYTHONPATH="src:.:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# Suppress byte-identical repeat tool calls within a rollout. Eval opts in; RL
# does not, until this is shown to help. See tool_loop_guard.py for why.
export NL2SQL_TOOL_LOOP_GUARD="${NL2SQL_TOOL_LOOP_GUARD:-1}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-$(ls -d "${HOME}"/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/*/ | head -1)}"

INPUT_FILE="${INPUT_FILE:-outputs/qwen-old-dev-schema-tool-unpatched.jsonl}"
DATABASE_DIR="${DATABASE_DIR:-databases/dev_databases}"
DIFF_JSON_PATH="${DIFF_JSON_PATH:-data/bird_dev_data/raw/bird_dev_unpatched.json}"

RUN_TAG="${RUN_TAG:-qwen3p8_27b_passk16_temp1p2_tp1_shards8_c16}"
BASE_OUT="${BASE_OUT:-outputs/passk/${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-logs/${RUN_TAG}}"

TOTAL="${TOTAL:-1534}"
TP="${TP:-1}"
SHARDS="${SHARDS:-8}"
NUM_GENERATIONS="${NUM_GENERATIONS:-16}"
TEMPERATURE="${TEMPERATURE:-1.2}"
CONCURRENCY_PER_SHARD="${CONCURRENCY_PER_SHARD:-16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-43000}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-34000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8000}"
MAX_TOOL_ROUNDS="${MAX_TOOL_ROUNDS:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.93}"

GPU_SETS=("0" "1" "2" "3" "4" "5" "6" "7")

# The qwen-* data files already carry the Qwen tool syntax, so the runtime
# rewrite must be off for them -- applying it strips the XML examples and
# re-inserts "do not print the function call", undoing the point of the file.
NO_PROMPT_REWRITE_ARGS=()
if [[ "${NO_PROMPT_REWRITE:-auto}" == "auto" ]]; then
  case "${INPUT_FILE}" in
    *qwen-*) NO_PROMPT_REWRITE="1" ;;
    *)       NO_PROMPT_REWRITE="0" ;;
  esac
fi
[[ "${NO_PROMPT_REWRITE}" == "1" ]] && NO_PROMPT_REWRITE_ARGS=(--no_prompt_rewrite)

mkdir -p "${BASE_OUT}" "${LOG_DIR}"
echo "${BASE_OUT}" > "${LOG_DIR}/.base_out"
echo "$((TOTAL * NUM_GENERATIONS))" > "${LOG_DIR}/.expected_total"

{
  echo "[$(date -Is)] pass@${NUM_GENERATIONS} temp=${TEMPERATURE} tp=${TP} shards=${SHARDS}"
  echo "[$(date -Is)] model=${MODEL_PATH}"
  echo "[$(date -Is)] input=${INPUT_FILE}  no_prompt_rewrite=${NO_PROMPT_REWRITE}"
  echo "[$(date -Is)] output=${BASE_OUT}"
  echo "[$(date -Is)] logs=${LOG_DIR}   progress: bash scripts/qwen/passk_progress.sh"
} | tee -a "${BASE_OUT}/run.log"

pids=()
for shard in $(seq 0 $((SHARDS - 1))); do
  shard_log="${LOG_DIR}/shard${shard}.log"
  echo "[$(date -Is)] launching shard=${shard}/${SHARDS} gpu=${GPU_SETS[$shard]} log=${shard_log}" | tee -a "${BASE_OUT}/run.log"
  CUDA_VISIBLE_DEVICES="${GPU_SETS[$shard]}" \
  VLLM_CACHE_ROOT="${BASE_OUT}/shard-${shard}-vllm-cache" \
  TORCHINDUCTOR_CACHE_DIR="${BASE_OUT}/shard-${shard}-inductor-cache" \
  "${PYTHON_BIN}" scripts/run_passk_bird_qwen_async.py \
    --model_name_or_path "${MODEL_PATH}" \
    --input_file "${INPUT_FILE}" \
    --database_dir "${DATABASE_DIR}" \
    --diff_json_path "${DIFF_JSON_PATH}" \
    --output_dir "${BASE_OUT}" \
    --limit "${TOTAL}" \
    --shard_index "${shard}" \
    --num_shards "${SHARDS}" \
    --num_generations "${NUM_GENERATIONS}" \
    --temperature "${TEMPERATURE}" \
    --top_p 1.0 \
    --top_k 20 \
    --max_prompt_length "${MAX_PROMPT_LENGTH}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --max_tool_rounds "${MAX_TOOL_ROUNDS}" \
    --eval_timeout 60 \
    --eval_workers 8 \
    --vllm_tensor_parallel_size "${TP}" \
    --vllm_gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
    --vllm_max_model_len "${MAX_MODEL_LEN}" \
    --vllm_async_concurrency "${CONCURRENCY_PER_SHARD}" \
    --tool_choice_policy required_first \
    --empty_tool_retries 1 \
    "${NO_PROMPT_REWRITE_ARGS[@]}" \
    --overwrite \
    >"${shard_log}" 2>&1 &
  pids+=("$!")
done

failed=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "[$(date -Is)] shard ${i} finished OK" | tee -a "${BASE_OUT}/run.log"
  else
    echo "[$(date -Is)] shard ${i} FAILED (see ${LOG_DIR}/shard${i}.log)" | tee -a "${BASE_OUT}/run.log"
    failed=1
  fi
done

if [[ "${failed}" -ne 0 ]]; then
  echo "[$(date -Is)] one or more shards failed; NOT merging" | tee -a "${BASE_OUT}/run.log"
  exit 1
fi

merge_dirs=()
for shard in $(seq 0 $((SHARDS - 1))); do
  merge_dirs+=( "$(printf '%s/shard-%05d-of-%05d' "${BASE_OUT}" "${shard}" "${SHARDS}")" )
done

echo "[$(date -Is)] all shards done, merging" | tee -a "${BASE_OUT}/run.log"
"${PYTHON_BIN}" scripts/run_passk_bird_qwen_async.py \
  --merge_shard_dirs "${merge_dirs[@]}" \
  --merge_output_dir "${BASE_OUT}/merged" \
  --model_name_or_path "${MODEL_PATH}" \
  --input_file "${INPUT_FILE}" \
  --database_dir "${DATABASE_DIR}" \
  --diff_json_path "${DIFF_JSON_PATH}" \
  --limit "${TOTAL}" \
  --num_generations "${NUM_GENERATIONS}" \
  --overwrite 2>&1 | tee -a "${BASE_OUT}/run.log"

echo "[$(date -Is)] complete; summary=${BASE_OUT}/merged/passk_summary.md" | tee -a "${BASE_OUT}/run.log"
