#!/usr/bin/env bash
# BIRD test-set inference, end to end.
#
#   bash scripts/bird_test/run_bird_test.sh
#
# Stages, each resumable -- a completed stage is skipped on re-run, so a crash
# in stage 4 does not repeat the hours of GPU time in stage 3:
#   1. BM25 few-shot retrieval from the train pool
#   2. Schema build over test_databases (stats + column comments)
#   3. Tool-format prompts with the Qwen system prompt
#   4. pass@16 generation, temperature 1.2, tool calls executed live
#   5. Self-consistency selection -> predict_test.json
#
# Requires MODEL_PATH to point at the model (a local dir or an HF repo id).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# CRITICAL: .venv/bin must be on PATH. FlashInfer's GDN kernel is JIT-compiled
# with ninja; invoking .venv/bin/python directly leaves .venv/bin off PATH and
# the JIT fails with "[Errno 2] No such file or directory: 'ninja'". Every
# generation then returns an error WHILE THE PROCESS STILL EXITS 0.
export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
export PYTHONPATH="src:.:scripts:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the model directory or HF repo id}"

TEST_JSON="${TEST_JSON:-data/bird_test_data/raw/test.json}"
TEST_DB_DIR="${TEST_DB_DIR:-databases/test_databases}"
TRAIN_POOL="${TRAIN_POOL:-data/bird_train_data/raw/train-6601.jsonl}"
FEW_SHOT_JSON="${FEW_SHOT_JSON:-data/bird_test_data/raw/test-few-shot.json}"
TOP_N="${TOP_N:-3}"

OUT_DIR="${OUT_DIR:-outputs/bird_test}"
LOG_DIR="${LOG_DIR:-logs/bird_test}"
SCHEMA_JSONL="${OUT_DIR}/test-schema.jsonl"
TOOL_JSONL="${OUT_DIR}/test-schema-tool.jsonl"
PASSK_DIR="${OUT_DIR}/passk16"
PREDICT_JSON="${PREDICT_JSON:-${OUT_DIR}/predict_test.json}"

NUM_GENERATIONS="${NUM_GENERATIONS:-16}"
TEMPERATURE="${TEMPERATURE:-1.2}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:-20}"
TP="${TP:-2}"
CONCURRENCY="${CONCURRENCY:-16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-43000}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-34000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8000}"
MAX_TOOL_ROUNDS="${MAX_TOOL_ROUNDS:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.93}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-60}"

# Shards = GPUs / tensor-parallel size. With the expected 2 GPUs and TP=2 this
# is a single shard; it scales automatically if more GPUs are allocated.
GPU_COUNT="${GPU_COUNT:-$(nvidia-smi --list-gpus | wc -l)}"
SHARDS="${SHARDS:-$(( GPU_COUNT / TP ))}"
[[ "${SHARDS}" -lt 1 ]] && SHARDS=1

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

echo "[bird-test] model=${MODEL_PATH}"
echo "[bird-test] gpus=${GPU_COUNT} tp=${TP} shards=${SHARDS} concurrency=${CONCURRENCY}/shard"
echo "[bird-test] pass@${NUM_GENERATIONS} temperature=${TEMPERATURE} max_tool_rounds=${MAX_TOOL_ROUNDS}"

# ---- 1. few-shot retrieval ------------------------------------------------
if [[ -s "${FEW_SHOT_JSON}" ]]; then
  echo "[bird-test] 1/5 few-shot: reusing ${FEW_SHOT_JSON}"
else
  echo "[bird-test] 1/5 few-shot retrieval"
  "${PYTHON_BIN}" scripts/bird_test/build_test_few_shots.py \
    --test-input "${TEST_JSON}" --train-input "${TRAIN_POOL}" \
    --output "${FEW_SHOT_JSON}" --top-n "${TOP_N}" 2>&1 | tee "${LOG_DIR}/1_few_shot.log"
fi

# ---- 2. schema build ------------------------------------------------------
# Stats and column comments on, matching the dev configuration this system was
# tuned against. column_meaning.json is required for the comments.
if [[ -s "${SCHEMA_JSONL}" ]]; then
  echo "[bird-test] 2/5 schema: reusing ${SCHEMA_JSONL}"
else
  echo "[bird-test] 2/5 schema build over ${TEST_DB_DIR}"
  "${PYTHON_BIN}" scripts/data_generation/schema_build.py \
    --split test --input-file "${FEW_SHOT_JSON}" \
    --n-examples -1 --output "${SCHEMA_JSONL}" \
    --messages-only --log-every 200 2>&1 | tee "${LOG_DIR}/2_schema.log"
fi

# ---- 3. tool-format prompts ----------------------------------------------
# --allow-missing-gold is mandatory here: every test.json row has "SQL": "".
if [[ -s "${TOOL_JSONL}" ]]; then
  echo "[bird-test] 3/5 prompts: reusing ${TOOL_JSONL}"
else
  echo "[bird-test] 3/5 tool-format prompts"
  "${PYTHON_BIN}" scripts/data_generation/build_tool_dataset.py \
    --input "${SCHEMA_JSONL}" --output "${TOOL_JSONL}" \
    --prompt-template default_qwen --allow-missing-gold \
    --log-every 200 2>&1 | tee "${LOG_DIR}/3_prompts.log"
fi

TOTAL="${TOTAL:-$(wc -l < "${TOOL_JSONL}")}"
echo "[bird-test] ${TOTAL} test questions"

# ---- 4. pass@16 generation ------------------------------------------------
if [[ -s "${PASSK_DIR}/merged/passk_candidates.jsonl" ]]; then
  echo "[bird-test] 4/5 generation: reusing ${PASSK_DIR}/merged"
else
  echo "[bird-test] 4/5 pass@${NUM_GENERATIONS} generation"
  pids=()
  for shard in $(seq 0 $((SHARDS - 1))); do
    first=$(( shard * TP ))
    gpus=$(seq -s, "${first}" $(( first + TP - 1 )))
    CUDA_VISIBLE_DEVICES="${gpus}" \
    VLLM_CACHE_ROOT="${PASSK_DIR}/shard-${shard}-vllm-cache" \
    TORCHINDUCTOR_CACHE_DIR="${PASSK_DIR}/shard-${shard}-inductor-cache" \
    "${PYTHON_BIN}" scripts/run_passk_bird_qwen_async.py \
      --model_name_or_path "${MODEL_PATH}" \
      --input_file "${TOOL_JSONL}" \
      --database_dir "${TEST_DB_DIR}" \
      --output_dir "${PASSK_DIR}" \
      --limit "${TOTAL}" --shard_index "${shard}" --num_shards "${SHARDS}" \
      --num_generations "${NUM_GENERATIONS}" \
      --temperature "${TEMPERATURE}" --top_p "${TOP_P}" --top_k "${TOP_K}" \
      --max_prompt_length "${MAX_PROMPT_LENGTH}" \
      --max_new_tokens "${MAX_NEW_TOKENS}" \
      --max_tool_rounds "${MAX_TOOL_ROUNDS}" \
      --eval_timeout "${EVAL_TIMEOUT}" --eval_workers 8 \
      --vllm_tensor_parallel_size "${TP}" \
      --vllm_gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
      --vllm_max_model_len "${MAX_MODEL_LEN}" \
      --vllm_async_concurrency "${CONCURRENCY}" \
      --tool_choice_policy required_first --empty_tool_retries 1 \
      --no_prompt_rewrite --skip_eval --overwrite \
      > "${LOG_DIR}/4_passk_shard${shard}.log" 2>&1 &
    pids+=("$!")
    echo "[bird-test]   shard ${shard} on GPUs ${gpus} -> ${LOG_DIR}/4_passk_shard${shard}.log"
  done

  failed=0
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { echo "[bird-test] shard ${i} FAILED (see ${LOG_DIR}/4_passk_shard${i}.log)"; failed=1; }
  done
  [[ "${failed}" -ne 0 ]] && { echo "[bird-test] generation failed; NOT merging"; exit 1; }

  merge_dirs=()
  for shard in $(seq 0 $((SHARDS - 1))); do
    merge_dirs+=( "$(printf '%s/shard-%05d-of-%05d' "${PASSK_DIR}" "${shard}" "${SHARDS}")" )
  done
  "${PYTHON_BIN}" scripts/run_passk_bird_qwen_async.py \
    --merge_shard_dirs "${merge_dirs[@]}" \
    --merge_output_dir "${PASSK_DIR}/merged" \
    --model_name_or_path "${MODEL_PATH}" --input_file "${TOOL_JSONL}" \
    --database_dir "${TEST_DB_DIR}" --limit "${TOTAL}" \
    --num_generations "${NUM_GENERATIONS}" --skip_eval --overwrite \
    2>&1 | tee "${LOG_DIR}/4_merge.log"
fi

# ---- 5. self-consistency selection ---------------------------------------
echo "[bird-test] 5/5 self-consistency selection"
"${PYTHON_BIN}" scripts/bird_test/select_and_export.py \
  --candidates "${PASSK_DIR}/merged/passk_candidates.jsonl" \
  --database-dir "${TEST_DB_DIR}" \
  --output "${PREDICT_JSON}" \
  --report "${OUT_DIR}/selection_report.jsonl" \
  --workers 16 --eval-timeout "${EVAL_TIMEOUT}" 2>&1 | tee "${LOG_DIR}/5_select.log"

echo "[bird-test] done -> ${PREDICT_JSON}"
