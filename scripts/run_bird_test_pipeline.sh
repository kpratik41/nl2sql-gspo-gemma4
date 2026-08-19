#!/usr/bin/env bash
# End-to-end BIRD test-set pipeline: raw test.json -> submitted predictions.
#
# One command runs every stage. Each stage is skipped when its output already
# exists, so rerunning after a failure continues where it stopped rather than
# repeating finished work. Generation itself also resumes per example, so an
# interrupted pass@k picks up at the candidate it reached.
#
#   bash scripts/run_bird_test_pipeline.sh
#
# Inputs expected from the BIRD team, under the repository root:
#   data/bird_test_data/raw/test.json            questions, "SQL" empty
#   data/bird_test_data/raw/column_meaning.json  column descriptions (REQUIRED)
#   databases/test_databases/                    the test SQLite databases
#
# Final artifact:
#   ${RUN_ROOT}/self_consistency/predict_test.json
#       official BIRD format, one entry per question id:
#       "<SQL>\t----- bird -----\t<db_id>"
#
# Every stage appends to ${RUN_ROOT}/pipeline.log.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the model directory or Hugging Face id}"

SPLIT="${SPLIT:-test}"
RUN_ROOT="${RUN_ROOT:-outputs/bird_${SPLIT}_pipeline}"
DATABASE_DIR="${DATABASE_DIR:-databases/${SPLIT}_databases}"
DIFF_JSON_PATH="${DIFF_JSON_PATH:-data/bird_${SPLIT}_data/raw/${SPLIT}.json}"
PREDICTIONS_FILENAME="${PREDICTIONS_FILENAME:-predict_${SPLIT}.json}"

SCHEMA_JSONL="${SCHEMA_JSONL:-${RUN_ROOT}/bird_${SPLIT}-schema.jsonl}"
TOOL_JSONL="${TOOL_JSONL:-${RUN_ROOT}/bird_${SPLIT}-schema-tool.jsonl}"
PASSK_DIR="${PASSK_DIR:-${RUN_ROOT}/passk}"
SC_DIR="${SC_DIR:-${RUN_ROOT}/self_consistency}"

NUM_EXAMPLES="${NUM_EXAMPLES:--1}"
NUM_GENERATIONS="${NUM_GENERATIONS:-16}"
# 1.2 is the sampling temperature every validated pass@16 run used (the output
# directories are named temp1p2). run_passk_bird.py's own default is 0.8, which
# does NOT match those runs -- always set it explicitly here.
TEMPERATURE="${TEMPERATURE:-1.2}"
TOP_P="${TOP_P:-1.0}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-44000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8000}"
MAX_TOOL_ROUNDS="${MAX_TOOL_ROUNDS:-8}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-2}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-53000}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.96}"
VLLM_ASYNC_CONCURRENCY="${VLLM_ASYNC_CONCURRENCY:-16}"
# 30s matches the official BIRD evaluator (--meta_time_out default).
EVAL_TIMEOUT="${EVAL_TIMEOUT:-30}"
# Tool calls during generation keep the more generous budget; this is the
# model exploring the database, not the graded query.
TOOL_TIMEOUT="${TOOL_TIMEOUT:-60}"
EVAL_WORKERS="${EVAL_WORKERS:-16}"
FALLBACK_SQL="${FALLBACK_SQL:-SELECT 1}"
# When a rollout exhausts MAX_TOOL_ROUNDS with a tool call still pending, give it
# one non-tool turn to commit to SQL instead of returning nothing. Set to 0 to
# restore the old cut-off-at-the-cap behaviour.
FORCE_FINALIZE="${FORCE_FINALIZE:-1}"

mkdir -p "${RUN_ROOT}"
LOG="${RUN_ROOT}/pipeline.log"
exec > >(tee -a "${LOG}") 2>&1

say() { echo "[$(date -Is)] [pipeline] $*"; }

say "repo=${REPO_ROOT}"
say "model=${MODEL_PATH}"
say "split=${SPLIT} run_root=${RUN_ROOT}"
say "num_generations=${NUM_GENERATIONS} temperature=${TEMPERATURE} tp=${VLLM_TENSOR_PARALLEL_SIZE}"

# ---------------------------------------------------------------- preflight --
for required in \
  "data/bird_${SPLIT}_data/raw/${SPLIT}.json" \
  "data/bird_${SPLIT}_data/raw/column_meaning.json" \
  "${DATABASE_DIR}"
do
  if [[ ! -e "${required}" ]]; then
    say "FATAL: missing required input ${required}"
    exit 1
  fi
done
say "preflight OK: inputs present"

# ------------------------------------------------------------- 0. few-shots --
# Demonstrations are retrieved from the TRAIN pool, never from the split being
# evaluated: a question must not be able to retrieve itself, or a sibling from
# the same database, as its own worked example. schema_build reads them from the
# few_shot_examples field, so this has to run before it -- pointing schema_build
# at a raw test.json silently produces prompts with no few-shots at all, which
# do not match the prompt format the model was validated on.
FEWSHOT_JSON="${FEWSHOT_JSON:-data/bird_${SPLIT}_data/raw/${SPLIT}-few-shot.json}"
FEWSHOT_REFERENCE="${FEWSHOT_REFERENCE:-data/bird_train_data/raw/train-6601.jsonl}"
# 3, not 5: every validated dev prompt carries exactly three demonstrations, and
# the prompt the model was tuned on is the one it should be evaluated on.
FEWSHOT_TOP_N="${FEWSHOT_TOP_N:-3}"

if [[ -s "${FEWSHOT_JSON}" ]]; then
  say "stage 0/4 few-shots: reusing ${FEWSHOT_JSON}"
else
  say "stage 0/4 few-shots: retrieving top-${FEWSHOT_TOP_N} from ${FEWSHOT_REFERENCE}"
  "${PYTHON_BIN}" scripts/data_generation/few_shot_bm25.py \
    --reference-input "${FEWSHOT_REFERENCE}" \
    --dev-input "data/bird_${SPLIT}_data/raw/${SPLIT}.json" \
    --dev-output "${FEWSHOT_JSON}" \
    --top-n "${FEWSHOT_TOP_N}"
  say "stage 0/4 done"
fi

# Refuse to continue if any demonstration came from a database under evaluation.
"${PYTHON_BIN}" - "${FEWSHOT_JSON}" <<'PY'
import json
import sys

rows = json.load(open(sys.argv[1], encoding="utf-8"))
split_dbs = {row.get("db_id", "") for row in rows}
leaked = [
    row.get("question_id")
    for row in rows
    if any(ex.get("db_id", "") in split_dbs for ex in row.get("few_shot_examples", []))
]
if leaked:
    raise SystemExit(
        f"[fewshot] FAILED: {len(leaked)} row(s) retrieved a demonstration from an "
        f"evaluated database; first={leaked[:10]}"
    )
missing = [r.get("question_id") for r in rows if not r.get("few_shot_examples")]
if missing:
    raise SystemExit(f"[fewshot] FAILED: {len(missing)} row(s) have no few-shots")
print(f"[fewshot] OK: {len(rows)} rows, no demonstration from an evaluated database")
PY

# ------------------------------------------------------- 1. schema-built rows --
if [[ -s "${SCHEMA_JSONL}" ]]; then
  say "stage 1/4 schema rows: reusing $(wc -l < "${SCHEMA_JSONL}") rows in ${SCHEMA_JSONL}"
else
  say "stage 1/4 schema rows: building ${SCHEMA_JSONL} from ${FEWSHOT_JSON}"
  "${PYTHON_BIN}" scripts/data_generation/schema_build.py \
    --split "${SPLIT}" \
    --input-file "${FEWSHOT_JSON}" \
    --n-examples "${NUM_EXAMPLES}" \
    --output "${SCHEMA_JSONL}"
  say "stage 1/4 done: $(wc -l < "${SCHEMA_JSONL}") rows"
fi

# ---------------------------------------------------------- 2. tool-aware rows --
if [[ -s "${TOOL_JSONL}" ]]; then
  say "stage 2/4 tool rows: reusing $(wc -l < "${TOOL_JSONL}") rows in ${TOOL_JSONL}"
else
  say "stage 2/4 tool rows: building ${TOOL_JSONL}"
  "${PYTHON_BIN}" scripts/data_generation/build_tool_dataset.py \
    --input "${SCHEMA_JSONL}" \
    --output "${TOOL_JSONL}"
  say "stage 2/4 done: $(wc -l < "${TOOL_JSONL}") rows"
fi

# ------------------------------------------------------------- 3. pass@k sampling --
# Tensor parallelism is fixed at 2: a 31B model in bf16 is ~62 GB of weights, so
# it does not fit on one 80 GB card at this context length. Remaining GPUs are
# used for data parallelism instead -- N independent tp=2 engines, one shard of
# the questions each, which beats a single tp=8 engine (less cross-GPU traffic,
# N generations in flight). 8 GPUs -> 4 shards, 2 GPUs -> 1 shard.
PASSK_EXTRA_ARGS=()
if [[ "${FORCE_FINALIZE}" != "1" ]]; then
  PASSK_EXTRA_ARGS+=(--no_force_finalize)
fi

run_passk_shard() {
  local shard_index="$1" num_shards="$2" gpus="$3"
  CUDA_VISIBLE_DEVICES="${gpus}" "${PYTHON_BIN}" scripts/run_passk_bird.py \
    "${PASSK_EXTRA_ARGS[@]}" \
    --model_name_or_path "${MODEL_PATH}" \
    --input_file "${TOOL_JSONL}" \
    --database_dir "${DATABASE_DIR}" \
    --diff_json_path "${DIFF_JSON_PATH}" \
    --output_dir "${PASSK_DIR}" \
    --shard_index "${shard_index}" \
    --num_shards "${num_shards}" \
    --limit "${NUM_EXAMPLES}" \
    --num_generations "${NUM_GENERATIONS}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --max_prompt_length "${MAX_PROMPT_LENGTH}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --max_tool_rounds "${MAX_TOOL_ROUNDS}" \
    --eval_timeout "${EVAL_TIMEOUT}" \
    --tool_timeout "${TOOL_TIMEOUT}" \
    --eval_workers "${EVAL_WORKERS}" \
    --vllm_tensor_parallel_size "${VLLM_TENSOR_PARALLEL_SIZE}" \
    --vllm_max_model_len "${VLLM_MAX_MODEL_LEN}" \
    --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
    --vllm_async_concurrency "${VLLM_ASYNC_CONCURRENCY}" \
    --fallback_sql "${FALLBACK_SQL}" \
    --overwrite
}

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  DETECTED_GPUS="$(awk -F',' '{print NF}' <<<"${CUDA_VISIBLE_DEVICES}")"
else
  DETECTED_GPUS="$(nvidia-smi -L 2>/dev/null | wc -l)"
fi
NUM_GPUS="${NUM_GPUS:-${DETECTED_GPUS}}"
NUM_SHARDS="${NUM_SHARDS:-$(( NUM_GPUS / VLLM_TENSOR_PARALLEL_SIZE ))}"

if [[ "${NUM_SHARDS}" -lt 1 ]]; then
  say "FATAL: need at least ${VLLM_TENSOR_PARALLEL_SIZE} GPUs for tensor_parallel_size=${VLLM_TENSOR_PARALLEL_SIZE}; detected ${NUM_GPUS}"
  exit 1
fi
say "gpus=${NUM_GPUS} tp=${VLLM_TENSOR_PARALLEL_SIZE} -> shards=${NUM_SHARDS}"
if [[ $(( NUM_GPUS % VLLM_TENSOR_PARALLEL_SIZE )) -ne 0 ]]; then
  say "note: ${NUM_GPUS} GPUs is not a multiple of tp=${VLLM_TENSOR_PARALLEL_SIZE}; $(( NUM_GPUS % VLLM_TENSOR_PARALLEL_SIZE )) will be idle"
fi

if [[ "${NUM_SHARDS}" -eq 1 ]]; then
  PASSK_CANDIDATES="${PASSK_DIR}/passk_candidates.jsonl"
else
  PASSK_CANDIDATES="${PASSK_DIR}/merged/passk_candidates.jsonl"
fi

if [[ -s "${PASSK_CANDIDATES}" ]]; then
  say "stage 3/4 pass@${NUM_GENERATIONS}: reusing candidates in ${PASSK_CANDIDATES}"
else
  say "stage 3/4 pass@${NUM_GENERATIONS}: sampling into ${PASSK_DIR} across ${NUM_SHARDS} shard(s)"
  say "           (an interrupted shard resumes from its generation_progress.jsonl)"

  if [[ "${NUM_SHARDS}" -eq 1 ]]; then
    run_passk_shard 0 1 "$(seq -s, 0 $(( VLLM_TENSOR_PARALLEL_SIZE - 1 )))"
  else
    shard_pids=()
    for (( shard = 0; shard < NUM_SHARDS; shard++ )); do
      first_gpu=$(( shard * VLLM_TENSOR_PARALLEL_SIZE ))
      gpus="$(seq -s, "${first_gpu}" $(( first_gpu + VLLM_TENSOR_PARALLEL_SIZE - 1 )))"
      say "  shard ${shard}/${NUM_SHARDS} -> GPUs ${gpus}"
      run_passk_shard "${shard}" "${NUM_SHARDS}" "${gpus}" \
        > "${PASSK_DIR}-shard${shard}.log" 2>&1 &
      shard_pids+=("$!")
    done

    # Wait on each shard individually so one failure is reported with its index
    # rather than collapsing into a single opaque non-zero status.
    failed=0
    for shard in "${!shard_pids[@]}"; do
      if wait "${shard_pids[$shard]}"; then
        say "  shard ${shard} finished"
      else
        say "  shard ${shard} FAILED; see ${PASSK_DIR}-shard${shard}.log"
        failed=1
      fi
    done
    if [[ "${failed}" -ne 0 ]]; then
      say "FATAL: at least one pass@k shard failed; rerun this script to resume"
      exit 1
    fi

    say "merging ${NUM_SHARDS} shards"
    "${PYTHON_BIN}" scripts/run_passk_bird.py \
      --model_name_or_path "${MODEL_PATH}" \
      --input_file "${TOOL_JSONL}" \
      --database_dir "${DATABASE_DIR}" \
      --diff_json_path "${DIFF_JSON_PATH}" \
      --output_dir "${PASSK_DIR}" \
      --limit "${NUM_EXAMPLES}" \
      --num_generations "${NUM_GENERATIONS}" \
      --merge_shard_dirs "${PASSK_DIR}"/shard-*-of-* \
      --merge_output_dir "${PASSK_DIR}/merged" \
      --overwrite
  fi
  say "stage 3/4 done"
fi

# ------------------------------------------------------ 4. self-consistency vote --
PREDICTIONS_PATH="${SC_DIR}/${PREDICTIONS_FILENAME}"
if [[ -s "${PREDICTIONS_PATH}" ]]; then
  say "stage 4/4 self-consistency: reusing ${PREDICTIONS_PATH}"
else
  say "stage 4/4 self-consistency: voting into ${SC_DIR}"
  "${PYTHON_BIN}" scripts/run_self_consistency_bird.py \
    --passk_candidates_path "${PASSK_CANDIDATES}" \
    --input_file "${TOOL_JSONL}" \
    --database_dir "${DATABASE_DIR}" \
    --output_dir "${SC_DIR}" \
    --predictions_filename "${PREDICTIONS_FILENAME}" \
    --eval_timeout "${EVAL_TIMEOUT}" \
    --tool_timeout "${TOOL_TIMEOUT}" \
    --eval_workers "${EVAL_WORKERS}" \
    --fallback_sql "${FALLBACK_SQL}" \
    --overwrite
  say "stage 4/4 done"
fi

# ----------------------------------------------------------------- final check --
# The submitted file is scored by question id, so verify one prediction per input
# row before declaring success rather than letting a short file ship silently.
"${PYTHON_BIN}" - "${TOOL_JSONL}" "${PREDICTIONS_PATH}" <<'PY'
import json
import sys

rows_path, predictions_path = sys.argv[1], sys.argv[2]

expected = set()
with open(rows_path, encoding="utf-8") as handle:
    for position, line in enumerate(handle):
        line = line.strip()
        if line:
            expected.add(int(json.loads(line).get("source_idx", position)))

with open(predictions_path, encoding="utf-8") as handle:
    predictions = json.load(handle)

got = {int(key) for key in predictions}
missing = sorted(expected - got)
empty = sorted(int(k) for k, v in predictions.items() if not v.split("\t")[0].strip())

print(f"[verify] questions={len(expected)} predictions={len(got)}")
if missing:
    raise SystemExit(f"[verify] FAILED: {len(missing)} missing id(s); first={missing[:10]}")
if empty:
    print(f"[verify] WARNING: {len(empty)} prediction(s) have empty SQL; first={empty[:10]}")
malformed = [k for k, v in predictions.items() if "\t----- bird -----\t" not in v]
if malformed:
    raise SystemExit(f"[verify] FAILED: {len(malformed)} entries not in BIRD format")
print("[verify] OK: every question id present and correctly formatted")
PY

say "COMPLETE"
say "submit this file: ${PREDICTIONS_PATH}"
