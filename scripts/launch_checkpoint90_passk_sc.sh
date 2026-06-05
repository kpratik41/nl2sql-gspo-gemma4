#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

MODEL_PATH="${MODEL_PATH:-outputs/checkpoint-90}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
NUM_SHARDS="${NUM_SHARDS:-4}"
TP_SIZE="${TP_SIZE:-2}"
ASYNC_CONCURRENCY="${ASYNC_CONCURRENCY:-8}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-43000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.93}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-34000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8000}"
MAX_TOOL_ROUNDS="${MAX_TOOL_ROUNDS:-8}"
EVAL_WORKERS="${EVAL_WORKERS:-16}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-60}"
NUM_EXAMPLES="${NUM_EXAMPLES:--1}"

RAW_INPUT_FILE="${RAW_INPUT_FILE:-data/bird_dev_data/raw/bird_dev.json}"
INPUT_FILE="${INPUT_FILE:-${RAW_INPUT_FILE}}"
DATABASE_DIR="${DATABASE_DIR:-databases/dev_databases}"
DIFF_JSON_PATH="${DIFF_JSON_PATH:-data/bird_dev_data/raw/bird_dev.json}"
MEANINGS_FILE="${MEANINGS_FILE:-data/bird_dev_data/raw/column_meaning.json}"
FEWSHOT_TRAIN_FILE="${FEWSHOT_TRAIN_FILE:-data/bird_train_data/raw/train-6601.jsonl}"
FEWSHOT_TOP_N="${FEWSHOT_TOP_N:-5}"
EXAMPLE_NUM="${EXAMPLE_NUM:-3}"

SAMPLE_PLAN="${SAMPLE_PLAN:-default:16@1.2,default:1@0.0}"
NUM_GENERATIONS="${NUM_GENERATIONS:-17}"
PASSK_BASE_OUT="${PASSK_BASE_OUT:-outputs/passk/checkpoint90_dev_17gen_default_t1p2_t0}"
PASSK_MERGE_OUT="${PASSK_MERGE_OUT:-outputs/passk/checkpoint90_dev_17gen_default_t1p2_t0_merged}"
SC_OUT="${SC_OUT:-outputs/self_consistency/checkpoint90_dev_17gen_default_t1p2_t0}"
SHARDED_SC_OUT="${SHARDED_SC_OUT:-outputs/self_consistency/checkpoint90_dev_17gen_default_t1p2_t0_sharded}"

gpu_group_for_shard() {
  local shard="$1"
  local start=$((shard * TP_SIZE))
  local end=$((start + TP_SIZE - 1))
  local values=()
  local gpu
  for ((gpu = start; gpu <= end; gpu++)); do
    values+=("${gpu}")
  done
  local joined="${values[*]}"
  printf '%s' "${joined// /,}"
}

common_prompt_args=(
  --build_prompts_at_runtime
  --raw_input_file "${RAW_INPUT_FILE}"
  --input_file "${INPUT_FILE}"
  --database_dir "${DATABASE_DIR}"
  --diff_json_path "${DIFF_JSON_PATH}"
  --bird_mode dev
  --tool_mode default
  --prompt_template default
  --include_fewshots
  --fewshot_train_file "${FEWSHOT_TRAIN_FILE}"
  --fewshot_top_n "${FEWSHOT_TOP_N}"
  --include_stats
  --include_nullability
  --example_num "${EXAMPLE_NUM}"
)

common_generation_args=(
  --sample_plan "${SAMPLE_PLAN}"
  --num_generations "${NUM_GENERATIONS}"
  --max_prompt_length "${MAX_PROMPT_LENGTH}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --max_tool_rounds "${MAX_TOOL_ROUNDS}"
  --vllm_tensor_parallel_size "${TP_SIZE}"
  --vllm_max_model_len "${VLLM_MAX_MODEL_LEN}"
  --vllm_async_concurrency "${ASYNC_CONCURRENCY}"
  --vllm_gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}"
  --eval_workers "${EVAL_WORKERS}"
  --eval_timeout "${EVAL_TIMEOUT}"
  --incremental_writes
  --overwrite
)

run_passk_shard() {
  local shard="$1"
  local gpus="${2:-$(gpu_group_for_shard "${shard}")}"
  local log_file="passk_ckpt90_s${shard}.log"

  echo "[launcher] pass@k shard=${shard}/${NUM_SHARDS} gpus=${gpus} output=${PASSK_BASE_OUT}"
  CUDA_VISIBLE_DEVICES="${gpus}" PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}" \
    "${PYTHON_BIN}" scripts/run_passk_bird.py \
      --model_name_or_path "${MODEL_PATH}" \
      "${common_prompt_args[@]}" \
      "${common_generation_args[@]}" \
      --num_shards "${NUM_SHARDS}" \
      --shard_index "${shard}" \
      --output_dir "${PASSK_BASE_OUT}" \
      2>&1 | tee "${log_file}"
}

launch_passk() {
  local shard gpus screen_name
  for ((shard = 0; shard < NUM_SHARDS; shard++)); do
    gpus="$(gpu_group_for_shard "${shard}")"
    screen_name="passk_ckpt90_s${shard}"
    echo "[launcher] starting ${screen_name} on GPUs ${gpus}"
    screen -dmS "${screen_name}" bash -lc \
      "cd '${REPO_DIR}' && '${BASH_SOURCE[0]}' run-passk-shard '${shard}' '${gpus}'"
  done
}

passk_screen_name() {
  local shard="$1"
  printf 'passk_ckpt90_s%s' "${shard}"
}

wait_for_passk_shards() {
  local shard screen_name shard_dir
  for ((shard = 0; shard < NUM_SHARDS; shard++)); do
    screen_name="$(passk_screen_name "${shard}")"
    echo "[launcher] waiting for ${screen_name}"
    while screen -ls | grep -Eq "[0-9]+[.]${screen_name}[[:space:]]"; do
      sleep 60
    done
    shard_dir="${PASSK_BASE_OUT}/shard-$(printf '%05d' "${shard}")-of-$(printf '%05d' "${NUM_SHARDS}")"
    if [[ ! -s "${shard_dir}/passk_candidates_raw.jsonl" && ! -s "${shard_dir}/passk_candidates_raw.incremental.jsonl" ]]; then
      echo "[launcher] shard ${shard} did not produce candidates in ${shard_dir}" >&2
      return 1
    fi
  done
}

passk_shard_dirs() {
  local shard
  for ((shard = 0; shard < NUM_SHARDS; shard++)); do
    printf '%s\n' "${PASSK_BASE_OUT}/shard-$(printf '%05d' "${shard}")-of-$(printf '%05d' "${NUM_SHARDS}")"
  done
}

validate_passk_shards() {
  local shard_dirs=()
  local shard_dir
  while IFS= read -r shard_dir; do
    shard_dirs+=("${shard_dir}")
  done < <(passk_shard_dirs)

  PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}" "${PYTHON_BIN}" scripts/validate_passk_shard_counts.py \
    --input_file "${RAW_INPUT_FILE}" \
    --limit "${NUM_EXAMPLES}" \
    --num_shards "${NUM_SHARDS}" \
    --sample_plan "${SAMPLE_PLAN}" \
    --num_generations "${NUM_GENERATIONS}" \
    --top_p 1.0 \
    --shard_dirs "${shard_dirs[@]}"
}

merge_sharded_sc() {
  local shard_dirs=()
  local shard_dir
  while IFS= read -r shard_dir; do
    shard_dirs+=("${shard_dir}")
  done < <(passk_shard_dirs)

  validate_passk_shards

  PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}" "${PYTHON_BIN}" scripts/merge_passk_shards_to_self_consistency.py \
    --shard_dirs "${shard_dirs[@]}" \
    --output_dir "${SHARDED_SC_OUT}" \
    --database_dir "${DATABASE_DIR}" \
    --eval_timeout "${EVAL_TIMEOUT}" \
    --eval_workers "${EVAL_WORKERS}" \
    --overwrite
}

run_sharded_sc_pipeline() {
  echo "[launcher] starting sharded SC pipeline"
  echo "[launcher] this uses ${NUM_SHARDS} shards x TP=${TP_SIZE}; default GPU use is 0-$((NUM_SHARDS * TP_SIZE - 1))"
  launch_passk
  wait_for_passk_shards
  merge_sharded_sc
  echo "[launcher] sharded SC complete: ${SHARDED_SC_OUT}"
}

launch_sharded_sc() {
  screen -dmS sc_ckpt90_sharded bash -lc \
    "cd '${REPO_DIR}' && '${BASH_SOURCE[0]}' run-sharded-sc-pipeline"
}

merge_passk() {
  local shard_dirs=()
  local shard_dir
  while IFS= read -r shard_dir; do
    shard_dirs+=("${shard_dir}")
  done < <(passk_shard_dirs)

  validate_passk_shards

  PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}" "${PYTHON_BIN}" scripts/run_passk_bird.py \
    --merge_shard_dirs "${shard_dirs[@]}" \
    --merge_output_dir "${PASSK_MERGE_OUT}" \
    --output_dir "${PASSK_MERGE_OUT}"
}

run_sc() {
  local gpus="${1:-0,1}"
  echo "[launcher] self-consistency gpus=${gpus} output=${SC_OUT}"
  CUDA_VISIBLE_DEVICES="${gpus}" PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}" \
    "${PYTHON_BIN}" scripts/run_self_consistency_bird.py \
      --model_name_or_path "${MODEL_PATH}" \
      "${common_prompt_args[@]}" \
      "${common_generation_args[@]}" \
      --num_examples "${NUM_EXAMPLES}" \
      --output_dir "${SC_OUT}" \
      2>&1 | tee sc_ckpt90.log
}

launch_sc() {
  local gpus="${1:-0,1}"
  screen -dmS sc_ckpt90 bash -lc \
    "cd '${REPO_DIR}' && '${BASH_SOURCE[0]}' run-sc '${gpus}'"
}

status() {
  echo "== screens =="
  screen -ls || true
  echo
  echo "== gpu =="
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
  echo
  echo "== pass@k shard outputs =="
  find "${PASSK_BASE_OUT}" -maxdepth 2 -type f \( -name 'passk_candidates_raw.incremental.jsonl' -o -name 'passk_summary.json' -o -name 'passk_summary.md' \) -printf '%p %s\n' 2>/dev/null | sort || true
  echo
  echo "== self-consistency outputs =="
  find "${SC_OUT}" -maxdepth 1 -type f -printf '%p %s\n' 2>/dev/null | sort || true
}

usage() {
  cat <<EOF
Usage: $0 <command>

Commands:
  launch-passk              Launch ${NUM_SHARDS} pass@k shard screens, TP=${TP_SIZE} each.
  launch-sharded-sc         Single-command sharded SC: launch pass@k shards, wait, then compute SC.
  run-sharded-sc-pipeline   Direct: run the sharded SC pipeline in this shell.
  run-passk-shard I [GPUS]  Internal/direct: run one pass@k shard.
  merge-passk               Merge completed pass@k shard directories.
  merge-sharded-sc          Compute SC from completed pass@k shard candidates.
  validate-passk            Validate shard candidate counts before merge.
  launch-sc [GPUS]          Launch self-consistency in a screen; default GPUS=0,1.
  run-sc [GPUS]             Direct: run self-consistency in this shell.
  status                    Show screens, GPU usage, and key output files.

Important defaults:
  MODEL_PATH=${MODEL_PATH}
  SAMPLE_PLAN=${SAMPLE_PLAN}
  PASSK_BASE_OUT=${PASSK_BASE_OUT}
  PASSK_MERGE_OUT=${PASSK_MERGE_OUT}
  SC_OUT=${SC_OUT}
  SHARDED_SC_OUT=${SHARDED_SC_OUT}

Override any default by prefixing env vars, e.g.:
  NUM_SHARDS=2 TP_SIZE=2 $0 launch-passk
EOF
}

command="${1:-}"
case "${command}" in
  launch-passk)
    launch_passk
    ;;
  launch-sharded-sc)
    launch_sharded_sc
    ;;
  run-sharded-sc-pipeline)
    run_sharded_sc_pipeline
    ;;
  run-passk-shard)
    run_passk_shard "${2:?missing shard index}" "${3:-}"
    ;;
  merge-passk)
    merge_passk
    ;;
  merge-sharded-sc)
    merge_sharded_sc
    ;;
  validate-passk)
    validate_passk_shards
    ;;
  launch-sc)
    launch_sc "${2:-0,1}"
    ;;
  run-sc)
    run_sc "${2:-0,1}"
    ;;
  status)
    status
    ;;
  -h|--help|help|"")
    usage
    ;;
  *)
    echo "Unknown command: ${command}" >&2
    usage >&2
    exit 2
    ;;
esac
