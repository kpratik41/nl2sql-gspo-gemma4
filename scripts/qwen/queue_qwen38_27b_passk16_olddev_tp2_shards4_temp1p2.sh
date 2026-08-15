#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

MODEL_PATH="${MODEL_PATH:-/home/ec2-user/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3p8-27b-passk}"
VLLM_BIN="${VLLM_BIN:-/home/ec2-user/miniconda3/envs/nl2sql312/bin/vllm}"
PYTHON_BIN="${PYTHON_BIN:-/home/ec2-user/miniconda3/envs/nl2sql312/bin/python}"

INPUT_FILE="${INPUT_FILE:-outputs/old-dev-schema-tool-unpatched.jsonl}"
DATABASE_DIR="${DATABASE_DIR:-databases/dev_databases}"
DIFF_JSON_PATH="${DIFF_JSON_PATH:-data/bird_dev_data/raw/bird_dev_unpatched.json}"
BASE_OUT="${BASE_OUT:-outputs/passk/qwen3p8_27b_olddev_schema_tool_passk16_temp1p2_tp2_shards4_c16}"
LOG_DIR="${LOG_DIR:-logs/qwen38_27b_passk16_olddev_tp2_shards4_temp1p2_c16}"

TOTAL="${TOTAL:-1534}"
TP="${TP:-2}"
SHARDS="${SHARDS:-4}"
TEMPERATURE="${TEMPERATURE:-1.2}"
NUM_GENERATIONS="${NUM_GENERATIONS:-16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
CONCURRENCY_PER_SHARD="${CONCURRENCY_PER_SHARD:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8000}"
MAX_TOOL_ROUNDS="${MAX_TOOL_ROUNDS:-8}"
TOOL_CHOICE_POLICY="${TOOL_CHOICE_POLICY:-required_first}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"
EMPTY_TOOL_RETRIES="${EMPTY_TOOL_RETRIES:-1}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-900}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-60}"
EVAL_WORKERS="${EVAL_WORKERS:-16}"
IDLE_MEMORY_MB="${IDLE_MEMORY_MB:-5000}"

PORTS=(8040 8041 8042 8043)
GPU_SETS=("0,1" "2,3" "4,5" "6,7")
VLLM_INTERNAL_PORTS=(44100 44200 44300 44400)

mkdir -p "${BASE_OUT}" "${LOG_DIR}"
exec > >(tee -a "${BASE_OUT}/queue_and_run.log") 2>&1

server_pids=()
client_pids=()

cleanup() {
  for pid in "${client_pids[@]:-}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  for pid in "${server_pids[@]:-}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT

max_gpu_memory_used_mb() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk 'BEGIN{m=0} {if ($1+0>m) m=$1+0} END{print m}'
}

wait_for_idle_gpus() {
  echo "[$(date -Is)] waiting for GPUs to be idle; threshold=${IDLE_MEMORY_MB} MiB"
  while true; do
    used="$(max_gpu_memory_used_mb)"
    if [[ "${used}" -lt "${IDLE_MEMORY_MB}" ]]; then
      echo "[$(date -Is)] GPUs idle enough; max_used=${used} MiB"
      break
    fi
    echo "[$(date -Is)] GPUs busy; max_used=${used} MiB"
    sleep 120
  done
}

wait_for_server() {
  local port="$1"
  local deadline=$((SECONDS + 1800))
  while true; do
    if curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      break
    fi
    if (( SECONDS > deadline )); then
      echo "[qwen38-passk] server on port ${port} did not become ready in time" >&2
      return 1
    fi
    sleep 5
  done
}

wait_for_idle_gpus

echo "[$(date -Is)] starting Qwen3.8 pass@${NUM_GENERATIONS}"
echo "[$(date -Is)] input=${INPUT_FILE}"
echo "[$(date -Is)] output=${BASE_OUT}"
echo "[$(date -Is)] tp=${TP} shards=${SHARDS} temp=${TEMPERATURE} concurrency_per_shard=${CONCURRENCY_PER_SHARD}"

for shard in $(seq 0 $((SHARDS - 1))); do
  port="${PORTS[$shard]}"
  gpu_set="${GPU_SETS[$shard]}"
  vllm_internal_port="${VLLM_INTERNAL_PORTS[$shard]}"
  server_log="${LOG_DIR}/server_shard${shard}_port${port}.log"
  echo "[$(date -Is)] server shard=${shard} port=${port} gpus=${gpu_set} log=${server_log}"
  CUDA_VISIBLE_DEVICES="${gpu_set}" VLLM_HOST_IP=127.0.0.1 VLLM_PORT="${vllm_internal_port}" "${VLLM_BIN}" serve "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --port "${port}" \
    --tensor-parallel-size "${TP}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --enable-prefix-caching \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser "${TOOL_CALL_PARSER}" \
    --generation-config auto \
    >"${server_log}" 2>&1 &
  server_pids+=("$!")
  sleep 3
done

for shard in $(seq 0 $((SHARDS - 1))); do
  wait_for_server "${PORTS[$shard]}"
  echo "[$(date -Is)] server ready shard=${shard} port=${PORTS[$shard]}"
done

for shard in $(seq 0 $((SHARDS - 1))); do
  port="${PORTS[$shard]}"
  shard_log="${LOG_DIR}/client_shard${shard}.log"
  echo "[$(date -Is)] client shard=${shard}/${SHARDS} port=${port} log=${shard_log}"
  PYTHONPATH="${PYTHONPATH:-src:.}" "${PYTHON_BIN}" scripts/run_passk_bird_qwen_server.py \
    --server_url "http://127.0.0.1:${port}/v1" \
    --model "${SERVED_MODEL_NAME}" \
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
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --max_tool_rounds "${MAX_TOOL_ROUNDS}" \
    --tool_choice_policy "${TOOL_CHOICE_POLICY}" \
    --empty_tool_retries "${EMPTY_TOOL_RETRIES}" \
    --request_timeout "${REQUEST_TIMEOUT}" \
    --eval_timeout "${EVAL_TIMEOUT}" \
    --eval_workers "${EVAL_WORKERS}" \
    --concurrency "${CONCURRENCY_PER_SHARD}" \
    --overwrite \
    >"${shard_log}" 2>&1 &
  client_pids+=("$!")
done

failed=0
for i in "${!client_pids[@]}"; do
  if wait "${client_pids[$i]}"; then
    echo "[$(date -Is)] shard ${i} finished OK"
  else
    echo "[$(date -Is)] shard ${i} FAILED"
    failed=1
  fi
done

if [[ "${failed}" -ne 0 ]]; then
  echo "[$(date -Is)] one or more shards failed; skipping merge"
  exit 1
fi

merge_dirs=()
for shard in $(seq 0 $((SHARDS - 1))); do
  merge_dirs+=( "$(printf '%s/shard-%05d-of-%05d' "${BASE_OUT}" "${shard}" "${SHARDS}")" )
done

echo "[$(date -Is)] all shards done, merging"
"${PYTHON_BIN}" scripts/run_passk_bird_qwen_server.py \
  --merge_shard_dirs "${merge_dirs[@]}" \
  --merge_output_dir "${BASE_OUT}/merged" \
  --input_file "${INPUT_FILE}" \
  --database_dir "${DATABASE_DIR}" \
  --diff_json_path "${DIFF_JSON_PATH}" \
  --limit "${TOTAL}" \
  --num_generations "${NUM_GENERATIONS}" \
  --overwrite

echo "[$(date -Is)] complete"
echo "[$(date -Is)] merged summary=${BASE_OUT}/merged/passk_summary.md"
