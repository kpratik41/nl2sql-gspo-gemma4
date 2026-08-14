#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

MODEL_PATH="${MODEL_PATH:-/home/ec2-user/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3p8-27b}"
VLLM_BIN="${VLLM_BIN:-/home/ec2-user/miniconda3/envs/nl2sql312/bin/vllm}"
PYTHON_BIN="${PYTHON_BIN:-/home/ec2-user/miniconda3/envs/nl2sql312/bin/python}"

INPUT_FILE="${INPUT_FILE:-outputs/old-dev-schema-tool-unpatched.jsonl}"
DATABASE_DIR="${DATABASE_DIR:-databases/dev_databases}"
DIFF_JSON_PATH="${DIFF_JSON_PATH:-data/bird_dev_data/raw/bird_dev_unpatched.json}"
RUN_ROOT="${RUN_ROOT:-outputs/inference/dev/old-dev-schema-tool-unpatched/Qwen3.8-27B/smoke20_tp2_shards4_temp0_openai_tool_qwen3_coder}"
LOG_DIR="${LOG_DIR:-logs/qwen38_27b_smoke20_tp2_shards4_temp0_openai_tool_qwen3_coder}"

TOTAL="${TOTAL:-20}"
TP="${TP:-2}"
SHARDS="${SHARDS:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
CONCURRENCY_PER_SHARD="${CONCURRENCY_PER_SHARD:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8000}"
MAX_TOOL_ROUNDS="${MAX_TOOL_ROUNDS:-8}"
TOOL_CHOICE_POLICY="${TOOL_CHOICE_POLICY:-required_first}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"
EMPTY_TOOL_RETRIES="${EMPTY_TOOL_RETRIES:-1}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-900}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-60}"
EVAL_WORKERS="${EVAL_WORKERS:-8}"
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-0}"

PORTS=(8030 8031 8032 8033)
GPU_SETS=("0,1" "2,3" "4,5" "6,7")
VLLM_INTERNAL_PORTS=(43100 43200 43300 43400)

mkdir -p "${RUN_ROOT}" "${LOG_DIR}"

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

wait_for_server() {
  local port="$1"
  local deadline=$((SECONDS + 1800))
  while true; do
    if curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      break
    fi
    if (( SECONDS > deadline )); then
      echo "[qwen38] server on port ${port} did not become ready in time" >&2
      return 1
    fi
    sleep 5
  done
}

echo "[qwen38] starting ${SHARDS} TP=${TP} servers parser=${TOOL_CALL_PARSER}"
for shard in $(seq 0 $((SHARDS - 1))); do
  port="${PORTS[$shard]}"
  gpu_set="${GPU_SETS[$shard]}"
  vllm_internal_port="${VLLM_INTERNAL_PORTS[$shard]}"
  server_log="${LOG_DIR}/server_shard${shard}_port${port}.log"
  extra_vllm_args=()
  if [[ "${LANGUAGE_MODEL_ONLY}" == "1" ]]; then
    extra_vllm_args+=(--language-model-only)
  fi
  echo "[qwen38] server shard=${shard} port=${port} vllm_port=${vllm_internal_port} gpus=${gpu_set} log=${server_log}"
  CUDA_VISIBLE_DEVICES="${gpu_set}" VLLM_HOST_IP=127.0.0.1 VLLM_PORT="${vllm_internal_port}" "${VLLM_BIN}" serve "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --port "${port}" \
    --tensor-parallel-size "${TP}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser "${TOOL_CALL_PARSER}" \
    --generation-config auto \
    "${extra_vllm_args[@]}" \
    >"${server_log}" 2>&1 &
  server_pids+=("$!")
  sleep 3
done

for shard in $(seq 0 $((SHARDS - 1))); do
  wait_for_server "${PORTS[$shard]}"
  echo "[qwen38] server ready shard=${shard} port=${PORTS[$shard]}"
done

echo "[qwen38] starting inference shards"
for shard in $(seq 0 $((SHARDS - 1))); do
  start=$((shard * TOTAL / SHARDS))
  end=$(((shard + 1) * TOTAL / SHARDS))
  port="${PORTS[$shard]}"
  output_dir="${RUN_ROOT}/shard_${shard}"
  client_log="${LOG_DIR}/client_shard${shard}_${start}_${end}.log"
  echo "[qwen38] client shard=${shard} rows=[${start},${end}) port=${port} log=${client_log}"
  PYTHONPATH="${PYTHONPATH:-src:.}" "${PYTHON_BIN}" scripts/run_inference_bird_qwen_server.py \
    --server_url "http://127.0.0.1:${port}/v1" \
    --model "${SERVED_MODEL_NAME}" \
    --input_file "${INPUT_FILE}" \
    --database_dir "${DATABASE_DIR}" \
    --diff_json_path "${DIFF_JSON_PATH}" \
    --output_dir "${output_dir}" \
    --start_index "${start}" \
    --end_index "${end}" \
    --num_examples -1 \
    --temperature 0.0 \
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
    >"${client_log}" 2>&1 &
  client_pids+=("$!")
done

status=0
for pid in "${client_pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

echo "[qwen38] inference shards finished with status=${status}"
exit "${status}"
