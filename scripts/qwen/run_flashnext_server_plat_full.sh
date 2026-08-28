#!/usr/bin/env bash
# Qwen3.8-Flash-Next on arcwise_plat_full, served by vLLM in Docker at tp=8.
#
# WHY DOCKER AND NOT A VENV
#
# The vLLM recipe for this model requires vLLM 0.29.0+ and states plainly that
# "PyPI installation is not supported for this recipe". As of 2026-08-28 the
# newest vLLM on PyPI is 0.28.0 and the newest nightly wheel is
# 0.28.1rc1.dev43 -- 0.29 is not published in any installable form, so there is
# no wheel to put in a venv. The purpose-built image is the only supported path:
#
#   vllm/vllm-openai:qwen38-flash-next
#
# This also isolates it completely from the working vllm 0.19.1 environment that
# produces the Qwen3.8-27B results, which a shared venv would not.
#
# CONSEQUENCE: THE SERVER PATH, NOT THE IN-PROCESS ENGINE
#
# Every 27B result on this branch came from scripts/run_inference_bird_qwen_async.py,
# which holds the engine in-process and parses tool calls itself. Docker forces the
# OpenAI-server path instead, so vLLM must parse tool calls out of the model's raw
# text via --tool-call-parser. That parser is load-bearing: if it does not match
# what the model emits, tool calls silently fail to extract, the model appears to
# make none, and accuracy collapses with no error raised anywhere.
#
# The recipe specifies qwen3_xml for this model. The older 27B server scripts on
# this branch used qwen3_coder, and qwen38_eval_plan.md asks for both to be smoke
# tested before a full run. Hence SMOKE below: run 20 examples first and confirm
# tool_call_count_total > 0 before spending the full 498.
#
#   SMOKE=20 bash scripts/qwen/run_flashnext_server_plat_full.sh   # sanity check
#   bash scripts/qwen/run_flashnext_server_plat_full.sh            # full 498
#
# Data is reused unchanged from the 27B runs. That was verified, not assumed:
# Flash-Next ships a chat_template.jinja byte-identical to Qwen3.8-27B's, the same
# Qwen2Tokenizer with the same 33 added tokens and the same eos/pad, and the same
# <tool_call> dialect -- so outputs/qwen-arcwise_plat_full-schema-tool.jsonl needs
# no regeneration, and --no_prompt_rewrite applies for the same reason as there.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
export PYTHONPATH="src:.:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

IMAGE="${IMAGE:-vllm/vllm-openai:qwen38-flash-next}"
CONTAINER="${CONTAINER:-flashnext-vllm}"
PORT="${PORT:-8100}"
SERVED_NAME="${SERVED_NAME:-flash-next}"

# Prefer the NVMe copy; 360GB off EBS dominates startup. Instance store does not
# survive a stop/start, so the fallback to the persistent copy is deliberate.
NVME_MODEL="${NVME_MODEL:-/opt/dlami/nvme/models/Qwen3.8-Flash-Next}"
EBS_MODEL="${EBS_MODEL:-/home/ubuntu/models/Qwen3.8-Flash-Next}"
if [[ -z "${MODEL_PATH:-}" && -f "${NVME_MODEL}/config.json" ]]; then
  MODEL_PATH="${NVME_MODEL}"
fi
MODEL_PATH="${MODEL_PATH:-${EBS_MODEL}}"
MODEL_TAG="${MODEL_TAG:-Qwen3.8-Flash-Next}"

DATASET="${DATASET:-arcwise_plat_full}"
INPUT_FILE="${INPUT_FILE:-outputs/qwen-arcwise_plat_full-schema-tool.jsonl}"
DIFF_JSON="${DIFF_JSON:-data/revisql/raw/arcwise_plat_full.json}"
DATABASE_DIR="${DATABASE_DIR:-databases/dev_databases}"

TP="${TP:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.96}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_xml}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
CONCURRENCY="${CONCURRENCY:-16}"
MAX_TOOL_ROUNDS="${MAX_TOOL_ROUNDS:-8}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-900}"
SMOKE="${SMOKE:--1}"
SMOKE_FIRST="${SMOKE_FIRST:-1}"   # gate the full run on a small sample first
SMOKE_N="${SMOKE_N:-20}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-3600}"

# Same thinking-aware budgets as the 27B suite, and for the same reason: at 8000
# the first thinking sweep truncated reasoning mid-thought and scored it as an
# answer. max_model_len must cover prompt + generation for one sequence.
ENABLE_THINKING="${ENABLE_THINKING:-0}"
THINK_ARGS=()
RUN_SUFFIX=""
if [[ "${ENABLE_THINKING}" == "1" ]]; then
  THINK_ARGS+=(--enable_thinking)
  RUN_SUFFIX="_think"
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16000}"
  MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
else
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8000}"
  MAX_MODEL_LEN="${MAX_MODEL_LEN:-43000}"
fi
if [[ "${PRESERVE_THINKING:-0}" == "1" ]]; then
  THINK_ARGS+=(--preserve_thinking); RUN_SUFFIX="${RUN_SUFFIX}_preserve"
fi

TS="$(date +%Y%m%d_%H%M%S)"
CTX_TAG="ctx$((MAX_MODEL_LEN/1000))k_o$((MAX_NEW_TOKENS/1000))k_r${MAX_TOOL_ROUNDS}"
[[ "${SMOKE}" != "-1" ]] && CTX_TAG="smoke${SMOKE}_${CTX_TAG}"
OUT_DIR="outputs/inference/${DATASET}/${MODEL_TAG}/vllm_server_tp${TP}_${CTX_TAG}_${TOOL_CALL_PARSER}_temp0${RUN_SUFFIX}_${TS}"
LOG="logs/flashnext_server_${DATASET}${RUN_SUFFIX}_${TS}.log"
SERVER_LOG="logs/flashnext_server_${TS}.container.log"
mkdir -p logs

# ---- preflight ------------------------------------------------------------
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "FATAL: no config.json under ${MODEL_PATH}" >&2; exit 1
fi
shard_count="$(ls "${MODEL_PATH}"/*.safetensors 2>/dev/null | wc -l)"
incomplete="$(find "${MODEL_PATH}" -name '*.incomplete' 2>/dev/null | wc -l)"
if [[ "${shard_count}" -ne 131 || "${incomplete}" -ne 0 ]]; then
  echo "FATAL: ${MODEL_PATH} has ${shard_count}/131 shards, ${incomplete} incomplete" >&2; exit 1
fi
if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "FATAL: image ${IMAGE} not present; docker pull ${IMAGE}" >&2; exit 1
fi
mkdir -p "${OUT_DIR}"
# ---------------------------------------------------------------------------

cleanup() { docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true; }
trap cleanup EXIT

{
  echo "[$(date -Is)] flash-next server eval"
  echo "[$(date -Is)] image=${IMAGE} model=${MODEL_PATH}"
  echo "[$(date -Is)] tp=${TP} util=${GPU_MEMORY_UTILIZATION} max_model_len=${MAX_MODEL_LEN} max_new_tokens=${MAX_NEW_TOKENS}"
  echo "[$(date -Is)] tool_call_parser=${TOOL_CALL_PARSER} reasoning_parser=${REASONING_PARSER} thinking=${ENABLE_THINKING}"
  echo "[$(date -Is)] dataset=${DATASET} rows=$(wc -l < "${INPUT_FILE}") smoke=${SMOKE}"
  echo "[$(date -Is)] output=${OUT_DIR}"

  echo "[$(date -Is)] waiting for GPUs idle"
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk 'BEGIN{m=0}{if($1+0>m)m=$1+0}END{print m}')
    [[ "${used}" -lt 5000 ]] && { echo "[$(date -Is)] GPUs idle (${used} MiB)"; break; }
    echo "[$(date -Is)] GPUs busy; max_used=${used} MiB"; sleep 60
  done

  cleanup
  echo "[$(date -Is)] starting server container"
  # The model directory is mounted read-only: nothing in the eval should ever
  # write into the weights.
  docker run -d --name "${CONTAINER}" \
    --gpus all --ipc=host \
    -v "${MODEL_PATH}:/model:ro" \
    -p "${PORT}:8000" \
    "${IMAGE}" \
    --model /model \
    --served-model-name "${SERVED_NAME}" \
    --tensor-parallel-size "${TP}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser "${TOOL_CALL_PARSER}" \
    --reasoning-parser "${REASONING_PARSER}" \
    >/dev/null

  echo "[$(date -Is)] waiting for /v1/models (timeout ${SERVER_READY_TIMEOUT}s; 360GB load is slow)"
  deadline=$((SECONDS + SERVER_READY_TIMEOUT))
  until curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; do
    if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER}"; then
      echo "[$(date -Is)] FATAL: container exited during startup; last lines:" >&2
      docker logs --tail 40 "${CONTAINER}" >&2 || true
      exit 1
    fi
    if (( SECONDS > deadline )); then
      echo "[$(date -Is)] FATAL: server not ready within ${SERVER_READY_TIMEOUT}s" >&2
      docker logs --tail 40 "${CONTAINER}" >&2 || true
      exit 1
    fi
    sleep 10
  done
  echo "[$(date -Is)] server ready on :${PORT}"
  docker logs "${CONTAINER}" > "${SERVER_LOG}" 2>&1 || true

  # One server, two evals. Loading 360GB is far too expensive to pay twice, so
  # the smoke gate and the full run share this container.
  run_eval() {
    local n="$1" outdir="$2"
    mkdir -p "${outdir}"
    .venv/bin/python scripts/run_inference_bird_qwen_server.py \
      --server_url "http://127.0.0.1:${PORT}/v1" \
      --model "${SERVED_NAME}" \
      --input_file "${INPUT_FILE}" \
      --database_dir "${DATABASE_DIR}" \
      --diff_json_path "${DIFF_JSON}" \
      --output_dir "${outdir}" \
      --num_examples "${n}" \
      --temperature 0.0 \
      --top_p 1.0 \
      --max_new_tokens "${MAX_NEW_TOKENS}" \
      --max_tool_rounds "${MAX_TOOL_ROUNDS}" \
      --tool_choice_policy required_first \
      --empty_tool_retries 1 \
      --request_timeout "${REQUEST_TIMEOUT}" \
      --eval_timeout 60 \
      --eval_workers 16 \
      --concurrency "${CONCURRENCY}" \
      --no_prompt_rewrite \
      "${THINK_ARGS[@]}" \
      --overwrite
  }

  # Zero extracted tool calls is the signature of a parser mismatch, and it
  # surfaces as a believable low accuracy rather than an error. Gate on it.
  tool_calls_in() {
    .venv/bin/python -c "
import json, sys, pathlib
p = pathlib.Path(sys.argv[1]) / 'eval_summary.json'
print(json.loads(p.read_text()).get('generation_stats', {}).get('tool_call_count_total', 0) if p.exists() else 0)
" "$1"
  }

  rc=0
  if [[ "${SMOKE_FIRST}" == "1" && "${SMOKE}" == "-1" ]]; then
    echo "[$(date -Is)] smoke gate: ${SMOKE_N} examples, parser=${TOOL_CALL_PARSER}"
    set +e; run_eval "${SMOKE_N}" "${OUT_DIR}/smoke"; set -e
    n_tools="$(tool_calls_in "${OUT_DIR}/smoke")"
    echo "[$(date -Is)] smoke extracted ${n_tools} tool calls"
    if [[ "${n_tools}" -eq 0 ]]; then
      echo "[$(date -Is)] FATAL: parser ${TOOL_CALL_PARSER} extracted no tool calls." >&2
      echo "  Not spending the full ${DATASET} on this. Re-run with the other parser:" >&2
      echo "    TOOL_CALL_PARSER=qwen3_coder bash scripts/qwen/run_flashnext_server_plat_full.sh" >&2
      exit 3
    fi
    echo "[$(date -Is)] gate passed; running the full set"
  fi

  set +e
  run_eval "${SMOKE}" "${OUT_DIR}"
  rc=$?
  set -e

  echo "[$(date -Is)] complete rc=${rc}; summary=${OUT_DIR}/eval_summary.md"
} 2>&1 | tee -a "${LOG}"
