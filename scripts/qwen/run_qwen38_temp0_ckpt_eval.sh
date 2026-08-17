#!/usr/bin/env bash
# Temp-0 BIRD-dev eval for one Qwen3.8-27B checkpoint: in-memory async vLLM
# engine, tp=2, 1 shard, on a single GPU pair. Thin wrapper around
# scripts/qwen/run_qwen38_eval_smoke.sh -- that script is the invocation
# verified on this box, so all the Qwen-specific handling (PATH for the
# FlashInfer ninja JIT, --no_prompt_rewrite for qwen-* inputs, the temp-0 tool
# loop guard, per-run compile caches) is inherited rather than duplicated.
#
#   RUN_TAG=rl1_ckpt10 MODEL_PATH=outputs/qwen-rl1/checkpoint-10 GPUS=6,7 \
#     bash scripts/qwen/run_qwen38_temp0_ckpt_eval.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

RUN_TAG="${RUN_TAG:?set RUN_TAG}"
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH}"
GPUS="${GPUS:?set GPUS, e.g. 2,3}"
# The smoke script reads both, and derives its per-run compile-cache paths from
# RUN_TAG; export explicitly rather than relying on the caller's environment.
export RUN_TAG MODEL_PATH

export CUDA_VISIBLE_DEVICES="${GPUS}"
export INPUT_FILE="${INPUT_FILE:-outputs/qwen-old-dev-schema-tool-unpatched.jsonl}"
export TOTAL="${TOTAL:--1}"
export TP="${TP:-2}"
export NUM_SHARDS="${NUM_SHARDS:-1}"
export TEMPERATURE="${TEMPERATURE:-0.0}"
export TOP_P="${TOP_P:-1.0}"
export TOP_K="${TOP_K:-20}"
export CONCURRENCY="${CONCURRENCY:-16}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-43000}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-34000}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8000}"
export MAX_TOOL_ROUNDS="${MAX_TOOL_ROUNDS:-8}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

export OUTPUT_DIR="${OUTPUT_DIR:-outputs/inference/qwen38_temp0_tp2/${RUN_TAG}}"
LOG="${LOG:-logs/qwen38_temp0_${RUN_TAG}.log}"
IDLE_MEMORY_MB="${IDLE_MEMORY_MB:-5000}"

mkdir -p "${OUTPUT_DIR}" logs

# Only this run's own GPU pair needs to be free; other pairs are running their
# own evals concurrently.
max_pair_memory_used_mb() {
  nvidia-smi --id="${GPUS}" --query-gpu=memory.used --format=csv,noheader,nounits \
    | awk 'BEGIN{m=0} {if ($1+0>m) m=$1+0} END{print m}'
}

{
  echo "[$(date -Is)] qwen3.8 temp0 eval tag=${RUN_TAG}"
  echo "[$(date -Is)] model=${MODEL_PATH}"
  echo "[$(date -Is)] input=${INPUT_FILE} rows=$(wc -l < "${INPUT_FILE}")"
  echo "[$(date -Is)] gpus=${GPUS} tp=${TP} shards=${NUM_SHARDS} concurrency=${CONCURRENCY}"
  echo "[$(date -Is)] output=${OUTPUT_DIR}"

  if [[ ! -e "${MODEL_PATH}/config.json" ]]; then
    echo "[$(date -Is)] FATAL: no config.json under ${MODEL_PATH}" >&2
    exit 1
  fi

  # Qwen3.8-27B is multimodal (config carries qwen3_5_vision), so the processor
  # load needs preprocessor_config.json even for a text-only workload. The RL
  # trainer saves only the text-side files, so a checkpoint dir is missing both
  # of these and the engine dies with
  #   OSError: Can't load image processor for '<ckpt>'
  # before generating anything. Backfill from the base snapshot.
  BASE_SNAPSHOT="${BASE_SNAPSHOT:-$(ls -d "${HOME}"/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/*/ | head -1)}"
  for cfg in preprocessor_config.json video_preprocessor_config.json; do
    if [[ ! -f "${MODEL_PATH}/${cfg}" ]]; then
      cp -L "${BASE_SNAPSHOT}${cfg}" "${MODEL_PATH}/${cfg}"
      echo "[$(date -Is)] backfilled ${MODEL_PATH}/${cfg} from base snapshot"
    fi
  done

  echo "[$(date -Is)] waiting for GPUs ${GPUS} to be idle; threshold=${IDLE_MEMORY_MB} MiB"
  while true; do
    used="$(max_pair_memory_used_mb)"
    if [[ "${used}" -lt "${IDLE_MEMORY_MB}" ]]; then
      echo "[$(date -Is)] GPUs idle enough; max_used=${used} MiB"
      break
    fi
    echo "[$(date -Is)] GPUs busy; max_used=${used} MiB"
    sleep 120
  done

  bash scripts/qwen/run_qwen38_eval_smoke.sh

  echo "[$(date -Is)] complete; summary=${OUTPUT_DIR}/eval_summary.md"
  grep -m1 "Overall EX Accuracy" "${OUTPUT_DIR}/eval_summary.md" || true
} 2>&1 | tee -a "${LOG}"
