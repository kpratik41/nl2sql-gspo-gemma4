#!/usr/bin/env bash
set -euo pipefail

# Waits for any in-flight sharded async inference to finish (generation + merge
# + eval), then runs the temp0 ckpt-50 job on all 8 GPUs.

cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="${PWD}"

CKPT_STEP="${CKPT_STEP:-50}"
QUEUE_LOG="${QUEUE_LOG:-${ROOT}/logs/queue_temp0_sft_rft3_ckpt${CKPT_STEP}.log}"
POLL_SECONDS="${POLL_SECONDS:-60}"

mkdir -p "$(dirname "${QUEUE_LOG}")"
exec > >(tee -a "${QUEUE_LOG}") 2>&1

echo "[$(date -Is)] queue armed for checkpoint-${CKPT_STEP}; poll=${POLL_SECONDS}s"

while true; do
  # Match only the parent processes of a sharded run, not this queue script.
  running="$(pgrep -fc "run_inference_bird_async_sharded\.py" || true)"
  if [[ "${running:-0}" -eq 0 ]]; then
    echo "[$(date -Is)] no sharded inference running; proceeding"
    break
  fi
  echo "[$(date -Is)] waiting; sharded inference processes=${running}"
  sleep "${POLL_SECONDS}"
done

echo "[$(date -Is)] launching checkpoint-${CKPT_STEP} temp0 run"
exec ./scripts/run_temp0_sft_rft3_ckpt_tp2_shards4.sh "${CKPT_STEP}"
