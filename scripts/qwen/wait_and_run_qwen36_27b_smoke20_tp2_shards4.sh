#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

LOG_DIR="${LOG_DIR:-logs/qwen36_27b_smoke20_tp2_shards4_temp0_openai_tool}"
mkdir -p "${LOG_DIR}"

echo "[qwen27-queue] queued at $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
echo "[qwen27-queue] waiting for GPUs to become idle"

while true; do
  busy_count="$(
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
      | awk '$1 > 1000 {count++} END {print count + 0}'
  )"
  if [[ "${busy_count}" == "0" ]]; then
    break
  fi
  echo "[qwen27-queue] GPUs still busy: ${busy_count}/8 at $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  sleep 60
done

echo "[qwen27-queue] GPUs idle; starting Qwen3.6-27B smoke"
exec scripts/qwen/run_qwen36_27b_olddev_smoke20_tp2_shards4.sh
