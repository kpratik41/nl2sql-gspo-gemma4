#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

LOG="${LOG:-gemma4_sharded_sc_queue.log}"

wait_for_screen() {
  local name="$1"
  while screen -ls | grep -Eq "[0-9]+[.]${name}[[:space:]]"; do
    sleep 60
  done
}

{
  echo "[$(date -Is)] waiting for checkpoint-90 sharded SC/pass@k screens to finish"
  for name in sc_ckpt90_sharded passk_ckpt90_s0 passk_ckpt90_s1 passk_ckpt90_s2 passk_ckpt90_s3; do
    echo "[$(date -Is)] waiting for ${name}"
    wait_for_screen "${name}"
  done

  echo "[$(date -Is)] launching Gemma 4 sharded SC pipeline"
  MODEL_PATH="google/gemma-4-31B-it" \
  PASSK_BASE_OUT="outputs/passk/gemma4-31b-it_dev_17gen_default_t1p2_t0" \
  PASSK_MERGE_OUT="outputs/passk/gemma4-31b-it_dev_17gen_default_t1p2_t0_merged" \
  SHARDED_SC_OUT="outputs/self_consistency/gemma4-31b-it_dev_17gen_default_t1p2_t0_sharded" \
  scripts/launch_checkpoint90_passk_sc.sh run-sharded-sc-pipeline

  echo "[$(date -Is)] Gemma 4 sharded SC pipeline finished"
} 2>&1 | tee -a "${LOG}"
