#!/usr/bin/env bash
# Run 3 A5 batch 2: queue checkpoints 50, 60, 64 behind the running batch-1 eval.
#
# Waits for the batch-1 driver PID to exit, then waits for GPU memory to actually
# drain (vLLM teardown lags process exit), then runs the same parallel eval
# script with CKPTS="50 60 64" -> GPU pairs 0,1 / 2,3 / 4,5.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/ubuntu/sft-rl/nl2sql-gspo-gemma4}"
cd "${REPO_ROOT}"

WAIT_PID="${WAIT_PID:?set WAIT_PID to the batch-1 driver pid}"
CKPTS="${CKPTS:-50 60 64}"
DRAIN_MIB="${DRAIN_MIB:-5000}"
DRAIN_TIMEOUT_S="${DRAIN_TIMEOUT_S:-1800}"

mkdir -p logs
LOG="${LOG:-logs/run3_temp0_eval_batch2_queue.log}"
exec > >(tee -a "${LOG}") 2>&1

echo "[b2] queued at $(date -Is); waiting on batch-1 pid=${WAIT_PID}"

while kill -0 "${WAIT_PID}" 2>/dev/null; do
  sleep 30
done
echo "[b2] batch-1 driver exited at $(date -Is)"

# vLLM engines can hold GPU memory briefly after the driver returns.
deadline=$(( SECONDS + DRAIN_TIMEOUT_S ))
while (( SECONDS < deadline )); do
  busy="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
          | awk -v t="${DRAIN_MIB}" '$1 > t' | wc -l)"
  if [[ "${busy}" -eq 0 ]]; then
    echo "[b2] all GPUs drained at $(date -Is)"
    break
  fi
  echo "[b2] $(date -Is) waiting for ${busy} GPU(s) to drain"
  sleep 20
done

if [[ "${busy:-1}" -ne 0 ]]; then
  echo "[b2] WARNING: GPUs still busy after ${DRAIN_TIMEOUT_S}s; starting anyway" >&2
fi

echo "[b2] starting batch-2 evals ckpts=${CKPTS} at $(date -Is)"
echo "[b2] sft_output_dir=${SFT_OUTPUT_DIR:-<script default>}"
# Export rather than using an assignment prefix: an env prefix must be literal
# shell syntax, so `${VAR:+VAR="$VAR"}` is parsed as a command name, not an
# assignment, and the child never receives it.
if [[ -n "${SFT_OUTPUT_DIR:-}" ]]; then
  export SFT_OUTPUT_DIR
fi
export CKPTS
export QUEUE_LOG="${B2_QUEUE_LOG:-logs/run3_temp0_eval_parallel_b2.log}"
bash scripts/sft/run3_temp0_eval_parallel.sh
rc=$?
echo "[b2] batch-2 finished at $(date -Is) rc=${rc}"
exit "${rc}"
