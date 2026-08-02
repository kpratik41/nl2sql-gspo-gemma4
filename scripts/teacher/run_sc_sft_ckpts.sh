#!/usr/bin/env bash
# Self-consistency over the SFT checkpoints' existing pass@16 candidates.
# CPU-only: reuses artifacts on disk, so it does not touch the GPUs and can run
# alongside RL training. Checkpoints are processed strictly one after another.
set -euo pipefail
cd /home/ubuntu/nl2sql-gspo-gemma4

export PYTHONUNBUFFERED=1
export PYTHONPATH=".:src:scripts:${PYTHONPATH:-}"

PY="${PY:-.venv/bin/python}"
ROOT=outputs/sft/gemma4_31b_rft_sft
PASSK_T="${ROOT}/checkpoint-{ckpt}/passk16_olddev_schema_tool_unpatched_temp1p2_tp2_shards4/merged"
TEMP0_T="${ROOT}/checkpoint-{ckpt}/temp0_olddev_schema_tool_unpatched_vllm_async_tp1"
WORKERS="${WORKERS:-8}"
LOG=outputs/sft/sc_sft.log
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

for CK in ${CKPTS:-20 70}; do
  echo "[$(date -Is)] ===== self-consistency for checkpoint-${CK} ====="
  "$PY" scripts/evaluate_self_consistency.py \
    --passk-dir-template "$PASSK_T" \
    --checkpoint-dir-template "$TEMP0_T" \
    --database-dir databases/dev_databases \
    --ckpts "$CK" \
    --num-generations 16 \
    --workers "$WORKERS" \
    --eval-timeout 60 \
    --output-dir "${ROOT}/checkpoint-${CK}/self_consistency" \
    --overwrite
  echo "[$(date -Is)] checkpoint-${CK} done"
done
echo "[$(date -Is)] all self-consistency runs complete"
