#!/usr/bin/env bash
# Wait for the ckpt-20 dev pass@16 to finish and free the GPUs, then run the
# identical pass@16 for ckpt-70.
set -euo pipefail
cd /home/ubuntu/nl2sql-gspo-gemma4

CK20=outputs/sft/gemma4_31b_rft_sft/checkpoint-20
CK70=outputs/sft/gemma4_31b_rft_sft/checkpoint-70
TAGDIR=passk16_olddev_schema_tool_unpatched_temp1p2_tp2_shards4
LOG=outputs/sft/gemma4_31b_rft_sft/chain_ckpt70.log
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date -Is)] waiting for ckpt-20 pass@16 to complete"
# Wait for the merged summary AND for every GPU process to exit.
until [ -f "${CK20}/${TAGDIR}/merged/passk_summary.json" ]; do sleep 60; done
echo "[$(date -Is)] ckpt-20 merged summary present; waiting for GPUs to release"
until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 30; done
echo "[$(date -Is)] GPUs free, starting ckpt-70 pass@16"

MODEL="${CK70}" \
INPUT_FILE=outputs/old-dev-schema-tool-unpatched.jsonl \
DATABASE_DIR=databases/dev_databases \
DIFF_JSON=data/bird_dev_data/raw/bird_dev_unpatched.json \
TP=2 NUM_SHARDS=4 TEMPERATURE=1.2 NUM_GENERATIONS=16 \
BASE_OUT="${CK70}/${TAGDIR}" \
bash scripts/run_passk16_train6601.sh

echo "[$(date -Is)] ckpt-70 pass@16 complete"
