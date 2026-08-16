#!/usr/bin/env bash
# Waits for the sft_rft3 ckpt-10 pass@16 merge, then runs execution-result
# self-consistency over the 16 candidates plus the ckpt-10 temperature-0 run.
#
# Runs from the consensus-branch checkout (Gemma checkpoint). Data and outputs live
# in the sibling nl2sql-gspo-gemma4 checkout and are referenced by absolute path.
#
#   bash scripts/queue_sc_after_ckpt10_passk16.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

MAIN_REPO="${MAIN_REPO:-/home/ec2-user/consensus/nl2sql-gspo-gemma4}"
PYTHON_BIN="${PYTHON_BIN:-/home/ec2-user/miniconda3/envs/nl2sql312/bin/python}"
CKPT_ROOT="${CKPT_ROOT:-${MAIN_REPO}/outputs/sft_rft3/gemma4_31b_sftckpt56_dapo12_lr2e6_beta_0p005_0p001_0/checkpoint-10}"

PASSK_DIR="${PASSK_DIR:-${MAIN_REPO}/outputs/passk/gemma4_31b_sftrft3_ckpt10_olddev_schema_tool_passk16_temp1p2_tp2_shards4/merged}"
TEMP0_DIR="${TEMP0_DIR:-${CKPT_ROOT}/temp0_olddev_schema_tool_unpatched_vllm_async_tp2_shards4_ctx43k}"
DATABASE_DIR="${DATABASE_DIR:-${MAIN_REPO}/databases/dev_databases}"
OUTPUT_DIR="${OUTPUT_DIR:-${MAIN_REPO}/outputs/analysis/gemma4_31b_sftrft3_ckpt10_sc_passk16}"
STATUS_FILE="${OUTPUT_DIR}/PROGRESS.md"

mkdir -p "${OUTPUT_DIR}"
LOG="${OUTPUT_DIR}/queue_and_run.log"

exec > >(tee -a "${LOG}") 2>&1

echo "[$(date -Is)] SC queued for ckpt-10 pass@16"
echo "[$(date -Is)] repo=$(pwd) branch=$(git rev-parse --abbrev-ref HEAD)"
echo "[$(date -Is)] passk_dir=${PASSK_DIR}"
echo "[$(date -Is)] temp0_dir=${TEMP0_DIR}"
echo "[$(date -Is)] output=${OUTPUT_DIR}"

if [[ ! -s "${TEMP0_DIR}/eval_results.jsonl" ]]; then
  echo "[$(date -Is)] missing temp0 eval_results.jsonl at ${TEMP0_DIR}" >&2
  exit 1
fi

status() {
  {
    echo "# SC progress — gemma4-31b sft_rft3 ckpt-10"
    echo
    echo "- updated: \`$(date -Is)\`"
    echo "- state: \`$1\`"
    echo "- passk: \`${PASSK_DIR}\`"
    echo "- temp0: \`${TEMP0_DIR}\`"
    if [[ -f "${OUTPUT_DIR}/self_consistency_summary.md" ]]; then
      echo
      cat "${OUTPUT_DIR}/self_consistency_summary.md"
    fi
  } > "${STATUS_FILE}.tmp"
  mv -f "${STATUS_FILE}.tmp" "${STATUS_FILE}"
}

status "waiting for pass@16 merge"
while [[ ! -s "${PASSK_DIR}/passk_candidates.jsonl" || ! -s "${PASSK_DIR}/passk_summary.json" ]]; do
  echo "[$(date -Is)] pass@16 merge not ready yet"
  status "waiting for pass@16 merge"
  sleep 120
done

echo "[$(date -Is)] merge ready; writing correct-count distribution"
status "computing distribution"
export PASSK_DIR
"${PYTHON_BIN}" - <<'PY'
import json, os
from collections import Counter
from pathlib import Path

passk_dir = Path(os.environ["PASSK_DIR"])
rows = [json.loads(l) for l in (passk_dir / "passk_per_example.jsonl").read_text().splitlines() if l.strip()]
summary = json.loads((passk_dir / "passk_summary.json").read_text())
dist = Counter(int(r.get("num_correct", 0)) for r in rows)

lines = [
    "# ckpt-10 pass@16 Correct-Count Distribution",
    "",
    f"- examples: `{len(rows)}`",
    f"- candidates: `{summary.get('total_candidates')}`",
    f"- pass@16 estimated: `{summary.get('pass_at_k_estimated', {}).get('16', 0):.2f}%`",
    f"- candidate accuracy: `{summary.get('candidate_accuracy', {}).get('accuracy', 0):.2f}%`",
    "",
    "| correct candidates out of 16 | examples |",
    "| ---: | ---: |",
]
for k in range(17):
    lines.append(f"| {k} | {dist.get(k, 0)} |")
(passk_dir / "correct_count_distribution.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY

echo "[$(date -Is)] starting self-consistency"
status "running self-consistency"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd):$(pwd)/src:${PYTHONPATH:-}"

"${PYTHON_BIN}" scripts/evaluate_self_consistency.py \
  --passk-dir-template "${PASSK_DIR}" \
  --checkpoint-dir-template "${TEMP0_DIR}" \
  --database-dir "${DATABASE_DIR}" \
  --ckpts 0 \
  --num-generations 16 \
  --workers 16 \
  --eval-timeout 60 \
  --output-dir "${OUTPUT_DIR}" \
  --overwrite

status "complete"
echo "[$(date -Is)] self-consistency complete"
echo "[$(date -Is)] summary=${OUTPUT_DIR}/self_consistency_summary.md"
cat "${OUTPUT_DIR}/self_consistency_summary.md"
