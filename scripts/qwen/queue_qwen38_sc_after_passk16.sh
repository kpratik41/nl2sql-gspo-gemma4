#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

PASSK_DIR="${PASSK_DIR:-outputs/passk/qwen3p8_27b_olddev_schema_tool_passk16_temp1p2_tp2_shards4_c16/merged}"
TEMP0_DIR="${TEMP0_DIR:-outputs/inference/dev/old-dev-schema-tool-unpatched/Qwen3.8-27B/full1534_tp1_shards8_temp0_openai_tool_qwen3_coder_c16}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/analysis/qwen3p8_27b_self_consistency_passk16_temp1p2_tp2_shards4_c16}"
DATABASE_DIR="${DATABASE_DIR:-databases/dev_databases}"
PYTHON_BIN="${PYTHON_BIN:-/home/ec2-user/miniconda3/envs/nl2sql312/bin/python}"
LOG="${OUTPUT_DIR}/queue_and_run.log"

mkdir -p "${OUTPUT_DIR}"
exec > >(tee -a "${LOG}") 2>&1

echo "[$(date -Is)] waiting for merged pass@16 candidates"
echo "[$(date -Is)] passk_dir=${PASSK_DIR}"
echo "[$(date -Is)] temp0_dir=${TEMP0_DIR}"

while [[ ! -s "${PASSK_DIR}/passk_candidates.jsonl" || ! -s "${PASSK_DIR}/passk_per_example.jsonl" ]]; do
  echo "[$(date -Is)] pass@16 merge not ready yet"
  sleep 120
done

echo "[$(date -Is)] pass@16 merge ready; writing distribution"
export PASSK_DIR
"${PYTHON_BIN}" - <<'PY'
import json
import os
from collections import Counter
from pathlib import Path

passk_dir = Path(os.environ["PASSK_DIR"])
rows = [json.loads(line) for line in (passk_dir / "passk_per_example.jsonl").read_text().splitlines() if line.strip()]
summary = json.loads((passk_dir / "passk_summary.json").read_text())
dist = Counter(int(row.get("num_correct", 0)) for row in rows)

lines = [
    "# Qwen3.8 pass@16 Correct-Count Distribution",
    "",
    f"- examples: `{len(rows)}`",
    f"- candidates: `{summary.get('total_candidates')}`",
    f"- pass@16 estimated: `{summary.get('pass_at_k_estimated', {}).get('16', 0):.2f}%`",
    f"- prefix pass@16: `{summary.get('prefix_pass_at_k', {}).get('16', 0):.2f}%`",
    f"- candidate accuracy: `{summary.get('candidate_accuracy', {}).get('accuracy', 0):.2f}%`",
    "",
    "| correct candidates out of 16 | examples |",
    "| ---: | ---: |",
]
for k in range(17):
    lines.append(f"| {k} | {dist.get(k, 0)} |")

(passk_dir / "correct_count_distribution.md").write_text("\n".join(lines) + "\n")
(passk_dir / "correct_count_distribution.json").write_text(json.dumps(dict(sorted(dist.items())), indent=2) + "\n")
print("\n".join(lines))
PY

echo "[$(date -Is)] starting self-consistency"
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

echo "[$(date -Is)] self-consistency complete"
echo "[$(date -Is)] summary=${OUTPUT_DIR}/self_consistency_summary.md"
