#!/usr/bin/env bash
# Full evaluation, in dependency order. Everything after 01 reads the corpus.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-../.venv/bin/python} -u"

echo "== 01 corpus =="       && $PY 01_generate_corpus.py --samples-per-prompt "${REPS:-4}"
echo "== 02 detectability ==" && $PY 02_detectability.py
echo "== 03 quality =="       && $PY 03_quality.py --gsm8k-n "${GSM8K_N:-250}"
echo "== 04 robustness =="    && $PY 04_robustness.py
echo "== 05 overhead =="      && $PY 05_overhead.py
echo "== 06 bayesian =="      && $PY 06_bayesian.py
echo "== report =="           && $PY ../scripts/build_report_md.py && $PY ../scripts/build_report.py
