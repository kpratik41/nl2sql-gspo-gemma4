#!/usr/bin/env bash
# Runs everything that depends on the corpus, using both GPUs.
cd "$(dirname "$0")/../experiments"
PY="../.venv/bin/python -u"
R=../results

# Wait until GPU1 has room for the judge model (a preflight may still hold it).
for i in $(seq 1 120); do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1)
  [ "$FREE" -gt 80000 ] && break
  sleep 10
done
echo "GPU1 free: ${FREE} MiB"

# Wave 1: quality (generation, GPU0) || detectability (scoring only, GPU1)
$PY 03_quality.py --device cuda:0 --gsm8k-n 250 > $R/03_quality.log 2>&1 &
P1=$!
$PY 02_detectability.py --device cuda:1 > $R/02_detectability.log 2>&1 &
P2=$!
wait $P1 $P2
echo "WAVE1_DONE"

# Wave 2: robustness (generation, GPU0) || bayesian (training, GPU1)
$PY 04_robustness.py --device cuda:0 > $R/04_robustness.log 2>&1 &
P3=$!
$PY 06_bayesian.py --device cuda:1 > $R/06_bayesian.log 2>&1 &
P4=$!
wait $P3 $P4
echo "WAVE2_DONE"

# Wave 3: overhead alone, so the timing numbers are not polluted by contention.
$PY 05_overhead.py --device cuda:0 > $R/05_overhead.log 2>&1
echo "ALL_DONE"
