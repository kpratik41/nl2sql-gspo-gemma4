#!/usr/bin/env bash
# Short replicated-weight forward screening, then attention-level gradient checks.
set -euo pipefail
cd "$(dirname "$0")/.."
benchmark_gpus="${CP_GPUS:-0,1}"
baseline_gpu="${benchmark_gpus%%,*}"
mkdir -p benchmark_results/cp_forward
CUDA_VISIBLE_DEVICES="$baseline_gpu" .venv-bench/bin/python benchmarks/cp_forward.py --cp-size 1 \
  2>&1 | tee benchmark_results/cp_forward/cp1.log
mkdir -p benchmark_results/cp_forward/ring_cp1
CUDA_VISIBLE_DEVICES="$baseline_gpu" .venv-bench/bin/torchrun --standalone --nnodes=1 --nproc-per-node=1 \
  benchmarks/cp_forward.py --cp-size 1 --ring-cp1 --output benchmark_results/cp_forward/ring_cp1 \
  2>&1 | tee benchmark_results/cp_forward/ring_cp1/run.log
CUDA_VISIBLE_DEVICES="$benchmark_gpus" .venv-bench/bin/torchrun --standalone --nnodes=1 --nproc-per-node=2 \
  benchmarks/cp_forward.py --cp-size 2 2>&1 | tee benchmark_results/cp_forward/cp2.log
.venv-bench/bin/python benchmarks/summarize_cp.py
CUDA_VISIBLE_DEVICES="$benchmark_gpus" .venv-bench/bin/torchrun --standalone --nnodes=1 --nproc-per-node=2 \
  benchmarks/cp_attention_gradients.py 2>&1 | tee benchmark_results/cp_forward/attention_gradients.log
# Forward screening is diagnostic: report failures without proceeding to training.
.venv-bench/bin/python - <<'PY'
import json
from pathlib import Path
rows = json.loads(Path('benchmark_results/cp_forward/cp2_rank0_results.json').read_text())
if not all(row['comparison']['screening_pass'] for row in rows):
    raise SystemExit('Whole-model probability screening failed; inspect cp2_rank0_results.json.')
PY
