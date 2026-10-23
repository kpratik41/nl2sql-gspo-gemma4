#!/usr/bin/env bash
# Reversible process-local precision controls; no model/library source edits.
set -euo pipefail
cd "$(dirname "$0")/.."
benchmark_gpus="${CP_GPUS:-0,1}"
baseline_gpu="${benchmark_gpus%%,*}"
attention_precision="${CP_ATTENTION_PRECISION:-bf16}"
case "$attention_precision" in
  bf16|fp32) ;;
  *) echo 'CP_ATTENTION_PRECISION must be bf16 or fp32.' >&2; exit 2 ;;
esac
result_dir="${CP_RESULT_DIR:-benchmark_results/cp_forward/precision_${attention_precision}}"
fixture_path="${CP_FIXTURE:-benchmark_results/model_parity/fixture.txt}"
read -r -a sequence_lengths <<< "${CP_LENGTHS:-128 1024}"
mkdir -p "$result_dir"
CUDA_VISIBLE_DEVICES="$baseline_gpu" .venv-bench/bin/torchrun --standalone --nnodes=1 --nproc-per-node=1 \
  benchmarks/cp_forward.py --cp-size 1 --ring-cp1 --matmul-policy cublaslt-full \
  --attention-precision "$attention_precision" --lengths "${sequence_lengths[@]}" \
  --fixture "$fixture_path" --output "$result_dir" 2>&1 | tee "$result_dir/cp1.log"
CUDA_VISIBLE_DEVICES="$benchmark_gpus" .venv-bench/bin/torchrun --standalone --nnodes=1 --nproc-per-node=2 \
  benchmarks/cp_forward.py --cp-size 2 --matmul-policy cublaslt-full \
  --attention-precision "$attention_precision" --lengths "${sequence_lengths[@]}" \
  --fixture "$fixture_path" --output "$result_dir" 2>&1 | tee "$result_dir/cp2.log"
.venv-bench/bin/python - "$result_dir" <<'PY'
import json
import sys
from pathlib import Path
rows = json.loads((Path(sys.argv[1])/'cp2_rank0_results.json').read_text())
if not all(row['comparison']['screening_pass'] for row in rows):
    raise SystemExit('Probability screening failed; inspect the per-token diagnostics. Thresholds were not relaxed.')
PY
