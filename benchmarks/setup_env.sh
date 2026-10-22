#!/usr/bin/env bash
# Main objective: Gemma 4 31B CP=2 + FFPA (or validated efficient attention)
# for long-context full-parameter NL2SQL RL in verl. CP reduces per-rank
# sequence pressure; efficient attention avoids quadratic score storage.
# Gates: forward memory/speed -> numerical/semantic parity -> real CP2
# -> backward/optimizer/resume -> verl RL -> NL2SQL. FFPA is optional.
# This file: Recreate the measured benchmark environment before validating kernels and model execution.
# See GEMMA4_LONG_CONTEXT_RL.md for the objective and complete roadmap.
# Recreate the measured overlay without modifying the system/project environments.
set -euo pipefail
cd "$(dirname "$0")/.."
base_python="${BENCH_BASE_PYTHON:-/opt/pytorch/bin/python}"
base_site="$($base_python -c 'import pathlib,torch; print(pathlib.Path(torch.__file__).parent.parent)')"
"$base_python" -m venv --system-site-packages .venv-bench
bench_site="$(.venv-bench/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
printf '%s\n' "$base_site" > "$bench_site/pytorch-base.pth"
.venv-bench/bin/python -m pip install --no-deps -r benchmarks/requirements-overlay.txt
.venv-bench/bin/python -c 'import torch; print(torch.__version__, torch.cuda.get_device_name())'
