#!/usr/bin/env bash
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
