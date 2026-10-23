#!/usr/bin/env bash
# Build a standalone environment without inheriting the system site-packages.
set -euo pipefail
cd "$(dirname "$0")/.."
base_python="${BENCH_BASE_PYTHON:-/opt/pytorch/bin/python}"
if [[ ! -d .venv-bench ]]; then
  "$base_python" -m venv .venv-bench
fi
if .venv-bench/bin/python -c 'import sys; assert sys.prefix != sys.base_prefix; assert "include-system-site-packages = false" in open(sys.prefix + "/pyvenv.cfg").read()'; then
  .venv-bench/bin/python -m pip install --upgrade pip
  .venv-bench/bin/python -m pip install 'torch==2.10.0' 'torchao==0.16.0+cu130' --index-url https://download.pytorch.org/whl/cu130
  .venv-bench/bin/python -m pip install -r benchmarks/requirements-automodel.txt
  .venv-bench/bin/python -m pip check
else
  echo 'Expected a standalone .venv-bench; use a fresh directory rather than the historical overlay.' >&2
  exit 1
fi
