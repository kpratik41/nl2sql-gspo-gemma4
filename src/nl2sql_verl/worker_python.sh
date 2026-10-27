#!/bin/sh
# Ray starts workers through bash; this host's .bashrc prepends system CUDA.
# Restore wheel library precedence AFTER that shell, before Python loads torch.
set -eu
repo_root="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
runtime_python="${VERL_PYTHON:-$repo_root/.venv-verl/bin/python}"
runtime_cuda_lib="$("$runtime_python" -c 'import sysconfig; from pathlib import Path; print(Path(sysconfig.get_paths()["purelib"])/"nvidia/cu13/lib")')"
if [ -d "$runtime_cuda_lib" ]; then
    export LD_LIBRARY_PATH="$runtime_cuda_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
exec "$runtime_python" "$@"
