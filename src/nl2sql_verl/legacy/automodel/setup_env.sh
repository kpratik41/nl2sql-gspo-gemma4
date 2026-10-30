#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$repo_root"
runtime_python="${VERL_BASE_PYTHON:-python3.12}"
runtime_venv="${VERL_VENV:-$repo_root/.venv-verl}"
source_root="${VERL_SOURCE:-$repo_root/outputs/verl_runtime/verl}"
revision=d430157d15516d89e0cc853fd10fcd21d9234c44
"$runtime_python" -c 'import sys; assert sys.version_info[:2] == (3,12), "Use Python 3.12 for the rollout runtime"'
if [[ ! -d "$runtime_venv" ]]; then
  "$runtime_python" -m venv "$runtime_venv"
fi
"$runtime_venv/bin/python" -m pip install --upgrade pip
"$runtime_venv/bin/python" -m pip install 'torch==2.10.0+cu130' 'torchvision==0.25.0+cu130' 'torchaudio==2.10.0+cu130' 'torchao==0.16.0+cu130' --index-url https://download.pytorch.org/whl/cu130
"$runtime_venv/bin/python" -m pip install -r src/nl2sql_verl/runtime-requirements.txt
if [[ ! -d "$source_root/.git" ]]; then
  git clone --no-checkout https://github.com/verl-project/verl.git "$source_root"
  git -C "$source_root" fetch --depth 1 origin "$revision"
  git -C "$source_root" checkout --detach "$revision"
fi
if [[ "$(git -C "$source_root" rev-parse HEAD)" != "$revision" ]]; then
  echo "Expected verl source at $revision; choose a clean VERL_SOURCE directory." >&2
  exit 1
fi
# Import the pinned source directly: installing its wheel would downgrade the
# Transformers 5.18 required by our AutoModel pin. This is an experimental local
# adapter, not an upstream-supported combined dependency matrix.
"$runtime_venv/bin/python" - "$source_root" "$repo_root" <<'PY'
import site,sys
from pathlib import Path
Path(site.getsitepackages()[0], 'nl2sql-verl-source.pth').write_text(sys.argv[1]+'\n'+sys.argv[2]+'\n')
PY
"$runtime_venv/bin/python" -m pip check
VERL_PYTHON="$runtime_venv/bin/python" src/nl2sql_verl/worker_python.sh -c 'import torch; import vllm._C; from vllm import LLM; print("vLLM CUDA extension import passed")'
"$runtime_venv/bin/python" -m src.nl2sql_verl.preflight --config smoke --config-only
mkdir -p outputs/verl_recipe_checks
"$runtime_venv/bin/python" -m pip freeze > outputs/verl_recipe_checks/runtime-lock.txt
echo "Runtime prepared at $runtime_venv; full 31B rollout/update still requires its smoke validation."
