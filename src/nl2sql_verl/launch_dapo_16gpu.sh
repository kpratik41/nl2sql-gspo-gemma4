#!/usr/bin/env bash
# Run on the Ray head after identical code/runtime/model/data are staged on both nodes.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
runtime_python="${VERL_PYTHON:-$repo_root/.venv-verl/bin/python}"
if [[ ! -x "$runtime_python" ]]; then
  echo "Missing training runtime: $runtime_python. Prepare it on both nodes first." >&2
  exit 1
fi
export VERL_PYTHON="$runtime_python"
export RAY_ADDRESS="${RAY_ADDRESS:-172.31.11.186:6379}"
export PYTHONPATH="$repo_root${VERL_SOURCE:+:$VERL_SOURCE}${PYTHONPATH:+:$PYTHONPATH}"
export NL2SQL_DATABASE_DIR="${NL2SQL_DATABASE_DIR:-/opt/dlami/nvme/nl2sql/databases}"
export NL2SQL_LENGTH_MAX=4096
export NL2SQL_LENGTH_BUFFER=512
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-2}"
model_path="${MODEL_PATH:-/opt/dlami/nvme/nl2sql/hf/hub/models--google--gemma-4-31B-it/snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475}"
exec "$repo_root/src/nl2sql_verl/worker_python.sh" -m src.nl2sql_verl.train \
  --config-name dapo_fsdp_cp2_16gpu \
  "actor_rollout_ref.model.path=$model_path" \
  "ray_kwargs.ray_init.runtime_env.py_executable=$repo_root/src/nl2sql_verl/worker_python.sh" \
  "$@"
