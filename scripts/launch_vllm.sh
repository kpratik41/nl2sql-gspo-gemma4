#!/usr/bin/env bash
set -euo pipefail

if ! python -c "import trl" >/dev/null 2>&1; then
  if [[ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]]; then
    set +u
    # shellcheck disable=SC1091
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
    conda activate nl2sql312
    set -u
  fi
fi

export CUDA_VISIBLE_DEVICES=6,7
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "${LOG_DIR}"
VLLM_LOG_FILE="${VLLM_LOG_FILE:-${LOG_DIR}/vllm_${RUN_TIMESTAMP}.log}"
echo "[launch_vllm] writing launcher log to ${VLLM_LOG_FILE}"
exec > >(tee -a "${VLLM_LOG_FILE}") 2>&1

MODEL_NAME="google/gemma-4-31B-it"

python -m nl2sql_gspo.vllm_serve_compat \
  --model "${MODEL_NAME}" \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor_parallel_size 2 \
  --gpu_memory_utilization 0.90 \
  --max_model_len 24576 \
  --dtype bfloat16