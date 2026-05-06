#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${PYTHON_BIN:-}" && -x "${REPO_ROOT}/.conda/nl2sql312/bin/python" ]]; then
  PYTHON_BIN="${REPO_ROOT}/.conda/nl2sql312/bin/python"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1 && [[ ! -x "${PYTHON_BIN}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
  else
    echo "[launch_vllm] no python interpreter found" >&2
    exit 1
  fi
fi

if ! "${PYTHON_BIN}" -c "import trl" >/dev/null 2>&1; then
  if [[ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]]; then
    set +u
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
    conda activate nl2sql312
    set -u
  fi
fi

export CUDA_VISIBLE_DEVICES=6,7
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_ALLREDUCE_USE_FLASHINFER="${VLLM_ALLREDUCE_USE_FLASHINFER:-0}"
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

if [[ -n "${VLLM_ATTENTION_BACKEND:-}" ]]; then
  export VLLM_ATTENTION_BACKEND
  echo "[launch_vllm] using explicit attention backend: ${VLLM_ATTENTION_BACKEND}"
else
  echo "[launch_vllm] using vLLM automatic attention backend selection"
fi

RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "${LOG_DIR}"
VLLM_LOG_FILE="${VLLM_LOG_FILE:-${LOG_DIR}/vllm_${RUN_TIMESTAMP}.log}"
echo "[launch_vllm] writing launcher log to ${VLLM_LOG_FILE}"
exec > >(tee -a "${VLLM_LOG_FILE}") 2>&1

MODEL_NAME="${MODEL_NAME:-google/gemma-4-31B-it}"

"${PYTHON_BIN}" -m nl2sql_gspo.vllm_serve_compat \
  --model "${MODEL_NAME}" \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor_parallel_size 2 \
  --gpu_memory_utilization 0.90 \
  --max_model_len 24576 \
  --dtype bfloat16