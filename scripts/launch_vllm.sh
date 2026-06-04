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
  elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    set +u
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate nl2sql312
    set -u
  fi
fi

export CUDA_VISIBLE_DEVICES="${VLLM_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-6,7}}"
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
VLLM_SERVER_KIND="${VLLM_SERVER_KIND:-trl}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-2}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.93}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-24576}"

if [[ "${VLLM_SERVER_KIND}" == "async_grpo" || "${VLLM_SERVER_KIND}" == "async" ]]; then
  export VLLM_SERVER_DEV_MODE="${VLLM_SERVER_DEV_MODE:-1}"
  echo "[launch_vllm] starting raw vLLM server for TRL AsyncGRPO"
  "${PYTHON_BIN}" -m vllm.entrypoints.cli.main serve "${MODEL_NAME}" \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE}" \
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${VLLM_MAX_MODEL_LEN}" \
    --dtype bfloat16 \
    --logprobs-mode processed_logprobs \
    --weight-transfer-config '{"backend":"nccl"}'
else
  echo "[launch_vllm] starting TRL compatibility vLLM server"
  "${PYTHON_BIN}" -m nl2sql_gspo.vllm_serve_compat \
    --model "${MODEL_NAME}" \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor_parallel_size "${VLLM_TENSOR_PARALLEL_SIZE}" \
    --gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
    --max_model_len "${VLLM_MAX_MODEL_LEN}" \
    --dtype bfloat16
fi
