#!/usr/bin/env bash
# TRL-compatible vLLM rollout server for Qwen3.8-27B RL (VLLM_MODE=server).
#
# IMPORTANT: this is NOT the OpenAI-compatible server the eval scripts use. TRL's
# vllm-serve returns raw generated text plus a weight-sync control plane, and its
# argument parser accepts ONLY:
#   --model --revision --tensor-parallel-size --data-parallel-size
#   --gpu-memory-utilization --max-model-len --dtype --kv-cache-dtype
#   --enable-prefix-caching --enforce-eager --speculative-config
#   --distributed-executor-backend --vllm-model-impl --trust-remote-code
#   --host --port --log-level
# Passing --tool-call-parser / --reasoning-parser / --enable-auto-tool-choice
# here is an argument error. Tool calls are parsed CLIENT-SIDE by the trainer via
# nl2sql_gspo.tool_dialects (dialect "qwen"). Those OpenAI flags remain correct
# for the eval scripts, which do talk to the OpenAI server.
#
# Attention: nothing to configure. vLLM auto-selects FLASH_ATTN (FA2) for the 16
# full-attention layers and Triton/FLA GDN kernels for the 48 linear-attention
# layers. Confirm in the log with:
#   grep -E "attention backend|FlashAttention version|GDN" <log>
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

# CRITICAL: put the venv's bin on PATH. FlashInfer's GDN prefill kernel is
# JIT-compiled with ninja; invoking .venv/bin/python directly (without
# activating) leaves .venv/bin off PATH and the JIT dies with
#   RuntimeError: Worker failed with error "[Errno 2] No such file or
#   directory: 'ninja'"
# which kills EngineCore and turns every generation into a generation_error.
export PATH="${REPO_ROOT}/.venv/bin:${PATH}"

export CUDA_VISIBLE_DEVICES="${VLLM_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-6,7}}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# Per-run compile/KV caches. Sharing these across concurrently starting engines
# causes cache races, which is why the pass@16 eval runs isolate them per shard.
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "${LOG_DIR}"
CACHE_ROOT="${CACHE_ROOT:-${LOG_DIR}/vllm_cache_${RUN_TIMESTAMP}}"
mkdir -p "${CACHE_ROOT}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${CACHE_ROOT}/vllm}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${CACHE_ROOT}/inductor}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.8-27B}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-2}"
VLLM_DATA_PARALLEL_SIZE="${VLLM_DATA_PARALLEL_SIZE:-1}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.93}"
# Must cover the LAST tool round, not just prompt + completion: every round
# appends a <tool_response> to the context, so with MAX_PROMPT_LENGTH=20000,
# MAX_COMPLETION_LENGTH=4096 and 8 rounds the final request is far larger than
# 24K. The validated evals ran 43000 at MAX_PROMPT_LENGTH=34000.
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-40960}"
# Safe for RL: TRL calls reset_prefix_cache() after every weight sync
# (trl/generation/vllm_generation.py), so rollouts never reuse KV computed
# under stale policy weights.
VLLM_ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-1}"

VLLM_LOG_FILE="${VLLM_LOG_FILE:-${LOG_DIR}/vllm_qwen38_${RUN_TIMESTAMP}.log}"
echo "[launch_qwen38_vllm] model=${MODEL_NAME} tp=${VLLM_TENSOR_PARALLEL_SIZE} dp=${VLLM_DATA_PARALLEL_SIZE} gpus=${CUDA_VISIBLE_DEVICES}"
echo "[launch_qwen38_vllm] max_model_len=${VLLM_MAX_MODEL_LEN} prefix_caching=${VLLM_ENABLE_PREFIX_CACHING}"
echo "[launch_qwen38_vllm] log=${VLLM_LOG_FILE}"
exec > >(tee -a "${VLLM_LOG_FILE}") 2>&1

EXTRA_ARGS=()
if [[ "${VLLM_ENABLE_PREFIX_CACHING}" == "1" ]]; then
  EXTRA_ARGS+=(--enable_prefix_caching)
fi
if [[ -n "${VLLM_KV_CACHE_DTYPE:-}" ]]; then
  EXTRA_ARGS+=(--kv_cache_dtype "${VLLM_KV_CACHE_DTYPE}")
fi
# GDN (linear-attention) prefill kernel. vLLM defaults to flashinfer, which is
# JIT-compiled on first run; set to "triton" to skip the JIT entirely (this is
# what the pass@16 eval runs used). TRL's vllm-serve parser does not expose
# this, so it goes through the generic vLLM engine arg path when supported.
if [[ -n "${VLLM_GDN_PREFILL_BACKEND:-}" ]]; then
  echo "[launch_qwen38_vllm] NOTE: gdn_prefill_backend=${VLLM_GDN_PREFILL_BACKEND} requested;"
  echo "[launch_qwen38_vllm]       TRL vllm-serve has no such flag -- keep ninja on PATH instead."
fi

"${PYTHON_BIN}" -m nl2sql_gspo.vllm_serve_compat \
  --model "${MODEL_NAME}" \
  --host 0.0.0.0 \
  --port "${VLLM_PORT:-8000}" \
  --tensor_parallel_size "${VLLM_TENSOR_PARALLEL_SIZE}" \
  --data_parallel_size "${VLLM_DATA_PARALLEL_SIZE}" \
  --gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  --max_model_len "${VLLM_MAX_MODEL_LEN}" \
  --dtype bfloat16 \
  "${EXTRA_ARGS[@]}"
