#!/usr/bin/env bash
# Qwen3.8-27B tool-calling eval smoke, in-memory async engine (no server).
#
# This is the exact invocation verified working on this box:
#   20 examples, EX 70.00% (14/20), 37 tool calls, 1.85 calls/example,
#   stop_reason_counts {'finished': 18, 'max_tool_rounds': 2}
#
# Run this BEFORE any RL smoke: it exercises the model, the chat template, the
# tool loop and the DB execution path in one shot, on a known-good code path.
#
# Attention backends selected automatically on 8xH200 (nothing forced):
#   FLASH_ATTN backend, FlashAttention version 3   <- the 16 full-attention layers
#   FlashInfer GDN prefill kernel                  <- the 48 linear-attention layers
# Both prefix caching and chunked prefill are on by default.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# CRITICAL: .venv/bin must be on PATH. FlashInfer's GDN prefill kernel is
# JIT-compiled with ninja, and calling .venv/bin/python directly (without
# activating the venv) leaves .venv/bin off PATH. The JIT then fails with
#   RuntimeError: Worker failed with error "[Errno 2] No such file or
#   directory: 'ninja'"
# EngineCore dies, and EVERY sample comes back as generation_error with
# 0 tool calls and 0% accuracy -- while the process still exits 0.
export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
export PYTHONPATH="src:.:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

MODEL_PATH="${MODEL_PATH:-$(ls -d "${HOME}"/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/*/ | head -1)}"
TOTAL="${TOTAL:-20}"
RUN_TAG="${RUN_TAG:-smoke${TOTAL}}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/inference/qwen38_${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
# Per-run caches: concurrent engines sharing these race on the compile cache.
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${LOG_DIR}/${RUN_TAG}-vllm-cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${LOG_DIR}/${RUN_TAG}-inductor-cache}"

# The Qwen-native data files (outputs/qwen-*.jsonl) already carry the Qwen tool
# syntax, so the runner's runtime rewrite MUST be disabled for them: applying it
# on top strips the XML examples (15725 -> 13699 chars) and re-inserts
# "do not print the function call", undoing the whole point of the file.
NO_PROMPT_REWRITE_ARGS=()
if [[ "${NO_PROMPT_REWRITE:-auto}" == "auto" ]]; then
  case "${INPUT_FILE:-}" in
    *qwen-*) NO_PROMPT_REWRITE="1" ;;
    *)       NO_PROMPT_REWRITE="0" ;;
  esac
fi
if [[ "${NO_PROMPT_REWRITE}" == "1" ]]; then
  NO_PROMPT_REWRITE_ARGS=(--no_prompt_rewrite)
fi

echo "[qwen38-eval] model=${MODEL_PATH}"
echo "[qwen38-eval] no_prompt_rewrite=${NO_PROMPT_REWRITE}"
echo "[qwen38-eval] examples=${TOTAL} gpus=${CUDA_VISIBLE_DEVICES} out=${OUTPUT_DIR}"

.venv/bin/python scripts/run_inference_bird_qwen_async.py \
  --model_name_or_path "${MODEL_PATH}" \
  --input_file "${INPUT_FILE:-outputs/old-dev-schema-tool-unpatched.jsonl}" \
  --database_dir "${DATABASE_DIR:-databases/dev_databases}" \
  --diff_json_path "${DIFF_JSON_PATH:-data/bird_dev_data/raw/bird_dev_unpatched.json}" \
  --output_dir "${OUTPUT_DIR}" \
  --num_examples "${TOTAL}" \
  --num_shards "${NUM_SHARDS:-1}" \
  --vllm_tensor_parallel_size "${TP:-2}" \
  --vllm_gpu_memory_utilization "${GPU_MEMORY_UTILIZATION:-0.90}" \
  --vllm_max_model_len "${MAX_MODEL_LEN:-43000}" \
  --vllm_async_concurrency "${CONCURRENCY:-8}" \
  --max_prompt_length "${MAX_PROMPT_LENGTH:-34000}" \
  --max_new_tokens "${MAX_NEW_TOKENS:-8000}" \
  --max_tool_rounds "${MAX_TOOL_ROUNDS:-8}" \
  --temperature "${TEMPERATURE:-0.0}" \
  --top_p "${TOP_P:-1.0}" \
  --top_k "${TOP_K:-20}" \
  --eval_timeout "${EVAL_TIMEOUT:-60}" \
  --eval_workers "${EVAL_WORKERS:-16}" \
  --tool_choice_policy "${TOOL_CHOICE_POLICY:-required_first}" \
  --empty_tool_retries "${EMPTY_TOOL_RETRIES:-1}" \
  "${NO_PROMPT_REWRITE_ARGS[@]}" \
  --overwrite

echo
echo "[qwen38-eval] backends actually used:"
grep -haE "attention backend|FlashAttention version|GDN prefill kernel" "${LOG_DIR}"/*.log 2>/dev/null | tail -3 || true
echo "[qwen38-eval] SANITY: tool_call_count_total must be > 0 and stop_reason_counts"
echo "               must NOT be dominated by generation_error."
