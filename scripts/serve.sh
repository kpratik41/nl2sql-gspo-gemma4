#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
SERVER_PYTHON=${SERVER_PYTHON:-/home/ec2-user/watermarking-test1008/.venv-v030/bin/python}
export PATH="$(dirname "$SERVER_PYTHON"):$PATH"
# The shared environment has mismatched CUDA compiler/runtime headers.
# These supported native/Triton paths avoid FlashInfer JIT entirely.
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ALLREDUCE_USE_FLASHINFER=0
MODEL_PATH=${MODEL_PATH:-/home/ec2-user/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/snapshots/a9904d24bcc1d289a1950fa9d2b978c47cf903b9}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
exec "$SERVER_PYTHON" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" --served-model-name nemotron-lightning-bf16 \
  --host 127.0.0.1 --port "${PORT:-8000}" \
  --dtype bfloat16 --tensor-parallel-size 4 \
  --max-model-len 43000 --max-num-seqs 16 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.85 --enable-prefix-caching \
  --moe-backend triton --mamba-backend triton --attention-backend FLASH_ATTN \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser nemotron_v3 --generation-config vllm \
  --seed 0 "$@"
