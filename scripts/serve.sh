#!/usr/bin/env bash
# Serve Qwen3.6-35B-A3B-FP8 with vLLM.
#
#   ./scripts/serve.sh            # 1 replica on GPU 0, port 8000
#   ./scripts/serve.sh 4          # 4 replicas on GPUs 0-3, ports 8000-8003
#
# Why replicas instead of one big tensor-parallel server: an agent episode is a
# serial chain (screenshot -> infer -> act), so it uses a sliver of one H100's
# throughput. The bottleneck is concurrent episodes, not single-request latency.
# FP8 weights are ~35GB, so one replica fits comfortably per 80GB H100.
set -euo pipefail
cd "$(dirname "$0")/.."

# vLLM JIT-compiles the Gated-DeltaNet kernels and shells out to `ninja`, which
# lives in .venv/bin. We invoke .venv/bin/python directly rather than activating
# the venv, so put it on PATH explicitly or engine init fails with FileNotFoundError.
export PATH="$PWD/.venv/bin:$PATH"

MODEL=${MODEL:-Qwen/Qwen3.6-35B-A3B-FP8}
REPLICAS=${1:-1}
BASE_PORT=${BASE_PORT:-8000}
# 262k native context is available, but KV memory scales with it. 32k fits many
# screenshots (1000 vision tokens each at 1280x800); raise if you need deeper runs.
#
# For OSWorld 2.0 use MAX_LEN=131072: its screens are 1920x1080 = 2025 vision
# tokens each, and the reference agent keeps 20 of them, so images alone are
# ~40k tokens before any text.
MAX_LEN=${MAX_LEN:-32768}

mkdir -p logs
for i in $(seq 0 $((REPLICAS - 1))); do
  PORT=$((BASE_PORT + i))
  echo "replica $i -> GPU $i, port $PORT"
  CUDA_VISIBLE_DEVICES=$i .venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name "$MODEL" \
    --port "$PORT" \
    --tensor-parallel-size 1 \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization 0.90 \
    --limit-mm-per-prompt "{\"image\": ${MAX_IMAGES:-8}}" \
    --enable-prefix-caching \
    --no-enable-log-requests \
    > "logs/vllm_$PORT.log" 2>&1 &
done

echo
echo "Starting $REPLICAS replica(s). First load takes a few minutes (weights + graph capture)."
echo "Watch:   tail -f logs/vllm_${BASE_PORT}.log"
echo "Ready:   curl -s localhost:${BASE_PORT}/v1/models"
wait
