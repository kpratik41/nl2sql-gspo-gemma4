#!/usr/bin/env bash
# Serve Muse-Glimmer through the dedicated vLLM Docker image.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

MODEL_PATH="${MODEL_PATH:-/root/.cache/huggingface/hub/models--meta-models--Muse-Glimmer-30B/snapshots/a4e59da52a7bc87ae7251dd5545c0dd437c44b68}"
HOST_HF_CACHE="${HOST_HF_CACHE:-/home/ec2-user/.cache/huggingface}"
IMAGE="${IMAGE:-vllm/vllm-openai:muse-glimmer}"
PORT="${PORT:-8000}"
TP="${TP:-4}"
GPU_DEVICES="${GPU_DEVICES:-all}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-43000}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"

if [[ "${GPU_DEVICES}" == "all" ]]; then
  GPU_ARGS=(--gpus all)
else
  GPU_ARGS=(--gpus "\"device=${GPU_DEVICES}\"")
fi

docker run --rm "${GPU_ARGS[@]}" --ipc=host --shm-size 32g \
  -p "${PORT}:8000" \
  -v "${HOST_HF_CACHE}:/root/.cache/huggingface" \
  "${IMAGE}" \
  "${MODEL_PATH}" \
  --served-model-name muse-glimmer \
  --tensor-parallel-size "${TP}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --enable-auto-tool-choice \
  --tool-call-parser muse_glimmer \
  --reasoning-parser muse_glimmer \
  --generation-config auto
