#!/usr/bin/env bash
# vLLM rollout server for the RL run from SFT ga16 checkpoint-25.
#
# Start this BEFORE the trainer and leave it running. Restart it whenever the
# training process dies: TRL's weight-sync group is established by the trainer
# calling init_communicator, and a training crash never calls
# close_communicator, so the server keeps a stale group and the next run fails
# with "Weight update group already initialized" on the server side and
# "NCCL error: remote process exited" on the client side.
#
# Serves on GPUs 6,7; the trainer uses 0-5.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/ubuntu/sft-rl/nl2sql-gspo-gemma4}"
cd "${REPO_ROOT}"

export PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
export MODEL_NAME="${MODEL_NAME:-${REPO_ROOT}/outputs/sft/run3_gemma4_31b_rft_sft_ga16/checkpoint-25}"
export VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-2}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-24576}"
export VLLM_CUDA_VISIBLE_DEVICES="${VLLM_CUDA_VISIBLE_DEVICES:-6,7}"

echo "[vllm] repo=${REPO_ROOT}"
echo "[vllm] model=${MODEL_NAME}"
echo "[vllm] gpus=${VLLM_CUDA_VISIBLE_DEVICES} tp=${VLLM_TENSOR_PARALLEL_SIZE} max_model_len=${VLLM_MAX_MODEL_LEN}"

bash scripts/launch_vllm.sh
