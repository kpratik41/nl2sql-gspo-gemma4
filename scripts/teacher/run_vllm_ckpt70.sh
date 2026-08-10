#!/usr/bin/env bash
# vLLM rollout server for the ckpt-70 RL run, serving the SFT checkpoint.
#
# Restart this whenever the training process dies. TRL's weight-sync group is
# established by the trainer calling init_communicator; a training crash never
# calls close_communicator, so the server keeps a stale group and the next run
# fails with "Weight update group already initialized" on the server side and
# "NCCL error: remote process exited" on the client side.
cd /home/ubuntu/nl2sql-gspo-gemma4

export PYTHON_BIN=/home/ubuntu/nl2sql-gspo-gemma4/.venv/bin/python
export MODEL_NAME="${MODEL_NAME:-/home/ubuntu/nl2sql-gspo-gemma4/outputs/sft/gemma4_31b_rft_sft/checkpoint-70}"
export VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-2}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-24576}"

bash scripts/launch_vllm.sh
