#!/usr/bin/env bash
# Production RL run (DAPO/GRPO) from the SFT ga16 checkpoint-25 warm start.
#
# checkpoint-25 is the best SFT checkpoint measured: 73.60% temp-0 dev EX
# (1129/1534) and 82.40% pass@16 over the first 500 dev examples.
#
# Requires scripts/teacher/run_vllm_sft_ga16_ckpt25.sh to already be serving the
# same checkpoint on GPUs 6,7. Restart that server whenever this job dies -- a
# crashed trainer never calls close_communicator, so the server keeps a stale
# weight-sync group and the next run fails with "Weight update group already
# initialized".
#
# Batch math: per_device_bs(2) x steps_per_generation(=grad_accum 32) = 64
# sequences per rank -> 4 groups of num_generations(16) per rank -> 24 unique
# prompts per optimizer step across 6 ranks -> 6574/24 = 274 steps for 1 epoch.
# With K=8 that is 3,072 rollouts generated per step.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/ubuntu/sft-rl/nl2sql-gspo-gemma4}"
cd "${REPO_ROOT}"

export ACCELERATE_BIN="${ACCELERATE_BIN:-${REPO_ROOT}/.venv/bin/accelerate}"
export MODEL_NAME="${MODEL_NAME:-${REPO_ROOT}/outputs/sft/run3_gemma4_31b_rft_sft_ga16/checkpoint-25}"

# Typefix train file: corrected SQLite column-type labels, 27 defective golds
# dropped. This is the file the SFT warm start was built from, so RL sees the
# same schema text and the same gold set.
export TRAIN_FILE="${TRAIN_FILE:-outputs/train-6574-schema-bare-tool-typefix.jsonl}"
export EVAL_FILE="${EVAL_FILE:-outputs/old-dev-schema-bare-tool.jsonl}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/rl/gemma4_31b_sftga16ckpt25_dapo8_lr3e6}"
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/ds_zero3_bf16_no_scheduler.json}"

# Rollout / DAPO
export NUM_GENERATIONS="${NUM_GENERATIONS:-16}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-32}"
export DAPO_OVERSAMPLE_FACTOR="${DAPO_OVERSAMPLE_FACTOR:-8}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
export PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-16}"
export TRAIN_LIMIT="${TRAIN_LIMIT:--1}"

# Reward scoring: threads per rank for the DB-backed rewards
# (execution/result/nonnull).
export REWARD_WORKERS="${REWARD_WORKERS:-8}"

# Schedule
export BETA_SCHEDULE="${BETA_SCHEDULE:-0:0.005,30:0.001,55:0}"
export LEARNING_RATE="${LEARNING_RATE:-3e-6}"

# Checkpointing and eval.
# SAVE_ONLY_MODEL keeps the numbered checkpoints weights-only (~59 GB each);
# SAVE_LATEST_FULL_CHECKPOINT keeps one rolling full checkpoint (optimizer
# state included) so the run can be resumed after a crash.
export SAVE_STEPS="${SAVE_STEPS:-5}"
export SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-1}"
export SAVE_LATEST_FULL_CHECKPOINT="${SAVE_LATEST_FULL_CHECKPOINT:-1}"
export EVAL_STEPS="${EVAL_STEPS:-250}"
export EVAL_LIMIT="${EVAL_LIMIT:-32}"

export REPORT_TO="${REPORT_TO:-none}"

echo "[rl] repo=${REPO_ROOT}"
echo "[rl] model=${MODEL_NAME}"
echo "[rl] train=${TRAIN_FILE}"
echo "[rl] eval=${EVAL_FILE}"
echo "[rl] output=${OUTPUT_DIR}"
echo "[rl] k=${NUM_GENERATIONS} ga=${GRADIENT_ACCUMULATION_STEPS} oversample=${DAPO_OVERSAMPLE_FACTOR} lr=${LEARNING_RATE}"
echo "[rl] save_steps=${SAVE_STEPS} save_only_model=${SAVE_ONLY_MODEL} rolling_full_ckpt=${SAVE_LATEST_FULL_CHECKPOINT}"

bash scripts/launch_train.sh
