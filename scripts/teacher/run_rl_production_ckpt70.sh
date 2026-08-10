#!/usr/bin/env bash
# Production RL run #2: DAPO/GRPO from the SFT ckpt-70 warm start.
#
# Differences from run_rl_production.sh (which started at SFT ckpt-20):
#   - warm start is SFT checkpoint-70 (same 72.62 temp-0 accuracy as ckpt-20,
#     but far fewer truncated completions: 12 vs 38)
#   - DAPO oversample factor K=12 (was 10)
#   - rolling full restart checkpoint written every save (was weight-only only),
#     so a NCCL/collective crash can resume with optimizer state intact
#
# Requires the vLLM server to already be serving the same checkpoint. Restart
# that server whenever this job dies -- a crashed trainer never calls
# close_communicator, so the server keeps a stale weight-sync group and the next
# run fails with "Weight update group already initialized".
#
# Batch math: per_device_bs(2) x steps_per_generation(=grad_accum 16) = 32
# sequences per rank -> 2 groups of num_generations(16) per rank -> 12 unique
# prompts per optimizer step across 6 ranks -> 6601/12 = 550 steps for 1 epoch.
# With K=12 that is 2,304 rollouts generated per step.
cd /home/ubuntu/nl2sql-gspo-gemma4

export ACCELERATE_BIN=/home/ubuntu/nl2sql-gspo-gemma4/.venv/bin/accelerate
export MODEL_NAME="${MODEL_NAME:-/home/ubuntu/nl2sql-gspo-gemma4/outputs/sft/gemma4_31b_rft_sft/checkpoint-70}"
export TRAIN_FILE=outputs/train-6601-schema-bare-tool.jsonl
export EVAL_FILE=outputs/old-dev-schema-bare-tool.jsonl
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/rl/gemma4_31b_sftckpt70_dapo12_lr2e6}"
export DEEPSPEED_CONFIG=configs/ds_zero3_bf16_no_scheduler.json

# Rollout / DAPO
export NUM_GENERATIONS=16
export GRADIENT_ACCUMULATION_STEPS=16
export DAPO_OVERSAMPLE_FACTOR=12
export PER_DEVICE_TRAIN_BATCH_SIZE=2
export PER_DEVICE_EVAL_BATCH_SIZE=16
export TRAIN_LIMIT=-1

# Reward scoring: threads per rank for the DB-backed rewards
# (execution/result/nonnull). Raised from the default 4.
export REWARD_WORKERS=8

# Schedule
export BETA_SCHEDULE=0:0.005,30:0.001,55:0
export LEARNING_RATE=2e-6

# Checkpointing and eval. SAVE_ONLY_MODEL keeps the numbered checkpoint-N dirs
# weight-only (~59G each); SAVE_LATEST_FULL_CHECKPOINT additionally maintains a
# single rolling directory with optimizer + RNG state for crash resume. The
# rolling dir is deleted and rewritten on each save, so it costs one copy.
export SAVE_STEPS=5
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-10}"
export SAVE_ONLY_MODEL=1
export SAVE_LATEST_FULL_CHECKPOINT=1
export LATEST_FULL_CHECKPOINT_DIR_NAME=latest-full-checkpoint
export EVAL_STEPS=250
export EVAL_LIMIT=32

export REPORT_TO=none

bash scripts/launch_train.sh
