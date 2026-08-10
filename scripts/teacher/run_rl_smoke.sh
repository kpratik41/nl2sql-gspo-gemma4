#!/usr/bin/env bash
# Fast RL smoke test: exercises the same code paths as the full run at tiny
# volumes, so bugs surface in minutes instead of after a full rollout round.
#
# Deliberately forces an eval and a checkpoint save within the first few steps;
# at production settings those paths would not run for hundreds of steps.
#
# Requires the vLLM server to already be serving the same checkpoint.
cd /home/ubuntu/nl2sql-gspo-gemma4

export ACCELERATE_BIN=/home/ubuntu/nl2sql-gspo-gemma4/.venv/bin/accelerate
export MODEL_NAME="${MODEL_NAME:-/home/ubuntu/nl2sql-gspo-gemma4/outputs/sft/gemma4_31b_rft_sft/checkpoint-20}"
export TRAIN_FILE=outputs/train-6601-schema-bare-tool.jsonl
export EVAL_FILE=outputs/old-dev-schema-bare-tool.jsonl
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/rl/_smoke}"
export DEEPSPEED_CONFIG=configs/ds_zero3_bf16_no_scheduler.json

# Tiny rollout volumes.
export NUM_GENERATIONS=2
export GRADIENT_ACCUMULATION_STEPS=2
export DAPO_OVERSAMPLE_FACTOR=2
export PER_DEVICE_TRAIN_BATCH_SIZE=2
export PER_DEVICE_EVAL_BATCH_SIZE=2
export TRAIN_LIMIT=200
export EVAL_LIMIT=4

# Force eval, save and every beta-schedule transition inside 4 steps.
export BETA_SCHEDULE=0:0.005,2:0.001,3:0
export EVAL_STEPS=2
export SAVE_STEPS=2
export SAVE_ONLY_MODEL=1
export SAVE_LATEST_FULL_CHECKPOINT=0
export MAX_STEPS=4

export LEARNING_RATE=2e-6
export REPORT_TO=none

bash scripts/launch_train.sh
