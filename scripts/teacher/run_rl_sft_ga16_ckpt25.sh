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
# With K=14 that is 5,376 rollouts generated per step.
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
#
# K=11: DAPO selection is per-rank, not global -- each of the 6 ranks takes the
# first `target_local_groups` (24/6 = 4) heterogeneous groups from its own
# `4 * K` attempts, and a rank with a surplus cannot donate to one that came up
# short. Heterogeneity fell from ~33% over the first ten steps to ~15% by step
# 16 as the policy converged (all_correct rose 112 -> 142), at which point K=8
# gave each rank a mean of 5 heterogeneous groups against 4 needed -- close
# enough to the boundary that 1-3 of 24 slots were padded with zero-advantage
# groups on most steps. K=11 raises the per-rank mean to ~6.6 and cuts the
# padding probability down. Raised again to 14 at the step-35 restart:
# heterogeneity kept falling as the policy converged (7.95% at step 35,
# fill_rate 66.67%), so K=11 left each rank a mean of ~4.0 against 4 needed.
export NUM_GENERATIONS="${NUM_GENERATIONS:-16}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-32}"
export DAPO_OVERSAMPLE_FACTOR="${DAPO_OVERSAMPLE_FACTOR:-14}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
export PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-16}"
export TRAIN_LIMIT="${TRAIN_LIMIT:--1}"

# Reward scoring: threads per rank for the DB-backed rewards
# (execution/result/nonnull).
export REWARD_WORKERS="${REWARD_WORKERS:-8}"

# Schedule
#
# Milestones are ABSOLUTE global_step values, and `_current_beta` is a step
# function (last milestone whose start_step has been reached), not interpolated.
# On a resume `global_step` continues from the checkpoint rather than resetting,
# so a run resumed at step 20 begins at beta=0.001 here -- steps 0-19 at 0.005
# were already served by the original run.
export BETA_SCHEDULE="${BETA_SCHEDULE:-0:0.005,20:0.001,35:0}"
export LEARNING_RATE="${LEARNING_RATE:-3e-6}"

# Resume from a full checkpoint (weights + DeepSpeed optimizer state). Leave
# unset for a fresh run. Must point at a directory containing global_stepN/,
# not a weights-only checkpoint.
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  export RESUME_FROM_CHECKPOINT
fi

# Checkpointing and eval.
# SAVE_ONLY_MODEL keeps the numbered checkpoints weights-only (~59 GB each);
# SAVE_LATEST_FULL_CHECKPOINT keeps one rolling full checkpoint (optimizer
# state included) so the run can be resumed after a crash.
# The rolling full checkpoint goes to ephemeral NVMe, the numbered weights-only
# checkpoints stay on EBS under OUTPUT_DIR.
#
# It is ~408 GB (327 GB of DeepSpeed optimizer state across 6 ranks) and took
# 36m22s to write to the EBS root -- ~167 MB/s sustained, on a volume that
# benchmarked at 7.9 MB/s under O_DIRECT while training was running. The local
# NVMe stripe measured 6.5 GB/s, so the same write should take ~2 min. That also
# shrinks the window in which no valid resume point exists, because
# _save_latest_restart_checkpoint rmtree's the old directory before writing the
# new one.
#
# An absolute path works because the trainer builds the location with
# os.path.join(run_dir, name), and os.path.join discards run_dir when the second
# argument is absolute. Do NOT symlink the directory instead: the rmtree above
# raises on a symlink-to-directory.
#
# NVMe is ephemeral -- lost on instance stop/terminate. The durable artifacts
# are the numbered checkpoints on EBS.
export SAVE_STEPS="${SAVE_STEPS:-5}"
export SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-1}"
export SAVE_LATEST_FULL_CHECKPOINT="${SAVE_LATEST_FULL_CHECKPOINT:-1}"
export LATEST_FULL_CHECKPOINT_DIR_NAME="${LATEST_FULL_CHECKPOINT_DIR_NAME:-/opt/dlami/nvme/rl_ckpt/gemma4_31b_sftga16ckpt25_dapo11_lr3e6/latest-full-checkpoint}"
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
echo "[rl] rolling_full_ckpt_dir=${LATEST_FULL_CHECKPOINT_DIR_NAME}"
echo "[rl] beta_schedule=${BETA_SCHEDULE}"
echo "[rl] resume_from=${RESUME_FROM_CHECKPOINT:-<none>}"

bash scripts/launch_train.sh
