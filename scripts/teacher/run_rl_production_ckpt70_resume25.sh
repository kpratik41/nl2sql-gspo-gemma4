#!/usr/bin/env bash
# Resume the ckpt-70 RL run from the rolling full checkpoint at global_step 25.
#
# The original run (logs/train_20260805_143121.log) died mid-rollout on step 29
# with a NCCL collective timeout: five ranks sat in the scalar ALLREDUCE from
# _global_sum_int for 1800s while one rank was still executing its rollouts'
# sqlite_query calls serially. The 1800s came from TrainingArguments.ddp_timeout
# (the TORCH_NCCL_TIMEOUT_MS export in launch_train.sh is inert -- PyTorch never
# reads it); ddp_timeout is now 7200 in train_gspo_nl2sql.py.
#
# latest-full-checkpoint carries DeepSpeed optimizer state (global_step25/) and
# per-rank RNG, so training picks up at optimizer step 26 with the beta schedule
# intact: 0.005 through step 30, 0.001 from step 31, 0 from step 56.
#
# checkpoint-25 is weight-only and cannot be used here; launch_train.sh rejects
# model-only checkpoints for --resume_from_checkpoint by design.
#
# The vLLM server must be restarted before this runs -- the crashed trainer
# never called close_communicator, so a surviving server holds a stale weight
# sync group.
cd /home/ubuntu/nl2sql-gspo-gemma4

export RESUME_FROM_CHECKPOINT=/home/ubuntu/nl2sql-gspo-gemma4/outputs/rl/gemma4_31b_sftckpt70_dapo12_lr2e6/latest-full-checkpoint

exec bash scripts/teacher/run_rl_production_ckpt70.sh
