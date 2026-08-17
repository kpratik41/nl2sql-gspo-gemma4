#!/usr/bin/env bash
# Qwen3.8-27B GSPO/DAPO production RL run.
#
# Two screens:
#   screen -S vllm  -dm bash scripts/qwen/launch_qwen38_vllm.sh        # GPUs 6,7
#   screen -S train -dm bash scripts/qwen/run_rl_train_qwen38.sh       # GPUs 0-5
# Start vLLM first and wait for /health to answer 200 (follows a 307).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# FlashInfer's GDN kernel is JIT-compiled with ninja; without .venv/bin on PATH
# the JIT dies and every generation returns an error while the process exits 0.
export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
export ACCELERATE_BIN="${ACCELERATE_BIN:-${REPO_ROOT}/.venv/bin/accelerate}"

# Resolve to the local snapshot rather than the repo id. With beta > 0 TRL loads
# a separate reference model, and passing a repo id makes all 6 ranks resolve it
# through the hub at the same moment. That races on the cache lock and one rank
# reports shards missing that are demonstrably present:
#   OSError: Qwen/Qwen3.8-27B does not appear to have files named
#            ('model-00014-of-00018.safetensors', ...)
# A local path plus HF_HUB_OFFLINE removes hub resolution from the critical path
# entirely. The snapshot is verified complete: all 18 shards in the index exist.
export MODEL_NAME="${MODEL_NAME:-$(ls -d "${HOME}"/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/*/ | head -1)}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

export TOOL_DIALECT="${TOOL_DIALECT:-qwen}"
export ENABLE_TOOL_ROLLOUTS="${ENABLE_TOOL_ROLLOUTS:-1}"

# The repeat-tool-call guard is a temperature-0 fix; rollouts sample at 1.2 and
# never form the fixed point it targets. Explicitly off so it cannot be
# inherited from a shell that ran the eval scripts.
export NL2SQL_TOOL_LOOP_GUARD=0

export VLLM_MODE="${VLLM_MODE:-server}"
export VLLM_SERVER_BASE_URL="${VLLM_SERVER_BASE_URL:-http://127.0.0.1:8000}"

# Full 6573-row bare tool set, not the 3000 subset: at 168 prompts consumed per
# optimizer step the subset is only ~18 steps per epoch, so the beta schedule
# (which runs to step 35) would cycle it twice before annealing to 0.
export TRAIN_FILE="${TRAIN_FILE:-outputs/qwen-train-6573-schema-bare-tool.jsonl}"
export EVAL_FILE="${EVAL_FILE:-outputs/qwen-old-dev-schema-bare-tool.jsonl}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/rl/qwen38_27b_gspo_run1}"
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/ds_zero3_bf16_no_scheduler.json}"
export FSDP_CONFIG="${FSDP_CONFIG:-configs/fsdp_qwen38_bf16.json}"

# torch 2.10 SDPA picks cuDNN's fused kernel at head_dim=256 on H200 (0.355 ms
# vs 0.746 for Flash). flash_attention_2 is not selectable: flash-attn will not
# build against host CUDA 13.2 with a cu128 torch.
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"

export ACCELERATE_NUM_PROCESSES="${ACCELERATE_NUM_PROCESSES:-6}"
export TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"

# Rows whose templated prompt exceeds MAX_PROMPT_LENGTH are dropped, not
# truncated. Measured max on this train file is 10220 tokens, so nothing drops.
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-20000}"
export MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-8000}"
# DAPO Soft Overlong Punishment: L_max must track MAX_COMPLETION_LENGTH or the
# ramp is never reached and the term is a constant 0.
export LENGTH_PENALTY_MAX="${LENGTH_PENALTY_MAX:-${MAX_COMPLETION_LENGTH}}"
export LENGTH_PENALTY_BUFFER="${LENGTH_PENALTY_BUFFER:-512}"

export TEMPERATURE="${TEMPERATURE:-1.2}"
export TOP_P="${TOP_P:-1.0}"
export TOP_K="${TOP_K:-20}"

# Qwen3.8 opens a <think> block on every assistant turn unless disabled.
# Thinking tokens count against MAX_COMPLETION_LENGTH and the length penalty,
# and every validated eval ran with it off. Keep RL consistent with them.
export CHAT_TEMPLATE_KWARGS="${CHAT_TEMPLATE_KWARGS:-{\"enable_thinking\": false, \"preserve_thinking\": false}}"

# DAPO: no gradient from rollouts truncated at MAX_COMPLETION_LENGTH.
export MASK_TRUNCATED_COMPLETIONS="${MASK_TRUNCATED_COMPLETIONS:-1}"
export ASYNC_MAX_TOOL_CALLING_ITERATIONS="${ASYNC_MAX_TOOL_CALLING_ITERATIONS:-8}"

# 6 procs * bs 4 * ga 16 = 384; /16 generations = 24 groups needed per step.
# K=4 oversample -> 96 groups attempted -> 1536 rollouts per optimizer step
# (down from 2688 at bs=2/ga=32/K=7; the global batch of 384 is unchanged).
export NUM_GENERATIONS="${NUM_GENERATIONS:-16}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
export DAPO_OVERSAMPLE_FACTOR="${DAPO_OVERSAMPLE_FACTOR:-4}"
# OOM WATCH: vocab_size is 248320, so completion logits alone are
# bs * 8000 * 248320 * 2 B -- 15.9 GB at bs=4 against 7.9 GB at bs=2, roughly
# doubling again on the fp32 log-softmax and again for backward (~64 GB peak).
# An OOM lands in the log-prob/loss step, not the forward, so gradient
# checkpointing will not help; drop back to bs=2 (with ga=32) if it blows.
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
# Safe at 16 only because EVAL_MODE=reward_only skips the eval loss forward.
export PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-16}"
export EVAL_MODE="${EVAL_MODE:-reward_only}"
export EVAL_LIMIT="${EVAL_LIMIT:-256}"

export LEARNING_RATE="${LEARNING_RATE:-3e-6}"
export BETA_SCHEDULE="${BETA_SCHEDULE:-0:0.005,20:0.001,35:0}"
export EVAL_STEPS="${EVAL_STEPS:-250}"
export SAVE_STEPS="${SAVE_STEPS:-5}"
export SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-1}"
export SAVE_LATEST_FULL_CHECKPOINT="${SAVE_LATEST_FULL_CHECKPOINT:-1}"
# 8 model-only checkpoints at ~51 GB each is ~408 GB, plus one rolling full
# checkpoint (~378 GB with fp32 master and Adam moments) is ~786 GB against the
# 1.6 TB free after clearing the smoke run. save_steps=5 means the oldest is
# evicted every 5 steps, so this is a steady-state ceiling, not growth.
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-8}"

# The stored W&B key is 36 chars and rejected (needs 40+), so wandb would fail
# at init. Set REPORT_TO=wandb once a valid key is in place.
export REPORT_TO="${REPORT_TO:-none}"
export WANDB_PROJECT="${WANDB_PROJECT:-qwen38-27b-bird-gspo}"

export TOOL_LOOP_DEBUG="${TOOL_LOOP_DEBUG:-1}"
export DAPO_DEBUG_ROLLOUTS="${DAPO_DEBUG_ROLLOUTS:-1}"
# No [rollout-sample] spam on the console. Those lines dump 500 chars of raw
# completion text per step and existed to diagnose the un-awaited-coroutine bug,
# which is fixed. The aggregate [rollout-debug] counters stay on -- they are one
# line per step and carry the reward and tool statistics -- and every rollout is
# still written in full to DAPO_ROLLOUT_DUMP_DIR, so nothing is lost for auditing.
export DAPO_DEBUG_ROLLOUT_SAMPLES="${DAPO_DEBUG_ROLLOUT_SAMPLES:-0}"
export DAPO_ROLLOUT_DUMP_DIR="${DAPO_ROLLOUT_DUMP_DIR:-${OUTPUT_DIR}/rollouts}"
mkdir -p "${OUTPUT_DIR}" "${DAPO_ROLLOUT_DUMP_DIR}"

echo "[rl-train] model=${MODEL_NAME} train=${TRAIN_FILE} out=${OUTPUT_DIR}"
echo "[rl-train] ng=${NUM_GENERATIONS} ga=${GRADIENT_ACCUMULATION_STEPS} K=${DAPO_OVERSAMPLE_FACTOR} lr=${LEARNING_RATE}"
echo "[rl-train] rollouts/step=$(( 6 * PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS / NUM_GENERATIONS * DAPO_OVERSAMPLE_FACTOR * NUM_GENERATIONS ))"
bash scripts/launch_train.sh
