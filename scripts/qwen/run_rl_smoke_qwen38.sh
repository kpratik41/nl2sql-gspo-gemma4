#!/usr/bin/env bash
# Fast RL smoke test for Qwen3.8-27B: same code paths as the full run at tiny
# volumes, so bugs surface in minutes instead of after a full rollout round.
# Mirrors scripts/teacher/run_rl_smoke.sh, with the Qwen-specific settings.
#
# Uses VLLM_MODE=server: start scripts/qwen/launch_qwen38_vllm.sh FIRST on GPUs
# disjoint from the trainer's. Set VLLM_MODE=colocate to run vLLM in-process
# instead (no server), at the cost of sharing GPU memory with training.
#
# Prerequisites (see scripts/qwen/README_qwen38_rl.md):
#   - BIRD databases under databases/
#   - Tool-format train/eval JSONL built with the QWEN system prompt
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# Keep .venv/bin on PATH so FlashInfer's ninja-based JIT can run; see
# scripts/qwen/launch_qwen38_vllm.sh for the failure mode this avoids.
export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
export ACCELERATE_BIN="${ACCELERATE_BIN:-${REPO_ROOT}/.venv/bin/accelerate}"
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.8-27B}"

# Qwen emits <tool_call><function=...>, not Gemma's call:name{...}. Set
# explicitly rather than relying on path inference.
export TOOL_DIALECT="${TOOL_DIALECT:-qwen}"
export ENABLE_TOOL_ROLLOUTS="${ENABLE_TOOL_ROLLOUTS:-1}"

# Separate vLLM server (started beforehand) on its own GPUs.
export VLLM_MODE="${VLLM_MODE:-server}"
export VLLM_SERVER_BASE_URL="${VLLM_SERVER_BASE_URL:-http://127.0.0.1:8000}"
# Only used when VLLM_MODE=colocate. 27B bf16 weights are ~55 GB, so each rank's
# in-process vLLM shard must fit alongside its ZeRO-3 training shard.
export VLLM_COLOCATE_TP="${VLLM_COLOCATE_TP:-2}"
export VLLM_COLOCATE_GPU_UTIL="${VLLM_COLOCATE_GPU_UTIL:-0.35}"

export TRAIN_FILE="${TRAIN_FILE:-outputs/train-6601-schema-bare-tool-qwen.jsonl}"
export EVAL_FILE="${EVAL_FILE:-outputs/old-dev-schema-bare-tool-qwen.jsonl}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/rl/_smoke_qwen38}"
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/ds_zero3_bf16_no_scheduler.json}"
# Only consulted when DISTRIBUTED_BACKEND=fsdp; wraps Qwen3_5DecoderLayer.
export FSDP_CONFIG="${FSDP_CONFIG:-configs/fsdp_qwen38_bf16.json}"

# torch SDPA already dispatches to a FlashAttention-2 backend at head_dim=256
# on Hopper. Set flash_attention_2 only if flash-attn is installed.
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"

# GPUs 0-5 train, 6-7 serve rollouts (matches launch_qwen38_vllm.sh's default).
export ACCELERATE_NUM_PROCESSES="${ACCELERATE_NUM_PROCESSES:-6}"
export TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"

# Qwen3.8 context is 262K, but rollout cost scales with these. Keep the smoke
# small; raise for production. Rows whose templated prompt exceeds
# MAX_PROMPT_LENGTH are dropped from the dataset, not truncated.
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-20000}"
export MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-4096}"

# DAPO Soft Overlong Punishment. L_max MUST track MAX_COMPLETION_LENGTH: the
# launcher default is 8000, so with a 4096 completion cap the ramp (starting at
# L_max - L_cache) could never be reached and this reward term would silently be
# a constant 0.
export LENGTH_PENALTY_MAX="${LENGTH_PENALTY_MAX:-${MAX_COMPLETION_LENGTH}}"
export LENGTH_PENALTY_BUFFER="${LENGTH_PENALTY_BUFFER:-512}"

# Sampling. temp 1.2 / top_p 1.0 / top_k 20 is the combination the validated
# Qwen3.8-27B pass@16 runs used; the model card's default is temp 1.0/top_p 0.95.
export TEMPERATURE="${TEMPERATURE:-1.2}"
export TOP_P="${TOP_P:-1.0}"
export TOP_K="${TOP_K:-20}"

# Qwen3.8 opens a <think> block on EVERY assistant turn unless this is set.
# Thinking tokens count against MAX_COMPLETION_LENGTH and the length penalty,
# and the validated evals ran with thinking off. Keep RL consistent with them.
export CHAT_TEMPLATE_KWARGS="${CHAT_TEMPLATE_KWARGS:-{\"enable_thinking\": false, \"preserve_thinking\": false}}"

# DAPO: drop gradient from rollouts truncated at MAX_COMPLETION_LENGTH.
export MASK_TRUNCATED_COMPLETIONS="${MASK_TRUNCATED_COMPLETIONS:-1}"

# Tool rounds per rollout; 8 matches the eval runs.
export ASYNC_MAX_TOOL_CALLING_ITERATIONS="${ASYNC_MAX_TOOL_CALLING_ITERATIONS:-8}"

# Tiny rollout volumes.
# TRL requires (num_processes * per_device_train_batch_size *
# gradient_accumulation_steps) to be divisible by num_generations.
#   6 procs * bs 4 * ga 1 = 24 -> 12 groups/step, x2 oversample = 48 rollouts/step
# If this OOMs, drop PER_DEVICE_TRAIN_BATCH_SIZE to 2 (see README: the failure
# lands in the log-prob/loss step, not the forward, because vocab_size is 248320).
export NUM_GENERATIONS="${NUM_GENERATIONS:-2}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
export DAPO_OVERSAMPLE_FACTOR="${DAPO_OVERSAMPLE_FACTOR:-2}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
# Safe at 16 only because EVAL_MODE=reward_only skips the eval loss/logprob
# forward pass; with a real eval forward this would OOM at 24K-token sequences.
export PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-16}"
export EVAL_MODE="${EVAL_MODE:-reward_only}"
export TRAIN_LIMIT="${TRAIN_LIMIT:-200}"
export EVAL_LIMIT="${EVAL_LIMIT:-4}"

# Force eval, save and every beta-schedule transition inside 4 steps.
export BETA_SCHEDULE="${BETA_SCHEDULE:-0:0.005,2:0.001,3:0}"
export EVAL_STEPS="${EVAL_STEPS:-2}"
export SAVE_STEPS="${SAVE_STEPS:-2}"
export SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-1}"
export SAVE_LATEST_FULL_CHECKPOINT="${SAVE_LATEST_FULL_CHECKPOINT:-0}"
export MAX_STEPS="${MAX_STEPS:-4}"

export LEARNING_RATE="${LEARNING_RATE:-2e-6}"
export REPORT_TO="${REPORT_TO:-none}"
export WANDB_PROJECT="${WANDB_PROJECT:-qwen38-27b-bird-gspo}"

# Surface the tool loop in logs: [tool-parse] must report a non-zero
# attached_native_tool_calls, otherwise rollouts are training without tools.
export TOOL_LOOP_DEBUG="${TOOL_LOOP_DEBUG:-1}"
export DAPO_DEBUG_ROLLOUTS="${DAPO_DEBUG_ROLLOUTS:-1}"

echo "[smoke] model=${MODEL_NAME} dialect=${TOOL_DIALECT} vllm_mode=${VLLM_MODE} attn=${ATTN_IMPLEMENTATION}"
bash scripts/launch_train.sh
