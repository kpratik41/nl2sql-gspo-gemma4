#!/usr/bin/env bash
set -euo pipefail

if ! command -v accelerate >/dev/null 2>&1; then
  if [[ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]]; then
    set +u
    # shellcheck disable=SC1091
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
    conda activate nl2sql312
    set -u
  fi
fi

if ! command -v accelerate >/dev/null 2>&1; then
  echo "accelerate not found on PATH. Activate nl2sql312 before running this launcher." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

# Long collective windows: ZeRO-3 forward over 20K-token prompts can take >>8 min
# on the very first step (cold caches) which would otherwise trigger the default
# NCCL watchdog (480s) and kill all ranks. Bump the heartbeat & collective timeouts.
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}"
export TORCH_NCCL_TIMEOUT_MS="${TORCH_NCCL_TIMEOUT_MS:-3600000}"
# Disable async-error tear-downs so that one slow rank doesn't take the whole job.
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-0}"

# DeepSpeed JIT-compiles ops at import time; needs CUDA_HOME with nvcc.
if [[ -z "${CUDA_HOME:-}" && -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/nvcc" ]]; then
  export CUDA_HOME="${CONDA_PREFIX}"
fi

export WANDB_PROJECT=gemma4-31b-bird-gspo
REPORT_TO="${REPORT_TO:-wandb}"
RUN_NAME="${RUN_NAME:-gemma4-31b-gspo-bird}"
LOGGING_DIR="${LOGGING_DIR:-outputs/gemma4_31b_gspo_bird/tb}"
TRAIN_LIMIT="${TRAIN_LIMIT:--1}"
EVAL_LIMIT="${EVAL_LIMIT:-64}"

MODEL_NAME="google/gemma-4-31B-it"
RESUME_ARGS=()

if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  RESUME_ARGS+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi

# Logging cadence.
SAVE_STEPS="${SAVE_STEPS:-25}"
EVAL_STEPS="${EVAL_STEPS:-25}"
LOGGING_STEPS="${LOGGING_STEPS:-5}"
# Set EVAL_ON_START=0 to skip the pre-training dev baseline (saves ~10min on first step).
if [[ "${EVAL_ON_START:-1}" == "1" ]]; then
  EVAL_ON_START_ARG="--eval_on_start"
else
  EVAL_ON_START_ARG=""
fi

# DAPO oversample-and-replace controls. Each optimizer step keeps
# exactly G heterogeneous groups (G = world_size * per_device_batch *
# steps_per_generation). With per_device_batch=1, world=6, steps_per_gen=1
# this is G=6 heterogeneous prompts per step. Set ENABLE_DYNAMIC_SAMPLING=0
# to skip filtering entirely.
NUM_ITERATIONS="${NUM_ITERATIONS:-1}"
ENABLE_DYNAMIC_SAMPLING="${ENABLE_DYNAMIC_SAMPLING:-1}"
DYNAMIC_SAMPLING_MIN_STD="${DYNAMIC_SAMPLING_MIN_STD:-1e-6}"
DAPO_MAX_ROUNDS="${DAPO_MAX_ROUNDS:-1}"  # max rollout rounds per step (1 = no resampling)
DAPO_OVERSAMPLE_FACTOR="${DAPO_OVERSAMPLE_FACTOR:-4}"  # K: single-shot oversample multiplier (>1 enables single-shot path)
# Optional: judge group heterogeneity by a single reward function (e.g.
# result_reward) instead of the aggregated/normalized advantages. Leave
# unset to use total advantages.
DYNAMIC_SAMPLING_REWARD_NAME="${DYNAMIC_SAMPLING_REWARD_NAME:-result_reward}"
MASK_TRUNCATED_COMPLETIONS="${MASK_TRUNCATED_COMPLETIONS:-1}"

# Reward shaping.
EXEC_TIMEOUT_S="${EXEC_TIMEOUT_S:-60}"
LENGTH_PENALTY_MAX="${LENGTH_PENALTY_MAX:-4096}"
LENGTH_PENALTY_BUFFER="${LENGTH_PENALTY_BUFFER:-512}"
REWARD_WEIGHTS="${REWARD_WEIGHTS:-0.2,0.5,2.0,0.5,0.5,0.1,0.1}"

DAPO_ARGS=(
  --num_iterations "${NUM_ITERATIONS}"
  --dynamic_sampling_min_std "${DYNAMIC_SAMPLING_MIN_STD}"
  --dapo_max_rounds "${DAPO_MAX_ROUNDS}"
  --dapo_oversample_factor "${DAPO_OVERSAMPLE_FACTOR}"
  --exec_timeout_s "${EXEC_TIMEOUT_S}"
  --length_penalty_max "${LENGTH_PENALTY_MAX}"
  --length_penalty_buffer "${LENGTH_PENALTY_BUFFER}"
)
if [[ "${ENABLE_DYNAMIC_SAMPLING}" == "1" ]]; then
  DAPO_ARGS+=(--enable_dynamic_sampling)
fi
if [[ "${MASK_TRUNCATED_COMPLETIONS}" == "1" ]]; then
  DAPO_ARGS+=(--mask_truncated_completions)
fi
if [[ -n "${STEPS_PER_GENERATION:-}" ]]; then
  DAPO_ARGS+=(--steps_per_generation "${STEPS_PER_GENERATION}")
fi
if [[ -n "${DYNAMIC_SAMPLING_REWARD_NAME:-}" ]]; then
  DAPO_ARGS+=(--dynamic_sampling_reward_name "${DYNAMIC_SAMPLING_REWARD_NAME}")
fi

accelerate launch \
  --num_processes 6 \
  --mixed_precision bf16 \
  src/nl2sql_gspo/train_gspo_nl2sql.py \
  --model_name_or_path "${MODEL_NAME}" \
  --train_file outputs/train-6601-schema-filtered.jsonl \
  --eval_file outputs/dev-20251106-schema.jsonl \
  --train_limit "${TRAIN_LIMIT}" \
  --eval_limit "${EVAL_LIMIT}" \
  --database_dir databases \
  --output_dir outputs/gemma4_31b_gspo_bird \
  --vllm_server_base_url http://127.0.0.1:8000 \
  --max_prompt_length 20000 \
  --max_completion_length 4096 \
  --num_generations 16 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 5e-7 \
  --num_train_epochs 1 \
  --reward_weights "${REWARD_WEIGHTS}" \
  --report_to "${REPORT_TO}" \
  --run_name "${RUN_NAME}" \
  --logging_dir "${LOGGING_DIR}" \
  --deepspeed configs/ds_zero3_bf16.json \
  --logging_steps "${LOGGING_STEPS}" \
  --save_steps "${SAVE_STEPS}" \
  --eval_steps "${EVAL_STEPS}" \
  ${EVAL_ON_START_ARG:+$EVAL_ON_START_ARG} \
  --loss_type dapo \
  --scale_rewards batch \
  --beta 0.0 \
  --epsilon 0.2 \
  --epsilon_high 0.28 \
  "${DAPO_ARGS[@]}" \
  "${RESUME_ARGS[@]}"