#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${ACCELERATE_BIN:-}" && -x "${REPO_ROOT}/.conda/nl2sql312/bin/accelerate" ]]; then
  ACCELERATE_BIN="${REPO_ROOT}/.conda/nl2sql312/bin/accelerate"
fi

ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
if ! command -v "${ACCELERATE_BIN}" >/dev/null 2>&1 && [[ ! -x "${ACCELERATE_BIN}" ]]; then
  if [[ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]]; then
    set +u
    # shellcheck disable=SC1091
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
    conda activate nl2sql312
    set -u
  elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    set +u
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate nl2sql312
    set -u
  fi
fi

if ! command -v "${ACCELERATE_BIN}" >/dev/null 2>&1 && [[ ! -x "${ACCELERATE_BIN}" ]]; then
  echo "accelerate not found on PATH. Activate nl2sql312 before running this launcher." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}}"
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Long collective windows: ZeRO-3 forward over 20K-token prompts can take >>8 min
# on the very first step (cold caches) which would otherwise trigger the default
# NCCL watchdog (480s) and kill all ranks. Bump the heartbeat & collective timeouts.
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}"
export TORCH_NCCL_TIMEOUT_MS="${TORCH_NCCL_TIMEOUT_MS:-3600000}"
# Disable async-error tear-downs so that one slow rank doesn't take the whole job.
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-0}"

RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "${LOG_DIR}"
TRAIN_LOG_FILE="${TRAIN_LOG_FILE:-${LOG_DIR}/train_${RUN_TIMESTAMP}.log}"
echo "[launch_train] writing launcher log to ${TRAIN_LOG_FILE}"
exec > >(tee -a "${TRAIN_LOG_FILE}") 2>&1

# DeepSpeed JIT-compiles ops at import time; needs CUDA_HOME with nvcc.
if [[ -z "${CUDA_HOME:-}" && -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/nvcc" ]]; then
  export CUDA_HOME="${CONDA_PREFIX}"
fi

# Some environments ship PyTorch + CUDA runtime without nvcc. In that case,
# DeepSpeed's op compatibility probe can fail during import before training
# starts. Disable optional op builds unless the caller explicitly overrides it.
if [[ -z "${CUDA_HOME:-}" && -z "${DS_BUILD_OPS:-}" ]]; then
  export DS_BUILD_OPS=0
fi
if [[ -z "${CUDA_HOME:-}" && -z "${DS_IGNORE_CUDA_DETECTION:-}" ]]; then
  export DS_IGNORE_CUDA_DETECTION=1
fi

export WANDB_PROJECT=gemma4-31b-bird-gspo
REPORT_TO="${REPORT_TO:-wandb}"
TRAIN_LIMIT="${TRAIN_LIMIT:--1}"
EVAL_LIMIT="${EVAL_LIMIT:-32}"
TRAIN_FILE="${TRAIN_FILE:-outputs/train-6601-schema-tool.jsonl}"
EVAL_FILE="${EVAL_FILE:-outputs/dev-20251106-schema-tool.jsonl}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-15500}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-8000}"
NUM_GENERATIONS="${NUM_GENERATIONS:-16}"
TEMPERATURE="${TEMPERATURE:-1.2}"
TOP_P="${TOP_P:-0.95}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-3}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
LEARNING_RATE="${LEARNING_RATE:-${LR:-1e-6}}"
MAX_STEPS="${MAX_STEPS:--1}"
ACCELERATE_NUM_PROCESSES="${ACCELERATE_NUM_PROCESSES:-6}"

MODEL_NAME="${MODEL_NAME:-google/gemma-4-31B-it}"
USER_OUTPUT_DIR="${OUTPUT_DIR:-}"
VLLM_GROUP_PORT="${VLLM_GROUP_PORT:-29600}"
VLLM_SERVER_BASE_URL="${VLLM_SERVER_BASE_URL:-http://127.0.0.1:8000}"
TRAINER_BACKEND="${TRAINER_BACKEND:-grpo}"
if [[ -z "${DISTRIBUTED_BACKEND:-}" ]]; then
  if [[ "${TRAINER_BACKEND}" == "async_grpo" ]]; then
    DISTRIBUTED_BACKEND="fsdp"
  else
    DISTRIBUTED_BACKEND="deepspeed"
  fi
fi
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/ds_zero3_bf16.json}"
FSDP="${FSDP:-full_shard auto_wrap}"
FSDP_CONFIG="${FSDP_CONFIG:-configs/fsdp_gemma4_bf16.json}"
RESUME_ARGS=()

sanitize_path_part() {
  local value="$1"
  value="${value##*/}"
  value="${value%.*}"
  value="$(printf '%s' "${value}" | tr -cs '[:alnum:]_.-' '_')"
  value="${value##_}"
  value="${value%%_}"
  if [[ -z "${value}" ]]; then
    value="unknown"
  fi
  printf '%s' "${value}"
}

format_number_tag() {
  local value="$1"
  printf '%s' "${value}" | sed -E 's/^-?([0-9]+)e-([0-9]+)$/\1e-\2/; s/\./p/g'
}

build_default_output_dir() {
  local train_tag model_tag lr_tag temp_tag run_tag

  train_tag="$(sanitize_path_part "${TRAIN_FILE}")"
  model_tag="$(sanitize_path_part "${MODEL_NAME}")"
  lr_tag="$(format_number_tag "${LEARNING_RATE}")"
  temp_tag="$(format_number_tag "${TEMPERATURE}")"
  run_tag="${TRAINER_BACKEND}_${DISTRIBUTED_BACKEND}_p${MAX_PROMPT_LENGTH}_c${MAX_COMPLETION_LENGTH}"
  run_tag="${run_tag}_g${NUM_GENERATIONS}_t${temp_tag}_bs${PER_DEVICE_TRAIN_BATCH_SIZE}"
  run_tag="${run_tag}_ga${GRADIENT_ACCUMULATION_STEPS}_lr${lr_tag}_${RUN_TIMESTAMP}"

  printf 'outputs/training/%s/%s/%s' "${train_tag}" "${model_tag}" "${run_tag}"
}

if [[ -n "${USER_OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${USER_OUTPUT_DIR}"
else
  OUTPUT_DIR="$(build_default_output_dir)"
fi

RUN_NAME="${RUN_NAME:-$(sanitize_path_part "${TRAIN_FILE}")-$(sanitize_path_part "${OUTPUT_DIR}")}"
LOGGING_DIR="${LOGGING_DIR:-${OUTPUT_DIR}/tb/${RUN_TIMESTAMP}}"
echo "[launch_train] output_dir=${OUTPUT_DIR}"
echo "[launch_train] run_name=${RUN_NAME}"
echo "[launch_train] logging_dir=${LOGGING_DIR}"

is_model_only_checkpoint() {
  local checkpoint_dir="$1"
  [[ -d "$checkpoint_dir" ]] || return 1
  [[ -f "$checkpoint_dir/trainer_state.json" ]] || return 1
  [[ -f "$checkpoint_dir/model.safetensors.index.json" || -f "$checkpoint_dir/model-00001-of-00002.safetensors" || -f "$checkpoint_dir/pytorch_model.bin" ]] || return 1
  [[ -f "$checkpoint_dir/optimizer.pt" || -f "$checkpoint_dir/zero_to_fp32.py" || -f "$checkpoint_dir/latest" ]] && return 1
  compgen -G "$checkpoint_dir/global_step*" >/dev/null && return 1
  return 0
}

if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  if is_model_only_checkpoint "${RESUME_FROM_CHECKPOINT}"; then
    echo "RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT} looks like a model-only checkpoint without DeepSpeed optimizer state." >&2
    echo "For model-only continuation, set MODEL_NAME=${RESUME_FROM_CHECKPOINT} and leave RESUME_FROM_CHECKPOINT unset." >&2
    exit 1
  fi
  RESUME_ARGS+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi

# Logging cadence.
SAVE_STEPS="${SAVE_STEPS:-10}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-10}"
SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-1}"
SAVE_LATEST_FULL_CHECKPOINT="${SAVE_LATEST_FULL_CHECKPOINT:-1}"
LATEST_FULL_CHECKPOINT_DIR_NAME="${LATEST_FULL_CHECKPOINT_DIR_NAME:-latest-full-checkpoint}"
EVAL_STEPS="${EVAL_STEPS:-10}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
LOG_COMPLETIONS="${LOG_COMPLETIONS:-0}"
NUM_COMPLETIONS_TO_PRINT="${NUM_COMPLETIONS_TO_PRINT:-0}"
EVAL_MODE="${EVAL_MODE:-reward_only}"
REWARD_ONLY_EVAL="${REWARD_ONLY_EVAL:-}"
if [[ -z "${REWARD_ONLY_EVAL}" ]]; then
  if [[ "${EVAL_MODE}" == "reward_only" || "${EVAL_MODE}" == "reward-only" ]]; then
    REWARD_ONLY_EVAL=1
  else
    REWARD_ONLY_EVAL=0
  fi
fi
export DAPO_DEBUG_ROLLOUTS="${DAPO_DEBUG_ROLLOUTS:-0}"
export TOOL_LOOP_DEBUG="${TOOL_LOOP_DEBUG:-0}"
# Set EVAL_ON_START=1 to run the pre-training dev baseline.
if [[ "${EVAL_ON_START:-0}" == "1" ]]; then
  EVAL_ON_START_ARG="--eval_on_start"
else
  EVAL_ON_START_ARG=""
fi
if [[ "${LOG_COMPLETIONS}" == "1" ]]; then
  LOG_COMPLETIONS_ARGS=(--log_completions --num_completions_to_print "${NUM_COMPLETIONS_TO_PRINT}")
else
  LOG_COMPLETIONS_ARGS=()
fi
if [[ "${REWARD_ONLY_EVAL}" == "1" ]]; then
  REWARD_ONLY_EVAL_ARGS=(--reward_only_eval)
else
  REWARD_ONLY_EVAL_ARGS=()
fi
echo "[launch_train] eval_mode=${EVAL_MODE} reward_only_eval=${REWARD_ONLY_EVAL}"
if [[ "${SAVE_ONLY_MODEL}" == "1" ]]; then
  SAVE_ONLY_MODEL_ARGS=(--save_only_model)
else
  SAVE_ONLY_MODEL_ARGS=()
fi
if [[ "${SAVE_LATEST_FULL_CHECKPOINT}" == "1" ]]; then
  SAVE_LATEST_FULL_CHECKPOINT_ARGS=(
    --save_latest_full_checkpoint
    --latest_full_checkpoint_dir_name "${LATEST_FULL_CHECKPOINT_DIR_NAME}"
  )
else
  SAVE_LATEST_FULL_CHECKPOINT_ARGS=()
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
DAPO_OVERSAMPLE_FACTOR="${DAPO_OVERSAMPLE_FACTOR:-12}"  # K: single-shot oversample multiplier (>1 enables single-shot path)
# Optional: judge group heterogeneity by a single reward function (e.g.
# result_reward) instead of the aggregated/normalized advantages. Leave
# unset to use total advantages.
DYNAMIC_SAMPLING_REWARD_NAME="${DYNAMIC_SAMPLING_REWARD_NAME:-result_reward}"
MASK_TRUNCATED_COMPLETIONS="${MASK_TRUNCATED_COMPLETIONS:-0}"

# Reward shaping.
EXEC_TIMEOUT_S="${EXEC_TIMEOUT_S:-60}"
REWARD_WORKERS="${REWARD_WORKERS:-4}"
LENGTH_PENALTY_MAX="${LENGTH_PENALTY_MAX:-8000}"
LENGTH_PENALTY_BUFFER="${LENGTH_PENALTY_BUFFER:-512}"
REWARD_WEIGHTS="${REWARD_WEIGHTS:-0.2,0.5,2.0,0.5,0.5,0.1,0.1}"
ENABLE_TOOL_ROLLOUTS="${ENABLE_TOOL_ROLLOUTS:-1}"
TOOL_DB_EXTRA_ROOTS="${TOOL_DB_EXTRA_ROOTS:-}"
ASYNC_MAX_TOOL_CALLING_ITERATIONS="${ASYNC_MAX_TOOL_CALLING_ITERATIONS:-8}"
ASYNC_REQUEST_TIMEOUT="${ASYNC_REQUEST_TIMEOUT:-600}"
ASYNC_MAX_INFLIGHT_TASKS="${ASYNC_MAX_INFLIGHT_TASKS:--1}"
ASYNC_MAX_STALENESS="${ASYNC_MAX_STALENESS:-4}"
ASYNC_QUEUE_MAXSIZE="${ASYNC_QUEUE_MAXSIZE:-1024}"
ASYNC_WEIGHT_SYNC_STEPS="${ASYNC_WEIGHT_SYNC_STEPS:-1}"

DAPO_ARGS=(
  --trainer_backend "${TRAINER_BACKEND}"
  --distributed_backend "${DISTRIBUTED_BACKEND}"
  --fsdp "${FSDP}"
  --fsdp_config "${FSDP_CONFIG}"
  --async_max_tool_calling_iterations "${ASYNC_MAX_TOOL_CALLING_ITERATIONS}"
  --async_request_timeout "${ASYNC_REQUEST_TIMEOUT}"
  --async_max_inflight_tasks "${ASYNC_MAX_INFLIGHT_TASKS}"
  --async_max_staleness "${ASYNC_MAX_STALENESS}"
  --async_queue_maxsize "${ASYNC_QUEUE_MAXSIZE}"
  --async_weight_sync_steps "${ASYNC_WEIGHT_SYNC_STEPS}"
  --num_iterations "${NUM_ITERATIONS}"
  --dynamic_sampling_min_std "${DYNAMIC_SAMPLING_MIN_STD}"
  --dapo_max_rounds "${DAPO_MAX_ROUNDS}"
  --dapo_oversample_factor "${DAPO_OVERSAMPLE_FACTOR}"
  --exec_timeout_s "${EXEC_TIMEOUT_S}"
  --reward_workers "${REWARD_WORKERS}"
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
if [[ "${ENABLE_TOOL_ROLLOUTS}" == "1" ]]; then
  DAPO_ARGS+=(--enable_tool_rollouts)
fi
if [[ -n "${TOOL_DB_EXTRA_ROOTS}" ]]; then
  DAPO_ARGS+=(--tool_db_extra_roots "${TOOL_DB_EXTRA_ROOTS}")
fi

"${ACCELERATE_BIN}" launch \
  --num_processes "${ACCELERATE_NUM_PROCESSES}" \
  --mixed_precision bf16 \
  src/nl2sql_gspo/train_gspo_nl2sql.py \
  --model_name_or_path "${MODEL_NAME}" \
  --train_file "${TRAIN_FILE}" \
  --eval_file "${EVAL_FILE}" \
  --train_limit "${TRAIN_LIMIT}" \
  --eval_limit "${EVAL_LIMIT}" \
  --database_dir databases \
  --output_dir "${OUTPUT_DIR}" \
  --vllm_server_base_url "${VLLM_SERVER_BASE_URL}" \
  --vllm_group_port "${VLLM_GROUP_PORT}" \
  --max_prompt_length "${MAX_PROMPT_LENGTH}" \
  --max_completion_length "${MAX_COMPLETION_LENGTH}" \
  --num_generations "${NUM_GENERATIONS}" \
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --learning_rate "${LEARNING_RATE}" \
  --num_train_epochs 1 \
  --max_steps "${MAX_STEPS}" \
  --reward_weights "${REWARD_WEIGHTS}" \
  --report_to "${REPORT_TO}" \
  --run_name "${RUN_NAME}" \
  --logging_dir "${LOGGING_DIR}" \
  --deepspeed "${DEEPSPEED_CONFIG}" \
  --logging_steps "${LOGGING_STEPS}" \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  "${SAVE_ONLY_MODEL_ARGS[@]}" \
  "${SAVE_LATEST_FULL_CHECKPOINT_ARGS[@]}" \
  --eval_steps "${EVAL_STEPS}" \
  ${EVAL_ON_START_ARG:+$EVAL_ON_START_ARG} \
  "${REWARD_ONLY_EVAL_ARGS[@]}" \
  "${LOG_COMPLETIONS_ARGS[@]}" \
  --loss_type dapo \
  --scale_rewards batch \
  --beta 0.0 \
  --epsilon 0.2 \
  --epsilon_high 0.28 \
  "${DAPO_ARGS[@]}" \
  "${RESUME_ARGS[@]}"
