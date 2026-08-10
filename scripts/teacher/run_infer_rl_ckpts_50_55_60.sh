#!/usr/bin/env bash
# Temp-0 async-vLLM inference for RL checkpoints 50/55/60 on the unpatched
# old-dev tool set. Three jobs run concurrently at tp=2 on GPU pairs (0,1),
# (2,3), (4,5); GPUs 6-7 are left to the RL rollout server.
set -euo pipefail

cd "$(dirname "$0")/../.."

BASE="outputs/rl/gemma4_31b_sftckpt20_dapo10_lr2e6"
PYBIN="/home/ubuntu/nl2sql-gspo-gemma4/.venv/bin/python"
INPUT_FILE="outputs/old-dev-schema-tool-unpatched.jsonl"
DATABASE_DIR="databases/dev_databases"
DIFF_JSON_PATH="data/bird_dev_data/raw/bird_dev_unpatched.json"
RUN_TAG="vllm_async_tp2_dp1_c16_ctx43k_p34k_o8k_r8_temp0"
OUT_ROOT="outputs/inference/dev/old-dev-schema-tool-unpatched/rl_sftckpt20_dapo10_lr2e6"
PIDFILE="logs/infer_rl_ckpts_50_55_60.pids"

mkdir -p logs
: > "${PIDFILE}"

launch_one() {
  local gpus="$1"
  local ckpt="$2"
  local model="${BASE}/${ckpt}"
  local out_dir="${OUT_ROOT}/${ckpt}/${RUN_TAG}"
  local log_file="logs/infer_rl_${ckpt}_tp2_temp0.log"

  if [[ ! -d "${model}" ]]; then
    echo "[launcher] missing checkpoint: ${model}" >&2
    exit 1
  fi

  echo "[launcher] launching ${ckpt} gpus=${gpus} model=${model} output=${out_dir} log=${log_file}"

  setsid env \
    INFERENCE_CUDA_VISIBLE_DEVICES="${gpus}" \
    PYTHON_BIN="${PYBIN}" \
    PYTHONUNBUFFERED=1 \
    INFERENCE_BACKEND="vllm_async" \
    MODEL_PATH="${model}" \
    INPUT_FILE="${INPUT_FILE}" \
    DATABASE_DIR="${DATABASE_DIR}" \
    DIFF_JSON_PATH="${DIFF_JSON_PATH}" \
    OUTPUT_DIR="${out_dir}" \
    APPEND_OUTPUT_TIMESTAMP=0 \
    NUM_EXAMPLES=-1 \
    MAX_PROMPT_LENGTH=34000 \
    MAX_NEW_TOKENS=8000 \
    MAX_TOOL_ROUNDS=8 \
    TEMPERATURE=0.0 \
    TOP_P=1.0 \
    VLLM_TENSOR_PARALLEL_SIZE=2 \
    VLLM_DATA_PARALLEL_SIZE=1 \
    VLLM_ASYNC_CONCURRENCY=16 \
    VLLM_GPU_MEMORY_UTILIZATION=0.93 \
    VLLM_MAX_MODEL_LEN=43000 \
    EVAL_TIMEOUT=60 \
    EVAL_WORKERS=16 \
    bash scripts/launch_inference.sh \
    > "${log_file}" 2>&1 &

  echo "$! ${ckpt} gpus=${gpus} ${log_file}" >> "${PIDFILE}"
}

launch_one "0,1" checkpoint-50
launch_one "2,3" checkpoint-55
launch_one "4,5" checkpoint-60

cat "${PIDFILE}"
