#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p logs

MODEL_PATH="${MODEL_PATH:-google/gemma-4-31B-it}"
POLL_SECONDS="${POLL_SECONDS:-60}"
FREE_MEMORY_THRESHOLD_MB="${FREE_MEMORY_THRESHOLD_MB:-2000}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-34000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8000}"
MAX_TOOL_ROUNDS="${MAX_TOOL_ROUNDS:-8}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-43000}"
VLLM_ASYNC_CONCURRENCY="${VLLM_ASYNC_CONCURRENCY:-16}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"

job_names=(
  "bird-dev-schema-tool"
)

input_files=(
  "outputs/bird_dev-schema-tool.jsonl"
)

output_dirs=(
  "outputs/inference/dev/bird_dev-schema-tool/google-gemma-4-31B-it/vllm_async_tp1_c16_ctx43k_p34k_o8k_r8_temp0"
)

log_files=(
  "logs/infer_31b_bird_dev-schema-tool_tp1_temp0.log"
)

declare -a running_pids=()
declare -a running_gpus=()
declare -a running_names=()
declare -A reserved_gpus=()
next_job=0
total_jobs="${#job_names[@]}"

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

refresh_running() {
  local new_pids=()
  local new_gpus=()
  local new_names=()

  for i in "${!running_pids[@]}"; do
    local pid="${running_pids[$i]}"
    local gpu="${running_gpus[$i]}"
    local name="${running_names[$i]}"
    if kill -0 "${pid}" 2>/dev/null; then
      new_pids+=("${pid}")
      new_gpus+=("${gpu}")
      new_names+=("${name}")
      reserved_gpus["${gpu}"]=1
    else
      echo "[$(timestamp)] completed name=${name} gpu=${gpu} pid=${pid}"
      unset 'reserved_gpus[$gpu]'
    fi
  done

  running_pids=("${new_pids[@]}")
  running_gpus=("${new_gpus[@]}")
  running_names=("${new_names[@]}")
}

free_gpu() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
    while IFS=',' read -r gpu mem_used; do
      gpu="$(echo "${gpu}" | xargs)"
      mem_used="$(echo "${mem_used}" | xargs)"
      if [[ -n "${reserved_gpus[$gpu]:-}" ]]; then
        continue
      fi
      if [[ "${mem_used}" -le "${FREE_MEMORY_THRESHOLD_MB}" ]]; then
        echo "${gpu}"
        return 0
      fi
    done
}

launch_job() {
  local job_index="$1"
  local gpu="$2"
  local name="${job_names[$job_index]}"
  local input_file="${input_files[$job_index]}"
  local output_dir="${output_dirs[$job_index]}"
  local log_file="${log_files[$job_index]}"

  echo "[$(timestamp)] launching name=${name} gpu=${gpu} input=${input_file} output=${output_dir} log=${log_file}"
  (
    INFERENCE_CUDA_VISIBLE_DEVICES="${gpu}" \
    PYTHON_BIN=".venv/bin/python" \
    MODEL_PATH="${MODEL_PATH}" \
    INPUT_FILE="${input_file}" \
    OUTPUT_DIR="${output_dir}" \
    APPEND_OUTPUT_TIMESTAMP=0 \
    NUM_EXAMPLES=-1 \
    MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH}" \
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
    MAX_TOOL_ROUNDS="${MAX_TOOL_ROUNDS}" \
    TEMPERATURE=0.0 \
    TOP_P=1.0 \
    VLLM_TENSOR_PARALLEL_SIZE=1 \
    VLLM_ASYNC_CONCURRENCY="${VLLM_ASYNC_CONCURRENCY}" \
    VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION}" \
    VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN}" \
    bash scripts/launch_inference.sh
  ) > "${log_file}" 2>&1 &

  local pid="$!"
  running_pids+=("${pid}")
  running_gpus+=("${gpu}")
  running_names+=("${name}")
  reserved_gpus["${gpu}"]=1
}

echo "[$(timestamp)] queue started model=${MODEL_PATH} total_jobs=${total_jobs} poll_seconds=${POLL_SECONDS} free_mem_threshold_mb=${FREE_MEMORY_THRESHOLD_MB}"

while [[ "${next_job}" -lt "${total_jobs}" || "${#running_pids[@]}" -gt 0 ]]; do
  refresh_running

  while [[ "${next_job}" -lt "${total_jobs}" ]]; do
    gpu="$(free_gpu || true)"
    if [[ -z "${gpu:-}" ]]; then
      break
    fi
    launch_job "${next_job}" "${gpu}"
    next_job=$((next_job + 1))
  done

  if [[ "${next_job}" -lt "${total_jobs}" || "${#running_pids[@]}" -gt 0 ]]; then
    echo "[$(timestamp)] queue status next_job=${next_job}/${total_jobs} running=${#running_pids[@]}"
    sleep "${POLL_SECONDS}"
  fi
done

echo "[$(timestamp)] queue complete"
