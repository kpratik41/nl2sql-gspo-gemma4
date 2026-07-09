#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

BASE="/home/ec2-user/consensus/nl2sql-gspo-gemma4/outputs/training/train-6601-schema-bare-tool/gemma-4-E4B-it/grpo_deepspeed_p15500_c8000_g16_t1p2_bs4_ga8_lr2e-6_e4b_bare_lr2e6_20260524_130108"
PYBIN="/home/ec2-user/miniconda3/envs/nl2sql312/bin/python"
DBDIR="/home/ec2-user/nl2sql-gspo-gemma4/databases/dev_databases"
INPUT_FILE="outputs/dev-20251106-schema-tool.jsonl"
DIFF_JSON_PATH="data/bird_dev_data/raw/dev_20251106.json"
RUN_DATE="20260525"
PIDFILE="logs/infer_e4b_bare_tool_full_${RUN_DATE}.pids"

mkdir -p logs
: > "${PIDFILE}"

launch_one() {
  local gpu="$1"
  local ckpt="$2"
  local model="$3"
  local out_dir="${BASE}/${ckpt}"
  local log_file="logs/infer_e4b_bare_tool_${ckpt}_gpu${gpu}_full_${RUN_DATE}.log"

  echo "[launcher] launching ${ckpt} gpu=${gpu} model=${model} output=${out_dir} log=${log_file}"

  setsid env \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    TOKENIZERS_PARALLELISM=false \
    PYTHONPATH="${PWD}/src:${PYTHONPATH:-}" \
    PYTHONUNBUFFERED=1 \
    "${PYBIN}" scripts/run_inference_bird.py \
      --inference_backend vllm_async \
      --model_name_or_path "${model}" \
      --input_file "${INPUT_FILE}" \
      --database_dir "${DBDIR}" \
      --diff_json_path "${DIFF_JSON_PATH}" \
      --output_dir "${out_dir}" \
      --num_examples -1 \
      --max_prompt_length 35000 \
      --max_new_tokens 8000 \
      --max_tool_rounds 8 \
      --temperature 0.0 \
      --top_p 1.0 \
      --eval_timeout 60 \
      --eval_workers 16 \
      --overwrite \
      --vllm_tensor_parallel_size 1 \
      --vllm_data_parallel_size 1 \
      --vllm_max_model_len 44000 \
      --vllm_gpu_memory_utilization 0.93 \
      --vllm_async_concurrency 16 \
    > "${log_file}" 2>&1 &

  echo "$! ${ckpt} gpu=${gpu} ${log_file}" >> "${PIDFILE}"
}

launch_one 0 checkpoint-0 "google/gemma-4-E4B-it"
launch_one 1 checkpoint-10 "${BASE}/checkpoint-10"
launch_one 2 checkpoint-20 "${BASE}/checkpoint-20"
launch_one 3 checkpoint-30 "${BASE}/checkpoint-30"
launch_one 4 checkpoint-40 "${BASE}/checkpoint-40"
launch_one 5 checkpoint-50 "${BASE}/checkpoint-50"
launch_one 6 checkpoint-60 "${BASE}/checkpoint-60"

cat "${PIDFILE}"
