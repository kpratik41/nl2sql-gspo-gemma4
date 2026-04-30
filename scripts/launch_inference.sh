#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${INFERENCE_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

MODEL_PATH="${MODEL_PATH:-outputs/gemma4_31b_gspo_bird}"
INPUT_FILE="${INPUT_FILE:-outputs/dev-20251106-schema.jsonl}"
DATABASE_DIR="${DATABASE_DIR:-databases/dev_databases}"
DIFF_JSON_PATH="${DIFF_JSON_PATH:-data/bird_dev_data/raw/dev_20251106.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/bird_dev_inference}"

c:/Users/kprat/OneDrive/Desktop/Project/nl2sql-gspo-gemma4/.venv/Scripts/python.exe \
  scripts/run_inference_bird.py \
  --model_name_or_path "${MODEL_PATH}" \
  --input_file "${INPUT_FILE}" \
  --database_dir "${DATABASE_DIR}" \
  --diff_json_path "${DIFF_JSON_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_prompt_length 16384 \
  --max_new_tokens 512 \
  --overwrite