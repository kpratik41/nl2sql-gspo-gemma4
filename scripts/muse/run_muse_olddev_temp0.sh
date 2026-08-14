#!/usr/bin/env bash
# Temp-0 BIRD old-dev eval against a running Muse-Glimmer vLLM server.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

export PYTHONPATH=".:src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

PYTHON_BIN="${PYTHON_BIN:-/home/ec2-user/miniconda3/envs/nl2sql312/bin/python}"
SERVER_URL="${SERVER_URL:-http://127.0.0.1:8000/v1}"
INPUT_FILE="${INPUT_FILE:-outputs/old-dev-schema-tool-unpatched.jsonl}"
DATABASE_DIR="${DATABASE_DIR:-databases/dev_databases}"
DIFF_JSON_PATH="${DIFF_JSON_PATH:-data/bird_dev_data/raw/bird_dev_unpatched.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/inference/dev/old-dev-schema-tool-unpatched/Muse-Glimmer-30B/temp0_openai_tool}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:--1}"
NUM_EXAMPLES="${NUM_EXAMPLES:--1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8000}"
MAX_TOOL_ROUNDS="${MAX_TOOL_ROUNDS:-16}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
REASONING_STRENGTH="${REASONING_STRENGTH:-high}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-600}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-60}"
EVAL_WORKERS="${EVAL_WORKERS:-16}"
CONCURRENCY="${CONCURRENCY:-4}"

"${PYTHON_BIN}" scripts/run_inference_bird_muse_server.py \
  --server_url "${SERVER_URL}" \
  --model muse-glimmer \
  --input_file "${INPUT_FILE}" \
  --database_dir "${DATABASE_DIR}" \
  --diff_json_path "${DIFF_JSON_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --start_index "${START_INDEX}" \
  --end_index "${END_INDEX}" \
  --num_examples "${NUM_EXAMPLES}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --max_tool_rounds "${MAX_TOOL_ROUNDS}" \
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}" \
  --request_timeout "${REQUEST_TIMEOUT}" \
  --eval_timeout "${EVAL_TIMEOUT}" \
  --eval_workers "${EVAL_WORKERS}" \
  --concurrency "${CONCURRENCY}" \
  --reasoning_strength "${REASONING_STRENGTH}" \
  --overwrite
