#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

PYTHON_BIN="${PYTHON_BIN:-/home/ec2-user/miniconda3/envs/nl2sql312/bin/python}"
PORT="${PORT:-8000}"
NUM_EXAMPLES="${NUM_EXAMPLES:-4}"
CONCURRENCY="${CONCURRENCY:-1}"
TOOL_CHOICE_POLICY="${TOOL_CHOICE_POLICY:-required_first}"
EMPTY_TOOL_RETRIES="${EMPTY_TOOL_RETRIES:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/inference/dev/old-dev-schema-tool-unpatched/Qwen3.6-35B-A3B/smoke4_temp0_openai_tool}"

PYTHONPATH="${PYTHONPATH:-src:.}" "${PYTHON_BIN}" scripts/run_inference_bird_qwen_server.py \
  --server_url "http://127.0.0.1:${PORT}/v1" \
  --model qwen3p6-35b-a3b \
  --input_file outputs/old-dev-schema-tool-unpatched.jsonl \
  --database_dir databases/dev_databases \
  --diff_json_path data/bird_dev_data/raw/bird_dev_unpatched.json \
  --output_dir "${OUTPUT_DIR}" \
  --num_examples "${NUM_EXAMPLES}" \
  --temperature 0.0 \
  --top_p 1.0 \
  --max_new_tokens 8000 \
  --max_tool_rounds 8 \
  --tool_choice_policy "${TOOL_CHOICE_POLICY}" \
  --empty_tool_retries "${EMPTY_TOOL_RETRIES}" \
  --concurrency "${CONCURRENCY}" \
  --overwrite
