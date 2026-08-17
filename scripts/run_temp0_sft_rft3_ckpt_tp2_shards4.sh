#!/usr/bin/env bash
set -euo pipefail

# Temp0 BIRD-dev inference for a gemma4-31b sft_rft3 checkpoint, async vLLM,
# tp=2 across 4 shards (8 GPUs). Checkpoint step comes from $1 or CKPT_STEP.
#
#   ./scripts/run_temp0_sft_rft3_ckpt_tp2_shards4.sh 50

cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="${PWD}"

CKPT_STEP="${1:-${CKPT_STEP:?usage: run_temp0_sft_rft3_ckpt_tp2_shards4.sh <checkpoint_step>}}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/outputs/sft_rft3/gemma4_31b_sftckpt56_dapo12_lr2e6_beta_0p005_0p001_0}"
CKPT="${CKPT:-${RUN_ROOT}/checkpoint-${CKPT_STEP}}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-${ROOT}/gemma-4-31b-it-local}"
RUN_TAG="${RUN_TAG:-temp0_olddev_schema_tool_unpatched_vllm_async_tp2_shards4_ctx43k}"
OUTPUT_DIR="${OUTPUT_DIR:-${CKPT}/${RUN_TAG}}"
LOG="${LOG:-${CKPT}/${RUN_TAG}.log}"

PYTHON_BIN="${PYTHON_BIN:-/home/ec2-user/miniconda3/envs/nl2sql312/bin/python}"
INPUT_FILE="${INPUT_FILE:-${ROOT}/outputs/old-dev-schema-tool-unpatched.jsonl}"
DATABASE_DIR="${DATABASE_DIR:-${ROOT}/databases/dev_databases}"
DIFF_JSON_PATH="${DIFF_JSON_PATH:-${ROOT}/data/bird_dev_data/raw/bird_dev_unpatched.json}"

NUM_EXAMPLES="${NUM_EXAMPLES:--1}"
TP="${TP:-2}"
SHARDS="${SHARDS:-4}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
CONCURRENCY_PER_SHARD="${CONCURRENCY_PER_SHARD:-16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-43000}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-34000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8000}"
MAX_TOOL_ROUNDS="${MAX_TOOL_ROUNDS:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.93}"
IDLE_MEMORY_MB="${IDLE_MEMORY_MB:-5000}"

GPU_GROUPS=("0,1" "2,3" "4,5" "6,7")

mkdir -p "${OUTPUT_DIR}"
exec > >(tee -a "${LOG}") 2>&1

echo "[$(date -Is)] queued Gemma checkpoint-${CKPT_STEP} temp0 inference"
echo "[$(date -Is)] branch=$(git rev-parse --abbrev-ref HEAD) commit=$(git rev-parse --short HEAD)"
echo "[$(date -Is)] ckpt=${CKPT}"
echo "[$(date -Is)] input=${INPUT_FILE}"
echo "[$(date -Is)] output=${OUTPUT_DIR}"

echo "[$(date -Is)] verifying checkpoint shards are fully written"
"${PYTHON_BIN}" - "${CKPT}" <<'PY'
import json
import sys
from pathlib import Path

ckpt = Path(sys.argv[1])
index = json.loads((ckpt / "model.safetensors.index.json").read_text())
expected = int(index["metadata"]["total_size"])
files = sorted(set(index["weight_map"].values()))
actual = 0
for name in files:
    path = ckpt / name
    if not path.exists():
        raise SystemExit(f"missing shard {path}")
    actual += path.stat().st_size
# Each safetensors file adds an 8-byte header length plus its JSON header on top
# of the tensor bytes counted by index total_size, so actual >= expected.
if actual < expected:
    raise SystemExit(f"incomplete checkpoint: bytes={actual} index_total={expected}")
print(f"checkpoint complete: files={len(files)} bytes={actual} index_total={expected}")
PY
echo "[$(date -Is)] checkpoint complete"

if [[ ! -f "${CKPT}/processor_config.json" ]]; then
  cp "${BASE_MODEL_DIR}/processor_config.json" "${CKPT}/processor_config.json"
fi
echo "[$(date -Is)] installed processor_config=${CKPT}/processor_config.json"

max_gpu_memory_used_mb() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk 'BEGIN{m=0} {if ($1+0>m) m=$1+0} END{print m}'
}

echo "[$(date -Is)] waiting for GPUs to be idle; threshold=${IDLE_MEMORY_MB} MiB"
while true; do
  used="$(max_gpu_memory_used_mb)"
  if [[ "${used}" -lt "${IDLE_MEMORY_MB}" ]]; then
    echo "[$(date -Is)] GPUs idle enough; max_used=${used} MiB"
    break
  fi
  echo "[$(date -Is)] GPUs busy; max_used=${used} MiB"
  sleep 120
done

echo "[$(date -Is)] starting async vLLM sharded inference"
echo "[$(date -Is)] tp=${TP} shards=${SHARDS} concurrency=${CONCURRENCY_PER_SHARD} max_model_len=${MAX_MODEL_LEN}"

TOKENIZERS_PARALLELISM=false \
PYTHONUNBUFFERED=1 \
PYTHONPATH="${ROOT}/src:${ROOT}" \
"${PYTHON_BIN}" scripts/run_inference_bird_async_sharded.py \
  --model_name_or_path "${CKPT}" \
  --input_file "${INPUT_FILE}" \
  --database_dir "${DATABASE_DIR}" \
  --diff_json_path "${DIFF_JSON_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --num_examples "${NUM_EXAMPLES}" \
  --num_shards "${SHARDS}" \
  --gpu_groups "${GPU_GROUPS[@]}" \
  --max_prompt_length "${MAX_PROMPT_LENGTH}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --max_tool_rounds "${MAX_TOOL_ROUNDS}" \
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}" \
  --eval_timeout 60 \
  --eval_workers 16 \
  --vllm_tensor_parallel_size "${TP}" \
  --vllm_gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
  --vllm_max_model_len "${MAX_MODEL_LEN}" \
  --vllm_async_concurrency "${CONCURRENCY_PER_SHARD}" \
  --overwrite

echo "[$(date -Is)] complete rc=$?"
