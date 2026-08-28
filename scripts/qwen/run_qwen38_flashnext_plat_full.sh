#!/usr/bin/env bash
# Qwen3.8-Flash-Next temp-0 eval on arcwise_plat_full only, tp=8 on a single
# shard (all 8 GPUs held by one engine, rather than 4 engines of tp=2).
#
# Reuses outputs/qwen-arcwise_plat_full-schema-tool.jsonl unchanged. That is
# safe and was verified rather than assumed: Flash-Next ships a chat_template.jinja
# byte-identical to Qwen3.8-27B's, the same Qwen2Tokenizer with the same 33 added
# tokens and the same eos/pad, and the same <tool_call> XML dialect and
# enable_thinking / preserve_thinking kwargs. So the Qwen-native file built for
# the 27B needs no regeneration, and --no_prompt_rewrite applies for the same
# reason it does there.
#
# Model shape, for sizing expectations: Qwen4ExpForConditionalGeneration, a
# 180B-parameter MoE with 512 experts and 10 active per token, hybrid attention
# (48 layers mixing linear_attention and full_attention), ~360GB of BF16 weights
# across 131 safetensors shards (~335 GiB resident). At tp=8 on 8xH100 80GB that
# is ~42GB of weights per GPU, leaving room for KV cache at 0.96 utilization.
#
# PREREQUISITE, and it is currently unmet on this box: vLLM must support the
# qwen4_exp architecture. See the preflight below.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# FlashInfer's GDN prefill kernel is ninja-JIT-compiled; without .venv/bin on
# PATH EngineCore dies with every sample a generation_error and 0% accuracy,
# while still exiting 0.
export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
export PYTHONPATH="src:.:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export NL2SQL_TOOL_LOOP_GUARD="${NL2SQL_TOOL_LOOP_GUARD:-1}"

# Prefer the NVMe copy: loading 360GB off the EBS root is the dominant cost of
# engine startup. /opt/dlami/nvme is instance store and does not survive a
# stop/start, so the fallback to the persistent copy is deliberate.
NVME_MODEL="${NVME_MODEL:-/opt/dlami/nvme/models/Qwen3.8-Flash-Next}"
EBS_MODEL="${EBS_MODEL:-/home/ubuntu/models/Qwen3.8-Flash-Next}"
if [[ -z "${MODEL_PATH:-}" && -f "${NVME_MODEL}/config.json" ]]; then
  MODEL_PATH="${NVME_MODEL}"
fi
MODEL_PATH="${MODEL_PATH:-${EBS_MODEL}}"
MODEL_TAG="${MODEL_TAG:-Qwen3.8-Flash-Next}"

DATASET="${DATASET:-arcwise_plat_full}"
INPUT_FILE="${INPUT_FILE:-outputs/qwen-arcwise_plat_full-schema-tool.jsonl}"
DIFF_JSON="${DIFF_JSON:-data/revisql/raw/arcwise_plat_full.json}"
DATABASE_DIR="${DATABASE_DIR:-databases/dev_databases}"

TP="${TP:-8}"
SHARDS="${SHARDS:-1}"
GPU_GROUPS=("${GPU_GROUPS:-0,1,2,3,4,5,6,7}")
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:-20}"
CONCURRENCY="${CONCURRENCY:-16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-43000}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-34000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8000}"
MAX_TOOL_ROUNDS="${MAX_TOOL_ROUNDS:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.96}"
IDLE_MEMORY_MB="${IDLE_MEMORY_MB:-5000}"

ENABLE_THINKING="${ENABLE_THINKING:-0}"
PRESERVE_THINKING="${PRESERVE_THINKING:-0}"
THINK_ARGS=()
RUN_SUFFIX=""
if [[ "${ENABLE_THINKING}" == "1" ]]; then
  THINK_ARGS+=(--enable_thinking); RUN_SUFFIX="_think"
fi
if [[ "${PRESERVE_THINKING}" == "1" ]]; then
  THINK_ARGS+=(--preserve_thinking); RUN_SUFFIX="${RUN_SUFFIX}_preserve"
fi

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="outputs/inference/${DATASET}/${MODEL_TAG}/vllm_async_tp${TP}_dp${SHARDS}_ctx43k_p34k_o8k_r8_temp0${RUN_SUFFIX}_${TS}"
LOG="logs/qwen38_flashnext_${DATASET}${RUN_SUFFIX}_${TS}.log"
mkdir -p logs

# ---- preflight ------------------------------------------------------------
# Fail here with something readable rather than deep inside engine init.

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "FATAL: no config.json under ${MODEL_PATH}" >&2
  echo "       the download may still be in flight; check:" >&2
  echo "       ls ${EBS_MODEL}/*.safetensors | wc -l   (expect 131)" >&2
  exit 1
fi

shard_count="$(ls "${MODEL_PATH}"/*.safetensors 2>/dev/null | wc -l)"
incomplete="$(find "${MODEL_PATH}" -name '*.incomplete' 2>/dev/null | wc -l)"
if [[ "${shard_count}" -ne 131 || "${incomplete}" -ne 0 ]]; then
  echo "FATAL: ${MODEL_PATH} has ${shard_count}/131 shards and ${incomplete} incomplete files" >&2
  exit 1
fi

# Qwen3.8-Flash-Next is Qwen4ExpForConditionalGeneration / model_type qwen4_exp,
# a new architecture. It ships no remote-code .py files, so trust_remote_code is
# not an escape hatch. Check support up front: without it vLLM fails partway
# through engine init after the model has already been read from disk.
if ! .venv/bin/python - "${MODEL_PATH}" <<'PY'
import json, sys, pathlib
cfg = json.loads((pathlib.Path(sys.argv[1]) / "config.json").read_text())
arch = (cfg.get("architectures") or ["<none>"])[0]
from vllm.model_executor.models.registry import ModelRegistry
if arch in ModelRegistry.get_supported_archs():
    print(f"[preflight] vLLM supports {arch}")
    raise SystemExit(0)
import vllm, transformers
print(f"[preflight] vLLM {vllm.__version__} does NOT support {arch} "
      f"(transformers {transformers.__version__}, model_type={cfg.get('model_type')})",
      file=sys.stderr)
raise SystemExit(1)
PY
then
  cat >&2 <<'MSG'

FATAL: this vLLM build cannot load Qwen3.8-Flash-Next.

  Qwen3.8-Flash-Next is Qwen4ExpForConditionalGeneration (model_type
  qwen4_exp), released 2026-08-27. As of this box:

    vllm 0.19.1        -- qwen4_exp not among the registered architectures
    transformers 5.9.0 -- qwen4_exp not in CONFIG_MAPPING_NAMES

  The model's own config declares transformers_version 5.8.0.dev0, yet our
  newer 5.9.0 still lacks it, so support landed in a fork or an unmerged PR
  rather than upstream. The repo ships no .py modeling files, so
  trust_remote_code cannot work around it either.

  Running this needs a vLLM/transformers upgrade, which must be validated
  against the current working setup: FA3 on the full-attention layers and the
  FlashInfer GDN prefill kernel on the linear-attention ones. The download and
  the NVMe copy are still worth completing in the meantime -- nothing about
  the data or this script changes.

MSG
  exit 2
fi
# ---------------------------------------------------------------------------

mkdir -p "${OUT_DIR}"

max_gpu_memory_used_mb() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
    | awk 'BEGIN{m=0} {if ($1+0>m) m=$1+0} END{print m}'
}

{
  echo "[$(date -Is)] flash-next eval start"
  echo "[$(date -Is)] model=${MODEL_PATH}"
  echo "[$(date -Is)] dataset=${DATASET} input=${INPUT_FILE} rows=$(wc -l < "${INPUT_FILE}")"
  echo "[$(date -Is)] tp=${TP} shards=${SHARDS} gpus=${GPU_GROUPS[*]} util=${GPU_MEMORY_UTILIZATION}"
  echo "[$(date -Is)] enable_thinking=${ENABLE_THINKING} preserve_thinking=${PRESERVE_THINKING}"
  echo "[$(date -Is)] output=${OUT_DIR}"

  echo "[$(date -Is)] waiting for all GPUs idle; threshold=${IDLE_MEMORY_MB} MiB"
  while true; do
    used="$(max_gpu_memory_used_mb)"
    if [[ "${used}" -lt "${IDLE_MEMORY_MB}" ]]; then
      echo "[$(date -Is)] GPUs idle; max_used=${used} MiB"; break
    fi
    echo "[$(date -Is)] GPUs busy; max_used=${used} MiB"
    sleep 60
  done

  set +e
  .venv/bin/python scripts/run_inference_bird_qwen_async.py \
    --model_name_or_path "${MODEL_PATH}" \
    --input_file "${INPUT_FILE}" \
    --database_dir "${DATABASE_DIR}" \
    --diff_json_path "${DIFF_JSON}" \
    --output_dir "${OUT_DIR}" \
    --num_examples -1 \
    --num_shards "${SHARDS}" \
    --gpu_groups "${GPU_GROUPS[@]}" \
    --vllm_tensor_parallel_size "${TP}" \
    --vllm_gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
    --vllm_max_model_len "${MAX_MODEL_LEN}" \
    --vllm_async_concurrency "${CONCURRENCY}" \
    --max_prompt_length "${MAX_PROMPT_LENGTH}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --max_tool_rounds "${MAX_TOOL_ROUNDS}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --top_k "${TOP_K}" \
    --eval_timeout 60 \
    --eval_workers 16 \
    --tool_choice_policy required_first \
    --empty_tool_retries 1 \
    --no_prompt_rewrite \
    "${THINK_ARGS[@]}" \
    --overwrite
  rc=$?
  set -e

  # A zero exit code is not proof of a real run; judge on the summary.
  .venv/bin/python - "${OUT_DIR}" <<'PY'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1]) / "eval_summary.json"
if not p.exists():
    print("FAIL no eval_summary.json"); raise SystemExit
s = json.loads(p.read_text()); t = s.get("total", {}); g = s.get("generation_stats", {})
stops = g.get("stop_reason_counts", {})
errs = sum(v for k, v in stops.items() if "error" in k.lower())
tools = g.get("tool_call_count_total", 0)
cnt = t.get("count", 0)
bad = errs > 0.10 * max(cnt, 1) or tools == 0
print(f"{'SUSPECT' if bad else 'OK'} EX={t.get('accuracy',0):.2f}% "
      f"({t.get('correct',0)}/{cnt}) tool_calls={tools} gen_errors={errs} stops={stops}")
PY
  echo "[$(date -Is)] complete rc=${rc}; summary=${OUT_DIR}/eval_summary.md"
} 2>&1 | tee -a "${LOG}"
