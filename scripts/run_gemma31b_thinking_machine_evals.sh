#!/usr/bin/env bash
# gemma-4-31B-it temp-0 evals over the three Thinking-Machine eval sets, in
# sequence, each on all 8 GPUs as tp=2 x 4 shards.
#
# This exists to close a gap in the comparison. The gemma-4-31B-it numbers
# recorded in thinking_machine_eval.md (85.74 / 88.96 / 71.40) were produced
# with thinking OFF -- not by choice but by omission: render_prompt passed no
# chat-template kwargs, and gemma-4's template sets
# `enable_thinking | default(false)`. Meanwhile Qwen3.8-27B was measured both
# ways, so "Qwen thinking-on beats Gemma" was really thinking-on against
# thinking-off. This runs Gemma at the same setting.
#
# Held identical to the Qwen thinking sweep so the two are comparable:
#   temp 0.0, top_p 1.0, tp=2, 4 shards, concurrency 16, max_tool_rounds 8,
#   max_new_tokens 16000, max_model_len 65536, preserve_thinking off
#
# ENABLE_THINKING defaults to 1 here -- the thinking-off numbers already exist,
# and this script is for the run that does not. Set 0 to reproduce the baseline.
#
# On the budgets: 8000 was too small for reasoning. The first Qwen thinking
# sweep truncated 9 of 498 mid-thought and scored the fragment as an answer, so
# 16000 is used here from the start. max_model_len must rise with it, since it
# bounds prompt + generation for one sequence and 34000 + 16000 does not fit in
# 43000. gemma-4-31B-it supports 262144, so 65536 is well within range.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="src:.:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-google/gemma-4-31B-it}"
MODEL_TAG="${MODEL_TAG:-gemma-4-31B-it}"
DATABASE_DIR="${DATABASE_DIR:-databases/dev_databases}"

TP="${TP:-2}"
SHARDS="${SHARDS:-4}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
CONCURRENCY="${CONCURRENCY:-16}"
MAX_TOOL_ROUNDS="${MAX_TOOL_ROUNDS:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.96}"
IDLE_MEMORY_MB="${IDLE_MEMORY_MB:-5000}"

ENABLE_THINKING="${ENABLE_THINKING:-1}"
PRESERVE_THINKING="${PRESERVE_THINKING:-0}"
THINK_ARGS=()
RUN_SUFFIX=""
if [[ "${ENABLE_THINKING}" == "1" ]]; then
  THINK_ARGS+=(--enable_thinking)
  RUN_SUFFIX="_think"
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16000}"
  MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
else
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8000}"
  MAX_MODEL_LEN="${MAX_MODEL_LEN:-43000}"
fi
if [[ "${PRESERVE_THINKING}" == "1" ]]; then
  THINK_ARGS+=(--preserve_thinking)
  RUN_SUFFIX="${RUN_SUFFIX}_preserve"
fi
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-34000}"

if (( MAX_PROMPT_LENGTH + MAX_NEW_TOKENS > MAX_MODEL_LEN )); then
  echo "FATAL: max_prompt_length(${MAX_PROMPT_LENGTH}) + max_new_tokens(${MAX_NEW_TOKENS})" \
       "exceeds max_model_len(${MAX_MODEL_LEN}); a full-length generation cannot fit" >&2
  exit 2
fi

# dataset : input jsonl : diff json   -- the Gemma-format files, not the qwen-* ones
DATASETS=(
  "arcwise_plat_sql:outputs/arcwise_plat_sql-schema-tool.jsonl:data/revisql/raw/arcwise_plat_sql.json"
  "arcwise_plat_full:outputs/arcwise_plat_full-schema-tool.jsonl:data/revisql/raw/arcwise_plat_full.json"
  "mini_dev_sqlite:outputs/mini_dev_sqlite-schema-tool.jsonl:data/bird_minidev_data/raw/mini_dev_sqlite.json"
)

if [[ -n "${ONLY_DATASETS:-}" ]]; then
  _keep=()
  for _e in "${DATASETS[@]}"; do
    _n="${_e%%:*}"
    case ",${ONLY_DATASETS}," in *",${_n},"*) _keep+=("${_e}") ;; esac
  done
  if [[ "${#_keep[@]}" -eq 0 ]]; then
    echo "ONLY_DATASETS='${ONLY_DATASETS}' matched no dataset" >&2; exit 2
  fi
  DATASETS=("${_keep[@]}")
fi

TS="$(date +%Y%m%d_%H%M%S)"
CTX_TAG="ctx$((MAX_MODEL_LEN/1000))k_p$((MAX_PROMPT_LENGTH/1000))k_o$((MAX_NEW_TOKENS/1000))k_r${MAX_TOOL_ROUNDS}"
SUITE_LOG="logs/gemma31b_thinking_machine_evals${RUN_SUFFIX}_${TS}.log"
mkdir -p logs

max_gpu_memory_used_mb() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
    | awk 'BEGIN{m=0} {if ($1+0>m) m=$1+0} END{print m}'
}

{
  echo "[$(date -Is)] suite start model=${MODEL_PATH}"
  echo "[$(date -Is)] tp=${TP} shards=${SHARDS} temp=${TEMPERATURE} concurrency=${CONCURRENCY}"
  echo "[$(date -Is)] max_new_tokens=${MAX_NEW_TOKENS} max_model_len=${MAX_MODEL_LEN} max_prompt_length=${MAX_PROMPT_LENGTH} util=${GPU_MEMORY_UTILIZATION}"
  echo "[$(date -Is)] enable_thinking=${ENABLE_THINKING} preserve_thinking=${PRESERVE_THINKING}"

  declare -a SUMMARY_LINES=()
  suite_status=0

  for entry in "${DATASETS[@]}"; do
    IFS=":" read -r name input_file diff_json <<<"${entry}"
    out_dir="outputs/inference/${name}/${MODEL_TAG}/vllm_async_tp${TP}_dp${SHARDS}_${CTX_TAG}_temp0${RUN_SUFFIX}_${TS}"

    echo
    echo "=============================================================="
    echo "[$(date -Is)] dataset=${name} rows=$(wc -l < "${input_file}")"
    echo "[$(date -Is)] input=${input_file}"
    echo "[$(date -Is)] output=${out_dir}"
    echo "=============================================================="

    echo "[$(date -Is)] waiting for all GPUs idle; threshold=${IDLE_MEMORY_MB} MiB"
    while true; do
      used="$(max_gpu_memory_used_mb)"
      if [[ "${used}" -lt "${IDLE_MEMORY_MB}" ]]; then
        echo "[$(date -Is)] GPUs idle; max_used=${used} MiB"; break
      fi
      echo "[$(date -Is)] GPUs busy; max_used=${used} MiB"; sleep 60
    done
    mkdir -p "${out_dir}"

    set +e
    "${PYTHON_BIN}" scripts/run_inference_bird_async_sharded.py \
      --model_name_or_path "${MODEL_PATH}" \
      --input_file "${input_file}" \
      --database_dir "${DATABASE_DIR}" \
      --diff_json_path "${diff_json}" \
      --output_dir "${out_dir}" \
      --num_examples -1 \
      --num_shards "${SHARDS}" \
      --vllm_tensor_parallel_size "${TP}" \
      --vllm_gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
      --vllm_max_model_len "${MAX_MODEL_LEN}" \
      --vllm_async_concurrency "${CONCURRENCY}" \
      --max_prompt_length "${MAX_PROMPT_LENGTH}" \
      --max_new_tokens "${MAX_NEW_TOKENS}" \
      --max_tool_rounds "${MAX_TOOL_ROUNDS}" \
      --temperature "${TEMPERATURE}" \
      --top_p "${TOP_P}" \
      --eval_timeout 60 \
      --eval_workers 16 \
      "${THINK_ARGS[@]}" \
      --overwrite
    rc=$?
    set -e

    # A zero exit code is not proof of a real run; judge on the summary.
    verdict="$(
      "${PYTHON_BIN}" - "${out_dir}" <<'PYEOF'
import json, sys, pathlib
d = pathlib.Path(sys.argv[1])
p = d / "eval_summary.json"
if not p.exists():
    print("FAIL no eval_summary.json"); raise SystemExit
s = json.loads(p.read_text())
t = s.get("total", {}); g = s.get("generation_stats", {})
stops = g.get("stop_reason_counts", {})
errs = sum(v for k, v in stops.items() if "error" in k.lower())
tools = g.get("tool_call_count_total", 0)
cnt = t.get("count", 0)
bad = errs > 0.10 * max(cnt, 1) or tools == 0
print(f"{'SUSPECT' if bad else 'OK'} EX={t.get('accuracy',0):.2f}% "
      f"({t.get('correct',0)}/{cnt}) tool_calls={tools} gen_errors={errs} stops={stops}")
PYEOF
    )"
    echo "[$(date -Is)] ${name}: rc=${rc} ${verdict}"
    SUMMARY_LINES+=("${name}: ${verdict}")
    if [[ "${rc}" -ne 0 || "${verdict}" != OK* ]]; then
      suite_status=1
      echo "[$(date -Is)] ${name} did not produce a clean run; continuing."
    fi
  done

  echo
  echo "=============================================================="
  echo "[$(date -Is)] suite complete status=${suite_status}"
  for line in "${SUMMARY_LINES[@]}"; do echo "  ${line}"; done
  echo "=============================================================="
  exit "${suite_status}"
} 2>&1 | tee -a "${SUITE_LOG}"
