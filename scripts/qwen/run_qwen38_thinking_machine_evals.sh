#!/usr/bin/env bash
# Qwen3.8-27B temp-0 evals over the three Thinking-Machine eval sets, in
# sequence, each using all 8 GPUs as tp=2 x 4 shards.
#
# The Gemma counterparts of these runs live in thinking_machine_eval.md on the
# consensus-thinky branch (85.74 / 88.96 / 71.40 EX for gemma-4-31B-it). The
# generation settings here are held identical to those runs so the only moving
# parts are the model and the Qwen-native system prompt:
#
#   temp 0.0, top_p 1.0, max_tool_rounds 8, max_new_tokens 8000,
#   max_prompt_length 34000, max_model_len 43000, tp=2, 4 shards, concurrency 16
#
# Qwen-specific, all inherited from scripts/qwen/run_qwen38_eval_smoke.sh --
# the invocation verified on this box:
#   - top_k 20 (Qwen's own generation config; Gemma runs had no top_k)
#   - --no_prompt_rewrite, because outputs/qwen-*.jsonl already carry the Qwen
#     tool syntax. Letting the runtime rewrite run on top strips the XML
#     examples and re-inserts "do not print the function call".
#   - thinking OFF by default, matching the Gemma baseline; ENABLE_THINKING=1
#     turns it on and tags the outputs with _think so the two never mix.
#   - NL2SQL_TOOL_LOOP_GUARD=1, which only applies at temperature 0.
#
# CRITICAL: .venv/bin must be on PATH. FlashInfer's GDN prefill kernel (the 48
# linear-attention layers) is JIT-compiled with ninja; without it on PATH
# EngineCore dies, every sample returns generation_error with 0 tool calls and
# 0% accuracy -- and the process still exits 0. The per-dataset sanity check at
# the bottom of this script is what catches that.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
export PYTHONPATH="src:.:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export NL2SQL_TOOL_LOOP_GUARD="${NL2SQL_TOOL_LOOP_GUARD:-1}"

# Weight loading dominates engine startup, and the instance-store NVMe is much
# faster than the EBS root. Prefer a copy there when one exists, falling back to
# the HF cache. /opt/dlami/nvme does NOT survive a stop/start, so the fallback is
# the correct behaviour after a restart, not an error.
NVME_MODEL="${NVME_MODEL:-/opt/dlami/nvme/models/Qwen3.8-27B}"
if [[ -z "${MODEL_PATH:-}" && -f "${NVME_MODEL}/config.json" ]]; then
  MODEL_PATH="${NVME_MODEL}"
fi
MODEL_PATH="${MODEL_PATH:-$(ls -d "${HOME}"/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/*/ | head -1)}"
MODEL_TAG="${MODEL_TAG:-Qwen3.8-27B}"
DATABASE_DIR="${DATABASE_DIR:-databases/dev_databases}"

TP="${TP:-2}"
SHARDS="${SHARDS:-4}"
GPU_GROUPS=("0,1" "2,3" "4,5" "6,7")
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:-20}"
CONCURRENCY="${CONCURRENCY:-16}"
MAX_TOOL_ROUNDS="${MAX_TOOL_ROUNDS:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.96}"
IDLE_MEMORY_MB="${IDLE_MEMORY_MB:-5000}"

# Thinking mode. Off by default, which is what the Gemma baseline compares
# against. ENABLE_THINKING=1 turns Qwen's reasoning on for generation.
#
# PRESERVE_THINKING stays off even when thinking is on, deliberately: the
# shipped chat template re-renders historical reasoning into the prompt, and
# qwen38_eval_plan.md records that it emits empty <think></think> blocks when
# reasoning is preserved. In a tool loop that drift compounds every round, so
# reasoning is generated per round and dropped from the history -- which is
# also Qwen's own recommendation.
ENABLE_THINKING="${ENABLE_THINKING:-0}"
PRESERVE_THINKING="${PRESERVE_THINKING:-0}"
THINK_ARGS=()
RUN_SUFFIX=""
if [[ "${ENABLE_THINKING}" == "1" ]]; then
  THINK_ARGS+=(--enable_thinking)
  RUN_SUFFIX="_think"
fi
if [[ "${PRESERVE_THINKING}" == "1" ]]; then
  THINK_ARGS+=(--preserve_thinking)
  RUN_SUFFIX="${RUN_SUFFIX}_preserve"
fi

# Token budgets depend on thinking, because reasoning traces are far longer than
# a bare answer. The thinking-off numbers recorded in thinking_machine_eval.md
# were produced at 8000/43000 and those values must not move, or they stop being
# comparable to the Gemma baseline.
#
# With thinking on, 8000 was too small: the first sweep truncated 9 of 498 on
# Plat-SQL with stop_reason max_new_tokens, i.e. reasoning cut mid-thought and
# scored as an answer. 16000 removes that.
#
# max_model_len has to rise with it. It bounds prompt + generation for one
# sequence, so 34000 + 16000 = 50000 does not fit in 43000. 65536 also covers
# tool-loop growth: the thinking-off runs already observed prompts of 47470
# tokens once query results had been appended, which is what produced the lone
# context_length_exceeded there.
if [[ "${ENABLE_THINKING}" == "1" ]]; then
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16000}"
  MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
else
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8000}"
  MAX_MODEL_LEN="${MAX_MODEL_LEN:-43000}"
fi
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-34000}"

if (( MAX_PROMPT_LENGTH + MAX_NEW_TOKENS > MAX_MODEL_LEN )); then
  echo "FATAL: max_prompt_length(${MAX_PROMPT_LENGTH}) + max_new_tokens(${MAX_NEW_TOKENS})" \
       "exceeds max_model_len(${MAX_MODEL_LEN}); a full-length generation cannot fit" >&2
  exit 2
fi

# Name the run directory after the budgets actually used, so a 16k thinking run
# is never filed under a tag claiming 8k.
CTX_TAG="ctx$((MAX_MODEL_LEN/1000))k_p$((MAX_PROMPT_LENGTH/1000))k_o$((MAX_NEW_TOKENS/1000))k_r${MAX_TOOL_ROUNDS}"

# dataset : input jsonl : diff json
DATASETS=(
  "arcwise_plat_sql:outputs/qwen-arcwise_plat_sql-schema-tool.jsonl:data/revisql/raw/arcwise_plat_sql.json"
  "arcwise_plat_full:outputs/qwen-arcwise_plat_full-schema-tool.jsonl:data/revisql/raw/arcwise_plat_full.json"
  "mini_dev_sqlite:outputs/qwen-mini_dev_sqlite-schema-tool.jsonl:data/bird_minidev_data/raw/mini_dev_sqlite.json"
)

# Run only a subset, e.g. ONLY_DATASETS=arcwise_plat_full,mini_dev_sqlite --
# used when a run is resumed and earlier datasets already have results.
if [[ -n "${ONLY_DATASETS:-}" ]]; then
  _keep=()
  for _e in "${DATASETS[@]}"; do
    _n="${_e%%:*}"
    case ",${ONLY_DATASETS}," in *",${_n},"*) _keep+=("${_e}") ;; esac
  done
  if [[ "${#_keep[@]}" -eq 0 ]]; then
    echo "ONLY_DATASETS='${ONLY_DATASETS}' matched no dataset" >&2
    exit 2
  fi
  DATASETS=("${_keep[@]}")
fi

TS="$(date +%Y%m%d_%H%M%S)"
SUITE_LOG="logs/qwen38_thinking_machine_evals${RUN_SUFFIX}_${TS}.log"
mkdir -p logs

max_gpu_memory_used_mb() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
    | awk 'BEGIN{m=0} {if ($1+0>m) m=$1+0} END{print m}'
}

wait_for_idle_gpus() {
  echo "[$(date -Is)] waiting for all GPUs idle; threshold=${IDLE_MEMORY_MB} MiB"
  while true; do
    used="$(max_gpu_memory_used_mb)"
    if [[ "${used}" -lt "${IDLE_MEMORY_MB}" ]]; then
      echo "[$(date -Is)] GPUs idle; max_used=${used} MiB"
      return 0
    fi
    echo "[$(date -Is)] GPUs busy; max_used=${used} MiB"
    sleep 60
  done
}

{
  echo "[$(date -Is)] suite start model=${MODEL_PATH}"
  echo "[$(date -Is)] tp=${TP} shards=${SHARDS} temp=${TEMPERATURE} top_k=${TOP_K} concurrency=${CONCURRENCY}"
  echo "[$(date -Is)] max_new_tokens=${MAX_NEW_TOKENS} max_model_len=${MAX_MODEL_LEN} max_prompt_length=${MAX_PROMPT_LENGTH} util=${GPU_MEMORY_UTILIZATION}"
  echo "[$(date -Is)] tool_loop_guard=${NL2SQL_TOOL_LOOP_GUARD} enable_thinking=${ENABLE_THINKING} preserve_thinking=${PRESERVE_THINKING}"

  declare -a SUMMARY_LINES=()
  suite_status=0

  for entry in "${DATASETS[@]}"; do
    IFS=":" read -r name input_file diff_json <<<"${entry}"
    out_dir="outputs/inference/${name}/${MODEL_TAG}/vllm_async_tp${TP}_dp${SHARDS}_${CTX_TAG}_temp0${RUN_SUFFIX}_${TS}"

    echo
    echo "=============================================================="
    echo "[$(date -Is)] dataset=${name} rows=$(wc -l < "${input_file}")"
    echo "[$(date -Is)] input=${input_file}"
    echo "[$(date -Is)] diff_json=${diff_json}"
    echo "[$(date -Is)] output=${out_dir}"
    echo "=============================================================="

    wait_for_idle_gpus
    mkdir -p "${out_dir}"

    set +e
    .venv/bin/python scripts/run_inference_bird_qwen_async.py \
      --model_name_or_path "${MODEL_PATH}" \
      --input_file "${input_file}" \
      --database_dir "${DATABASE_DIR}" \
      --diff_json_path "${diff_json}" \
      --output_dir "${out_dir}" \
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

    # A zero exit code is not proof of a real run: the FlashInfer ninja failure
    # exits 0 with every sample a generation_error. Judge on the summary.
    verdict="$(
      .venv/bin/python - "${out_dir}" <<'PY'
import json, sys, pathlib
d = pathlib.Path(sys.argv[1])
p = d / "eval_summary.json"
if not p.exists():
    print("FAIL no eval_summary.json"); raise SystemExit
s = json.loads(p.read_text())
t = s.get("total", {})
g = s.get("generation_stats", {})
stops = g.get("stop_reason_counts", {})
errs = sum(v for k, v in stops.items() if "error" in k.lower())
tools = g.get("tool_call_count_total", 0)
acc, cor, cnt = t.get("accuracy", 0.0), t.get("correct", 0), t.get("count", 0)
bad = errs > 0.10 * max(cnt, 1) or tools == 0
print(f"{'SUSPECT' if bad else 'OK'} EX={acc:.2f}% ({cor}/{cnt}) "
      f"tool_calls={tools} gen_errors={errs} stops={stops}")
PY
    )"
    echo "[$(date -Is)] ${name}: rc=${rc} ${verdict}"
    SUMMARY_LINES+=("${name}: ${verdict}")
    if [[ "${rc}" -ne 0 || "${verdict}" != OK* ]]; then
      suite_status=1
      echo "[$(date -Is)] ${name} did not produce a clean run; continuing to next dataset."
    fi
  done

  echo
  echo "=============================================================="
  echo "[$(date -Is)] suite complete status=${suite_status}"
  for line in "${SUMMARY_LINES[@]}"; do echo "  ${line}"; done
  echo "=============================================================="
  exit "${suite_status}"
} 2>&1 | tee -a "${SUITE_LOG}"
