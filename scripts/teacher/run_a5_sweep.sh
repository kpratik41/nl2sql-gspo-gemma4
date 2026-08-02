#!/usr/bin/env bash
# Stage A5 — temp-0 dev inference across every SFT checkpoint.
# One checkpoint per GPU (tp=1), all run concurrently. Results are written
# inside each checkpoint folder.
set -euo pipefail
cd /home/ubuntu/nl2sql-gspo-gemma4

export PYTHONUNBUFFERED=1
export PYTHONPATH=".:src:scripts:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

PY="${PY:-.venv/bin/python}"
SFT_ROOT="${SFT_ROOT:-outputs/sft/gemma4_31b_rft_sft}"
INPUT="${INPUT:-outputs/old-dev-schema-tool-unpatched.jsonl}"
DIFF="${DIFF:-data/bird_dev_data/raw/bird_dev_unpatched.json}"
DB="${DB:-databases/dev_databases}"
TAG="${TAG:-temp0_olddev_schema_tool_unpatched_vllm_async_tp1}"
LOGDIR="${LOGDIR:-outputs/sft/a5_sweep_logs}"

mkdir -p "$LOGDIR"
exec > >(tee -a "$LOGDIR/sweep.log") 2>&1
echo "[$(date -Is)] A5 temp-0 sweep | input=$INPUT | tp=1, one checkpoint per GPU"

ckpts=($(ls -d ${SFT_ROOT}/checkpoint-* | sort -t- -k2 -n))
echo "[$(date -Is)] ${#ckpts[@]} checkpoints: ${ckpts[*]##*/}"

pids=(); names=()
for i in "${!ckpts[@]}"; do
  ck="${ckpts[$i]}"; name="$(basename $ck)"
  echo "[$(date -Is)] launching $name on GPU $i"
  CUDA_VISIBLE_DEVICES="$i" "$PY" scripts/run_inference_bird.py \
    --inference_backend vllm_async \
    --model_name_or_path "$ck" \
    --input_file "$INPUT" \
    --database_dir "$DB" \
    --diff_json_path "$DIFF" \
    --output_dir "${ck}/${TAG}" \
    --num_examples -1 \
    --max_prompt_length 34000 --max_new_tokens 8000 \
    --temperature 0.0 --top_p 1.0 \
    --eval_timeout 60 --eval_workers 16 \
    --vllm_tensor_parallel_size 1 \
    --vllm_data_parallel_size 1 \
    --vllm_gpu_memory_utilization 0.90 \
    --vllm_max_model_len 43000 \
    --vllm_async_concurrency 16 \
    --max_tool_rounds 8 \
    --overwrite > "${LOGDIR}/${name}.log" 2>&1 &
  pids+=($!); names+=("$name")
done

fail=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then echo "[$(date -Is)] ${names[$i]} OK"; else echo "[$(date -Is)] ${names[$i]} FAILED (${LOGDIR}/${names[$i]}.log)"; fail=1; fi
done

echo "[$(date -Is)] === SUMMARY ==="
for ck in "${ckpts[@]}"; do
  s="${ck}/${TAG}/eval_summary.json"
  if [ -f "$s" ]; then
    acc=$("$PY" -c "import json;d=json.load(open('$s'));t=d['total'];print(f\"{t['accuracy']:.2f}%  ({t['correct']}/{t['count']})\")")
    printf "  %-14s %s\n" "$(basename $ck)" "$acc"
  else
    printf "  %-14s NO RESULT\n" "$(basename $ck)"
  fi
done
echo "[$(date -Is)] complete (fail=$fail)"
