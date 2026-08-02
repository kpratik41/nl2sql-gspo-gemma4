#!/usr/bin/env bash
# Stage A2 — greedy gold-conditioned teacher traces over the all-wrong band.
set -euo pipefail
cd /home/ubuntu/nl2sql-gspo-gemma4
export PYTHONUNBUFFERED=1
export PYTHONPATH=".:src:scripts:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
OUT="${OUT:-outputs/teacher/a2_greedy}"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1
echo "[$(date -Is)] Stage A2 greedy teacher traces"
.venv/bin/python scripts/teacher/gen_teacher_traces.py \
  --model_name_or_path google/gemma-4-31B-it \
  --input_file outputs/train-6601-schema-bare-tool.jsonl \
  --database_dir databases/train_databases \
  --target_idx_file outputs/teacher/target_idx_all_wrong.txt \
  --output_dir "$OUT" \
  --hint_strategy full_sql \
  --num_samples 1 \
  --temperature 0.0 --top_p 1.0 \
  --max_prompt_length 34000 --max_new_tokens 8000 --max_tool_rounds 8 \
  --eval_timeout 60 \
  --vllm_tensor_parallel_size 8 \
  --vllm_gpu_memory_utilization 0.93 \
  --vllm_max_model_len 43000 \
  --vllm_async_concurrency 16 \
  --overwrite
echo "[$(date -Is)] complete"
