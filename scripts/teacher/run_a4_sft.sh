#!/usr/bin/env bash
# Stage A4 — masked multi-turn SFT on the RFT dataset (31B, ZeRO-3, 8 GPUs).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

export PYTHONUNBUFFERED=1
export PYTHONPATH=".:src:scripts:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}"
export TORCH_NCCL_TIMEOUT_MS="${TORCH_NCCL_TIMEOUT_MS:-3600000}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PY="${PY:-/home/ec2-user/miniconda3/envs/nl2sql312/bin/python}"
MODEL="${MODEL:-google/gemma-4-31B-it}"
TRAIN_FILE="${TRAIN_FILE:-outputs/teacher/rft/train_rft_31b.jsonl}"
OUT="${OUT:-outputs/sft/gemma4_31b_rft_sft}"
NPROC="${NPROC:-8}"
CACHE_FILE="${CACHE_FILE:-${OUT}/tokenized_cache.jsonl}"

mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1
echo "[$(date -Is)] Stage A4 masked multi-turn SFT"
echo "[$(date -Is)] model=$MODEL train_file=$TRAIN_FILE out=$OUT gpus=$NPROC"

"${PY}" -m accelerate.commands.launch \
  --num_processes "$NPROC" \
  --mixed_precision bf16 \
  scripts/teacher/train_sft.py \
  --model_name_or_path "$MODEL" \
  --train_file "$TRAIN_FILE" \
  --output_dir "$OUT" \
  --deepspeed configs/ds_zero3_bf16_no_scheduler.json \
  --max_seq_len 20480 \
  --learning_rate 1e-5 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --save_steps 10 \
  --save_total_limit 8 \
  --logging_steps 1 \
  --cache_file "$CACHE_FILE"

echo "[$(date -Is)] complete"
