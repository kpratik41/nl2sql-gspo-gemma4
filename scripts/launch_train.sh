#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

export WANDB_PROJECT=gemma4-31b-bird-gspo
REPORT_TO="${REPORT_TO:-wandb}"
RUN_NAME="${RUN_NAME:-gemma4-31b-gspo-bird}"
LOGGING_DIR="${LOGGING_DIR:-outputs/gemma4_31b_gspo_bird/tb}"

MODEL_NAME="google/gemma-4-31B-it"
RESUME_ARGS=()

if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  RESUME_ARGS+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi

accelerate launch \
  --num_processes 6 \
  --mixed_precision bf16 \
  src/nl2sql_gspo/train_gspo_nl2sql.py \
  --model_name_or_path "${MODEL_NAME}" \
  --train_file outputs/train-6601-schema-filtered.jsonl \
  --eval_file outputs/dev-20251106-schema.jsonl \
  --database_dir databases \
  --output_dir outputs/gemma4_31b_gspo_bird \
  --vllm_server_base_url http://127.0.0.1:8000 \
  --max_prompt_length 16384 \
  --max_completion_length 4096 \
  --num_generations 16 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-7 \
  --num_train_epochs 1 \
  --reward_weights 0.25,1.0,2.5,0.5,0.5,0.25 \
  --report_to "${REPORT_TO}" \
  --run_name "${RUN_NAME}" \
  --logging_dir "${LOGGING_DIR}" \
  --deepspeed configs/ds_zero3_bf16.json \
  --eval_steps 100 \
  --loss_type dapo \
  --scale_rewards batch \
  --beta 0.0 \
  --epsilon 0.2 \
  --epsilon_high 0.28 \
  "${RESUME_ARGS[@]}"