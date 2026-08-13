#!/usr/bin/env bash
# Whole-node masked multi-turn SFT (Path A4).
#
# Works both under Ray (when CUDA_VISIBLE_DEVICES is assigned by Ray) and on the
# current node (defaults to every GPU reported by nvidia-smi).  torchrun is used
# over the selected GPUs with DeepSpeed ZeRO-3.
set -euo pipefail

CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${CODE}"

export PYTHONPATH="${CODE}/_pkgs_trl14:${CODE}:${CODE}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -x "${CODE}/.venv/bin/python" ]]; then
        PYTHON_BIN="${CODE}/.venv/bin/python"
    else
        PYTHON_BIN="python"
    fi
fi
if [[ -z "${TORCHRUN_BIN:-}" ]]; then
    if [[ -x "${CODE}/.venv/bin/torchrun" ]]; then
        TORCHRUN_BIN="${CODE}/.venv/bin/torchrun"
    else
        TORCHRUN_BIN="torchrun"
    fi
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)"
    export CUDA_VISIBLE_DEVICES
fi
SFT_GPUS="${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES or run on a GPU node}"
IFS=',' read -r -a GPU_ARR <<< "${SFT_GPUS}"
NGPU="${#GPU_ARR[@]}"

MODEL_PATH="${MODEL_PATH:-google/gemma-4-31B-it}"
TRAIN_FILE="${TRAIN_FILE:-outputs/teacher/a3_rft_from_traces/rft_train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/sft/gemma4_31b_rft_sft_consensus_sft}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${CODE}/configs/ds_zero3_bf16_no_scheduler.json}"

LEARNING_RATE="${LEARNING_RATE:-1e-5}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-20480}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
SAVE_STEPS="${SAVE_STEPS:-10}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-8}"
REPORT_TO="${REPORT_TO:-none}"
RUN_NAME="${RUN_NAME:-$(basename "${OUTPUT_DIR}")}"
MASTER_PORT="${MASTER_PORT:-29610}"
DRY_RUN="${DRY_RUN:-0}"

echo "[sft] MODEL_PATH=${MODEL_PATH}"
echo "[sft] gpus=${SFT_GPUS} nproc=${NGPU}"
echo "[sft] train_file=${TRAIN_FILE}"
echo "[sft] output_dir=${OUTPUT_DIR}"
echo "[sft] lr=${LEARNING_RATE} epochs=${NUM_TRAIN_EPOCHS} bs=${PER_DEVICE_TRAIN_BATCH_SIZE} ga=${GRADIENT_ACCUMULATION_STEPS}"
echo "[sft] python=${PYTHON_BIN}"
echo "[sft] torchrun=${TORCHRUN_BIN}"

if [[ ! -f "${TRAIN_FILE}" ]]; then
    echo "[sft] FATAL: missing TRAIN_FILE=${TRAIN_FILE}" >&2
    exit 1
fi
if [[ ! -f "${DEEPSPEED_CONFIG}" ]]; then
    echo "[sft] FATAL: missing DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG}" >&2
    exit 1
fi

"${PYTHON_BIN}" -c "import deepspeed,transformers,torch;print('[sft] deepspeed',deepspeed.__version__,'transformers',transformers.__version__,'torch',torch.__version__)" \
    || { echo "[sft] FATAL: import check failed"; exit 1; }

mkdir -p "${OUTPUT_DIR}"

RESUME_ARG=()
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
    RESUME_ARG=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[sft] dry run; command:"
    echo "CUDA_VISIBLE_DEVICES=${SFT_GPUS} ${TORCHRUN_BIN} --nproc_per_node=${NGPU} --master_port=${MASTER_PORT} scripts/sft/train_sft.py --model_name_or_path ${MODEL_PATH} --train_file ${TRAIN_FILE} --output_dir ${OUTPUT_DIR} --deepspeed ${DEEPSPEED_CONFIG} --max_seq_len ${MAX_SEQ_LEN} --learning_rate ${LEARNING_RATE} --num_train_epochs ${NUM_TRAIN_EPOCHS} --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} --warmup_ratio ${WARMUP_RATIO} --save_steps ${SAVE_STEPS} --save_total_limit ${SAVE_TOTAL_LIMIT} --report_to ${REPORT_TO} --run_name ${RUN_NAME}"
    exit 0
fi

"${TORCHRUN_BIN}" --nproc_per_node="${NGPU}" --master_port="${MASTER_PORT}" \
    scripts/sft/train_sft.py \
    --model_name_or_path "${MODEL_PATH}" \
    --train_file "${TRAIN_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --deepspeed "${DEEPSPEED_CONFIG}" \
    --max_seq_len "${MAX_SEQ_LEN}" \
    --learning_rate "${LEARNING_RATE}" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --warmup_ratio "${WARMUP_RATIO}" \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --report_to "${REPORT_TO}" \
    --run_name "${RUN_NAME}" \
    "${RESUME_ARG[@]}"

rc=$?
echo "[sft] exited rc=${rc}"
exit "${rc}"
