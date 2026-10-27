#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
runtime_python="${VERL_PYTHON:-$repo_root/.venv-verl/bin/python}"
if ! command -v "$runtime_python" >/dev/null 2>&1; then
  echo "Missing verl Python: $runtime_python. Run src/nl2sql_verl/setup_env.sh or set VERL_PYTHON." >&2
  exit 1
fi
runtime_python="$("$runtime_python" -c 'import sys; print(sys.executable)')"
export VERL_PYTHON="$runtime_python"
# Keep cuBLAS and cuBLASLt from the same CUDA wheel. The B200 host's
# LD_LIBRARY_PATH otherwise mixes system CUDA 13.2 with torch's CUDA 13.0.
runtime_cuda_lib="$("$runtime_python" -c 'import sysconfig; from pathlib import Path; print(Path(sysconfig.get_paths()["purelib"])/"nvidia/cu13/lib")')"
if [[ -d "$runtime_cuda_lib" ]]; then
  export LD_LIBRARY_PATH="$runtime_cuda_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
config_name="${RECIPE_CONFIG:-smoke}"
train_input="${TRAIN_PARQUET:-$repo_root/data/nl2sql/train.parquet}"
val_input="${VAL_PARQUET:-$repo_root/data/nl2sql/validation.parquet}"
export NL2SQL_DATABASE_DIR="${NL2SQL_DATABASE_DIR:-$repo_root/databases}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$repo_root${VERL_SOURCE:+:$VERL_SOURCE}${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p data/nl2sql databases outputs/verl_data outputs/verl_recipe_checks
if [[ "${CONFIG_ONLY:-0}" == 1 ]]; then
  exec "$runtime_python" -m src.nl2sql_verl.preflight --config "$config_name" --config-only --dump "$@"
fi
for input in "$train_input" "$val_input"; do
  if [[ ! -f "$input" ]]; then
    echo "Missing parquet: $input. Create the parquet and SQLite databases first." >&2
    exit 1
  fi
done
model_path="${MODEL_PATH:-google/gemma-4-31B-it}"
prompt_limit="${MAX_PROMPT_LENGTH:-2048}"
response_limit="${MAX_RESPONSE_LENGTH:-4096}"
if [[ "$config_name" == smoke ]]; then
  prompt_limit="${MAX_PROMPT_LENGTH:-1024}"
  response_limit="${MAX_RESPONSE_LENGTH:-1024}"
fi
if [[ "$config_name" == cp1 ]]; then
  prompt_limit="${MAX_PROMPT_LENGTH:-11264}"
  response_limit="${MAX_RESPONSE_LENGTH:-2048}"
fi
export NL2SQL_LENGTH_MAX="$response_limit"
export NL2SQL_LENGTH_BUFFER="${NL2SQL_LENGTH_BUFFER:-512}"
length_policy=()
if [[ "$config_name" == cp1 ]]; then
  length_policy=(--reject-overlength)
fi
"$runtime_python" -m src.nl2sql_verl.prepare_data --input "$train_input" --output outputs/verl_data/train.parquet --model "$model_path" --max-prompt-length "$prompt_limit" "${length_policy[@]}"
"$runtime_python" -m src.nl2sql_verl.prepare_data --input "$val_input" --output outputs/verl_data/validation.parquet --model "$model_path" --max-prompt-length "$prompt_limit" "${length_policy[@]}"
context_limit=$((prompt_limit + response_limit))
batch_token_limit="$context_limit"
rollout_token_limit="$context_limit"
if [[ "$config_name" == cp1 ]]; then
  batch_token_limit=$((context_limit * 2))
  rollout_token_limit="${MAX_BATCHED_TOKENS:-16384}"
fi
overrides=("actor_rollout_ref.model.path=$model_path" "data.max_prompt_length=$prompt_limit" "data.max_response_length=$response_limit"
  "actor_rollout_ref.rollout.max_model_len=$context_limit" "actor_rollout_ref.rollout.max_num_batched_tokens=$rollout_token_limit"
  "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$batch_token_limit" "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$batch_token_limit"
  "ray_kwargs.ray_init.runtime_env.py_executable=$repo_root/src/nl2sql_verl/worker_python.sh")
"$runtime_python" -m src.nl2sql_verl.preflight --config "$config_name" "${overrides[@]}" "$@"
exec "$runtime_python" -m src.nl2sql_verl.train --config-path "$repo_root/src/nl2sql_verl/configs" --config-name "$config_name" "${overrides[@]}" "$@"
