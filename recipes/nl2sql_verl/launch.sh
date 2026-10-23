#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
runtime_python="${VERL_PYTHON:-$repo_root/.venv-verl/bin/python}"
if ! command -v "$runtime_python" >/dev/null 2>&1; then
  echo "Missing verl Python: $runtime_python. Run recipes/nl2sql_verl/setup_env.sh or set VERL_PYTHON." >&2
  exit 1
fi
runtime_python="$("$runtime_python" -c 'import sys; print(sys.executable)')"
config_name="${RECIPE_CONFIG:-smoke}"
train_input="${TRAIN_PARQUET:-$repo_root/data/nl2sql/train.parquet}"
val_input="${VAL_PARQUET:-$repo_root/data/nl2sql/validation.parquet}"
export NL2SQL_DATABASE_DIR="${NL2SQL_DATABASE_DIR:-$repo_root/databases}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$repo_root${VERL_SOURCE:+:$VERL_SOURCE}${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p data/nl2sql databases outputs/verl_data outputs/verl_recipe_checks
if [[ "${CONFIG_ONLY:-0}" == 1 ]]; then
  exec "$runtime_python" -m recipes.nl2sql_verl.preflight --config "$config_name" --config-only --dump "$@"
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
export NL2SQL_LENGTH_MAX="$response_limit"
export NL2SQL_LENGTH_BUFFER="${NL2SQL_LENGTH_BUFFER:-512}"
"$runtime_python" -m recipes.nl2sql_verl.prepare_data --input "$train_input" --output outputs/verl_data/train.parquet --model "$model_path" --max-prompt-length "$prompt_limit"
"$runtime_python" -m recipes.nl2sql_verl.prepare_data --input "$val_input" --output outputs/verl_data/validation.parquet --model "$model_path" --max-prompt-length "$prompt_limit"
context_limit=$((prompt_limit + response_limit))
overrides=("actor_rollout_ref.model.path=$model_path" "data.max_prompt_length=$prompt_limit" "data.max_response_length=$response_limit"
  "actor_rollout_ref.rollout.max_model_len=$context_limit" "actor_rollout_ref.rollout.max_num_batched_tokens=$context_limit"
  "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$context_limit" "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$context_limit"
  "ray_kwargs.ray_init.runtime_env.py_executable=$runtime_python")
"$runtime_python" -m recipes.nl2sql_verl.preflight --config "$config_name" "${overrides[@]}" "$@"
exec "$runtime_python" -m verl.trainer.main_ppo --config-path "$repo_root/recipes/nl2sql_verl/configs" --config-name "$config_name" "${overrides[@]}" "$@"
