# Copilot Instructions for NL2SQL GSPO Training Project

## Project Overview

This repository implements reinforcement learning training for a large language model on an NL2SQL task using:

- TRL `GRPOTrainer`
- GSPO-style sequence-level importance sampling
- vLLM server mode for rollouts
- DeepSpeed ZeRO-3 for distributed training
- BIRD-style NL2SQL train/dev data
- SQLite execution-based rewards

The current target model is a Gemma 4 31B multimodal model. For this project, only text input and text output are used. Vision/audio/multimodal modules should be frozen if present. The language model decoder should remain trainable unless explicitly switching to LoRA/adapters.

The task is NL2SQL: given a natural language question, schema, and optional evidence/hint, generate valid SQLite SQL.

---

## Repository Structure

Expected structure:

```text
nl2sql-gspo-gemma4/
├── .github/
│   └── copilot-instructions.md
├── CLAUDE.md
├── configs/
│   └── ds_zero3_bf16.json
├── scripts/
│   ├── launch_vllm.sh
│   ├── launch_train.sh
│   ├── smoke_test_rewards.py
│   ├── inspect_jsonl.py
│   └── data_generation/
│       ├── schema_build.py
│       └── few_shot_bm25.py
├── src/
│   └── nl2sql_gspo/
│       ├── __init__.py
│       ├── train_gspo_nl2sql.py
│       ├── data.py
│       ├── model_utils.py
│       ├── rewards.py
│       ├── sql_utils.py
│       └── schema_utils.py
├── tests/
│   └── test_core.py
├── data/
│   ├── bird_train_data/
│   │   ├── raw/
│   │   └── processed/
│   └── bird_dev_data/
│       ├── raw/
│       └── processed/
├── databases/
│   ├── train_databases/
│   └── dev_databases/
├── outputs/
└── logs/

Current top-level processed files such as `data/bird_train.jsonl` and `data/bird_dev.jsonl` may not exist yet. Raw split-specific folders are the current source of truth.

Data Layout

The data/ folder holds raw and processed dataset artifacts.

data/bird_train_data/raw/ and data/bird_dev_data/raw/ contain original downloaded BIRD or Hugging Face source files.
data/bird_train_data/processed/ and data/bird_dev_data/processed/ contain cleaned or converted files used for training/evaluation.

Raw train sources in this workspace are commonly JSONL, while raw dev sources may be JSON arrays.

Generated few-shot artifacts may also be stored alongside the raw split files, for example `data/bird_train_data/raw/train-6601-few-shot.jsonl` and `data/bird_dev_data/raw/dev_20251106-few-shot.json`.

The databases/ folder holds SQLite databases.

The current workspace contains 69 train databases under `databases/train_databases` and 11 dev databases under `databases/dev_databases`.

Expected database layout:

databases/
├── train_databases/
│   └── <db_id>/
│       ├── <db_id>.sqlite
│       └── database_description/
└── dev_databases/
    └── <db_id>/
        ├── <db_id>.sqlite
        └── database_description/

Training should usually point to:

--database_dir databases/train_databases

Evaluation/dev should usually point to:

--database_dir databases/dev_databases

Do not assume all databases live directly under databases/. The code supports split-based database folders and may also be passed the top-level databases/ directory.

Dataset Format

Training data is expected to be JSONL, and the code should handle both normalized chat-style records and raw BIRD-style records.

Chat/harmony-style input:

{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "SELECT ..."}
  ],
  "db_id": "database_name",
  "gold_sql": "SELECT ...",
  "evidence": "optional external knowledge or hint"
}

Raw BIRD-style input also appears in this workspace:

{
  "db_id": "database_name",
  "question": "...",
  "evidence": "optional external knowledge or hint",
  "SQL": "SELECT ..."
}

Utility scripts under `scripts/data_generation/` may need to support both raw JSONL and JSON-array dataset files.

The `scripts/data_generation/few_shot_bm25.py` utility generates both inference and training few-shot files. For dev/inference data, each example receives top-k few-shot examples retrieved from the train file. For train data, each example receives top-k few-shot examples retrieved from the train file while excluding examples from the same `db_id`.

The training prompt should contain only system/user messages. The assistant message should be used as the gold SQL target for rewards, not included in the prompt.
When normalizing raw BIRD data, preserve uppercase `SQL` fields as `gold_sql`.

The normalized dataset should expose these fields:

{
    "prompt": [...],
    "messages": [...],
    "db_id": "...",
    "gold_sql": "...",
    "evidence": "..."
}

Keep remove_unused_columns=False in TRL config because reward functions need db_id, gold_sql, evidence, and messages.

Training Method

This project uses TRL GRPOTrainer, but configured for GSPO-style training by setting:

importance_sampling_level="sequence"

This is important. Do not accidentally remove this setting.

The current intended setup is:
Training GPUs: 0,1,2,3,4,5
vLLM GPUs:     6,7
Training:      DeepSpeed ZeRO-3
Rollouts:      vLLM server mode
Precision:     bf16

The vLLM server should be started separately before training.

vLLM Server Mode

The training script expects vLLM to be running in server mode.

Typical command:
bash scripts/launch_vllm.sh

Then training:
bash scripts/launch_train.sh

To resume from a saved checkpoint through the launcher workflow, set `RESUME_FROM_CHECKPOINT` before running `bash scripts/launch_train.sh`.

The training script uses:
use_vllm=True
vllm_mode="server"
vllm_server_base_url="http://127.0.0.1:8000"

The training script also accepts an optional `--resume_from_checkpoint` argument for explicit checkpoint resume.

Do not switch to colocated vLLM unless explicitly requested. Server mode is preferred for this project because it separates rollout memory from training memory.

Model Loading Rules

The model is Gemma 4 31B or an internal equivalent.

Because Gemma 4 may be multimodal, model loading should:

Try AutoModelForCausalLM first for text-only training.
If that fails, try a multimodal class such as AutoModelForImageTextToText.
Freeze non-text modules if present.

Reward Functions

Reward functions live in: src/nl2sql_gspo/rewards.py
Supporting SQL/database utilities live in: src/nl2sql_gspo/sql_utils.py
Supporting schema utilities live in: src/nl2sql_gspo/schema_utils.py

Current reward functions include:

format_reward
execution_reward
result_reward
schema_linking_reward
ngram_reward
evidence_utilization_reward

Tests

Focused unit tests live in: tests/test_core.py

These tests currently cover:

- BIRD raw record normalization
- SQL extraction and readonly checks
- Database path resolution for split-based database folders

Documentation Maintenance

After any relevant code change, review this file for drift.

Update `.github/copilot-instructions.md` when a code change affects repo structure, workflow, assumptions, supported behavior, or tests.

Also update `README.md` when setup, usage, or run instructions change.


When Editing Code

When making code changes, prefer minimal targeted edits.

If changing database path logic, update only sql_utils.py.

If changing data format handling, update only data.py.

If changing model loading/freezing, update only model_utils.py.

If changing reward logic, update only rewards.py, sql_utils.py, or schema_utils.py.

If changing distributed training settings, update:
scripts/launch_train.sh
configs/ds_zero3_bf16.json
src/nl2sql_gspo/train_gspo_nl2sql.py

