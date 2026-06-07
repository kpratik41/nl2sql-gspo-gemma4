# Copilot Instructions for NL2SQL Inference

This repository is inference-focused. Do not add training launchers, trainer subclasses, reward functions, DeepSpeed/FSDP configs, or W&B-specific workflow code unless the user explicitly asks to restore training support.

Primary runtime paths:

- `scripts/launch_inference.sh`: shell launcher for BIRD inference/evaluation
- `scripts/run_inference_bird.py`: vLLM and async vLLM generation plus BIRD-style scoring
- `scripts/run_passk_bird.py`: pass@k generation/evaluation
- `scripts/run_self_consistency_bird.py`: execution-result self-consistency
- `src/nl2sql_gspo/inference_tool_executor.py`: Gemma-style tool-call parsing/execution
- `src/nl2sql_gspo/sql_utils.py`: SQL extraction, SQLite execution, and BIRD result matching
- `src/nl2sql_gspo/data.py`: input row normalization
- `src/nl2sql_gspo/model_utils.py`: inference model/tokenizer loading
- `src/nl2sql_gspo/tool_calling.py`: tool schemas and prompt catalog
- `scripts/data_generation/`: dev/inference data preparation utilities

The target task is NL2SQL: given a natural-language question, schema, optional hint/evidence, and optional tool access, generate valid SQLite SQL inside the expected final-answer shape.

Keep generated `outputs/`, local model files, and database paths out of source-level assumptions where possible. Prefer configurable paths and preserve compatibility with BIRD-style `dev_databases` layouts.
