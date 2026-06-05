# Copilot Instructions for BIRD NL2SQL Inference

## Project Overview

This branch is inference-only for BIRD-style NL2SQL. It supports runtime prompt construction, vLLM/OpenAI-compatible generation, pass@k sampling, self-consistency selection, local dev evaluation, and BIRD leaderboard submission artifacts.

Model-update code has been removed from this branch. Do not add fit/fine-tune launchers, trainer classes, optimization loops, or restart/checkpoint training workflows unless the user explicitly asks to restore that functionality.

The target task is NL2SQL: given a natural-language question, schema, optional evidence/hint, optional few-shot examples, and optional tool access, generate valid SQLite SQL.

## Source Of Truth

Prefer the active inference entrypoints and shared prompt code over older docs or historical artifacts:

- Runtime prompt builder: `src/nl2sql_gspo/prompt_builder.py`
- Temp-0 inference and BIRD prediction writing: `scripts/run_inference_bird.py`
- Pass@k candidate generation: `scripts/run_passk_bird.py`
- Self-consistency generation/selection: `scripts/run_self_consistency_bird.py`
- Inference launcher: `scripts/launch_inference.sh`
- SQL/database helpers: `src/nl2sql_gspo/sql_utils.py`
- Tool definitions/wrappers: `src/nl2sql_gspo/tool_calling.py`
- Inference tool execution: `src/nl2sql_gspo/inference_tool_executor.py`
- Prompt templates: `prompts.py`
- Replica skill headers: `prompts_suffix_idea.py`
- Schema/few-shot generation utilities: `scripts/data_generation/`
- Tests: `tests/test_core.py`

Generated files under `outputs/` and `logs/` are workflow artifacts, not source code. Do not generate nested `outputs/outputs/...` paths.

## Repository Focus

Expected active layout:

```text
submission/
├── .github/copilot-instructions.md
├── data/
│   └── bird_dev_data/raw/
├── databases/
│   └── dev_databases/
├── scripts/
│   ├── launch_inference.sh
│   ├── run_inference_bird.py
│   ├── run_passk_bird.py
│   ├── run_self_consistency_bird.py
│   └── data_generation/
├── src/nl2sql_gspo/
├── tests/test_core.py
├── prompts.py
├── prompts_suffix_idea.py
├── outputs/
└── README.md
```

Raw train data may still exist for reproducibility or few-shot retrieval, but it is data only. Do not infer that training code should exist because train data or historical result files are present.

## Prompt-Time Pipeline

The preferred architecture is:

```text
raw BIRD row
  -> PromptConfig
  -> SchemaCache[(db_id, schema flags, database_dir, meanings_file)]
  -> render user prompt
  -> choose system prompt/template
  -> attach tools/tool allowlist
  -> inference
```

Use `PromptBuilder` for new prompt-related work. It should be shared by temp-0 inference, pass@k, and self-consistency so all modes render prompts consistently.

`SchemaCache` exists because DB introspection, column comments, stats, and top values are the expensive part. Preserve cache keys that include the DB id, database directory, meanings file, and schema-rendering flags. Do not rebuild schema text once per candidate.

Prebuilt JSONL prompts remain supported for reproducibility. If an input row already has `prompt` or `messages`, use it directly unless runtime rebuilding is explicitly requested with `--build_prompts_at_runtime` or a raw input file is supplied.

## Runtime Prompt Flags

Keep these flags available across inference, pass@k, and self-consistency where applicable:

```text
--bird_mode dev|test
--build_prompts_at_runtime
--raw_input_file
--meanings_file
--include_column_comments / --no_column_comments
--include_fewshots / --no_fewshots
--include_stats / --no_stats
--include_nullability / --no_nullability
--example_num
--fewshot_train_file
--fewshot_top_n
--tool_mode none|default|consensus
--prompt_template default|consensus
```

Launcher environment variables in `scripts/launch_inference.sh` should map cleanly to these CLI flags.

## Inputs And Defaults

Use old-dev defaults for local dev work:

- Raw dev rows: `data/bird_dev_data/raw/bird_dev.json`
- Few-shot dev rows: `data/bird_dev_data/raw/bird_dev-few-shot.json`
- Dev databases: `databases/dev_databases`
- Standard output root: `outputs/`

Do not restore defaults that point to deleted `dev_20251106.json` or `dev_20251106-few-shot.json` files. Do not add code that writes `outputs/outputs/...`.

## Sample Plans

`prompts_suffix_idea.py` defines prompt-variation headers used by sampled inference:

- default / no prefix
- `DECOMPOSE-FIRST`
- `DIRECT-CODER`
- `EXPLORE-HEAVY`
- `CONSERVATIVE`

These are replica-level prompt variations for pass@k and self-consistency, not training data generation.

Use `--sample_plan` to choose counts and decoding temperatures:

```text
default:16@0.8,decompose-first:4@0.8,default:1@0.0
```

The compact format is `skill:count@temperature` with optional top-p as `skill:count@temperature/top_p`.

Temp-0 inference should default to no skill header. Candidate rows should record `sample_plan_id`, `skill_id`, `skill_name`, `temperature`, `top_p`, and `replica_label`.

For pass@k/self-consistency, build the prompt once per `(example, skill_name)` and reuse it for all candidates using that skill.

## Inference Modes

Temp-0 inference:

- Entry: `scripts/run_inference_bird.py`
- Launch wrapper: `scripts/launch_inference.sh`
- Uses `temperature=0` unless overridden.
- Writes per-example results and BIRD prediction JSON.

Pass@k:

- Entry: `scripts/run_passk_bird.py`
- Generates multiple candidates per example.
- Intended to measure candidate-set coverage and inspect failure diversity.
- Should preserve candidate metadata including `sample_id`, `skill_id`, `skill_name`, SQL text, db id, and any execution result fields.

Self-consistency:

- Entry: `scripts/run_self_consistency_bird.py`
- Generates or consumes candidates, executes them where possible, groups compatible SQL outputs, and selects a final answer.
- Should not rebuild prompts per candidate.

## BIRD Dev/Test Behavior

`--bird_mode dev`:

- Gold SQL may be present.
- Local execution evaluation is allowed when gold SQL exists.
- Prediction file should be named `predict_dev.json`.

`--bird_mode test`:

- Test rows contain no usable gold SQL.
- Do not require `SQL`/`gold_sql`.
- Do not run local execution-accuracy evaluation.
- Write `predict_test.json`.

Official-style prediction files are JSON objects keyed by example index/string id. Each value must be exactly:

```text
<SQL>\t----- bird -----\t<db_id>
```

Example:

```json
{
  "0": "SELECT COUNT(*) FROM players;\t----- bird -----\tnba_data"
}
```

Expected BIRD test input layout:

```text
test_databases/
test_tables.json
column_meaning.json
test.json
```

For this branch, BIRD leaderboard preparation means producing the correct test predictions from a single model path and preserving enough metadata to explain inference settings.

## Tools

Tool declarations live in `src/nl2sql_gspo/tool_calling.py`. Inference-time tool execution lives in `src/nl2sql_gspo/inference_tool_executor.py`.

Supported prompt/tool modes should remain explicit:

- `tool_mode=none`: no tools attached.
- `tool_mode=default`: attach the normal SQL helper tools.
- `tool_mode=consensus`: attach consensus-oriented tool definitions and matching prompt template.

Keep tool schemas OpenAI/Gemma-compatible. Avoid silently changing tool names, argument shapes, or execution semantics because old outputs and evaluation scripts may depend on them.

## Data Generation Utilities

`scripts/data_generation/` is still useful for inference prompt preparation:

- few-shot retrieval artifacts
- schema rendering
- column meanings/comments
- stats/top-values/nullability rendering

Treat these scripts as prompt/data preparation utilities, not model training code. If adding new prompt-generation behavior, prefer exposing it through `PromptConfig` and `PromptBuilder` so inference can choose it at runtime.

## Output Conventions

Use `outputs/` directly:

- Good: `outputs/old-dev-schema-tool-unpatched.jsonl`
- Bad: `outputs/outputs/old-dev-schema-tool-unpatched.jsonl`

For new runs, make output paths explicit and stable enough to compare configurations. Include prompt metadata, tool mode, schema flags, bird mode, and skill-header mode in summaries where practical.

## Testing And Validation

Prefer focused tests for shared inference behavior:

- `PromptConfig` construction
- raw dev row builds prompt plus `gold_sql`
- raw test row with empty/no SQL does not fail
- runtime BM25 few-shots attach to raw rows
- schema flags change rendered content
- `SchemaCache` reuses DB/schema renders
- `tool_mode` attaches expected tools
- sample plans expand deterministically and attach expected metadata
- prebuilt JSONL prompts still work
- test mode writes `predict_test.json` and skips local evaluation

Useful commands:

```bash
PYTHONPATH=src:. python3 -m unittest tests/test_core.py
python3 -m py_compile src/nl2sql_gspo/prompt_builder.py scripts/run_inference_bird.py scripts/run_passk_bird.py scripts/run_self_consistency_bird.py
rg -n "outputs/outputs|dev_20251106\\.json|dev_20251106-few-shot\\.json" scripts src tests README.md .github || true
```

## Guardrails

- Do not reintroduce training code or training docs into this branch.
- Do not treat raw train data as permission to add training workflows.
- Do not include assistant gold SQL in inference prompts.
- Do not require gold SQL in BIRD test mode.
- Do not run local dev evaluation in BIRD test mode.
- Do not silently change the BIRD prediction JSON value format.
- Do not create nested output directories under `outputs/outputs`.
- Do not change SQL execution/evaluation semantics without updating tests and README notes.
