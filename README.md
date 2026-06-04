# NL2SQL BIRD Submission Pipeline

This branch is focused on BIRD dev/test inference, candidate sampling, self-consistency, and submission artifact generation. Training code has been removed; raw train/dev data is intentionally left in place for prompt construction, few-shot retrieval, and reproducibility checks.

## BIRD Data Notes

This checkout defaults to the older BIRD dev files that are currently present:

- `data/bird_dev_data/raw/bird_dev.json`
- `data/bird_dev_data/raw/bird_dev-few-shot.json`

The newer public `dev_20251106` files are not kept in this branch. For leaderboard-oriented runs, pass the exact split files through launch/config arguments instead of relying on implicit defaults. This keeps old-dev experiments and newer-dev/test submissions from quietly mixing.

Official BIRD-style predictions use:

```text
SQL\t----- bird -----\tdb_id
```

stored in a JSON object keyed by example index/id, for example `predict_dev.json`.

## Build Schema Prompts

Generate schema-augmented dev prompts:

```bash
python3 scripts/data_generation/schema_build.py \
  --split dev \
  --input-file data/bird_dev_data/raw/bird_dev-few-shot.json \
  --output outputs/old-dev-schema.jsonl
```

Useful prompt/config flags:

- `--no-comments`: omit `column_meaning.json` column definitions.
- `--no-fewshots`: omit BM25 few-shot examples.
- `--no-stats`: omit column stats and sampled values.
- `--no-nullability`: omit nullable/not-null labels.
- `--example-num N`: control the number of column examples.
- `--messages-only`: write legacy chat messages only.

Tool-oriented prompt files are produced by the tool dataset generation scripts and are consumed by the same inference runner when a row contains a `tools` field.

## Run Inference

```bash
bash scripts/launch_inference.sh
```

Common overrides:

```bash
MODEL_PATH=/path/to/model-or-checkpoint \
INPUT_FILE=outputs/old-dev-schema.jsonl \
DATABASE_DIR=databases/dev_databases \
DIFF_JSON_PATH=data/bird_dev_data/raw/bird_dev.json \
INFERENCE_BACKEND=vllm \
TEMPERATURE=0.0 \
MAX_PROMPT_LENGTH=34000 \
MAX_NEW_TOKENS=8000 \
bash scripts/launch_inference.sh
```

Runtime prompt generation is also supported. This lets inference build prompts
directly from raw BIRD rows while caching schema introspection per database:

```bash
MODEL_PATH=/path/to/model-or-checkpoint \
BUILD_PROMPTS_AT_RUNTIME=1 \
RAW_INPUT_FILE=data/bird_dev_data/raw/bird_dev-few-shot.json \
INPUT_FILE=data/bird_dev_data/raw/bird_dev-few-shot.json \
DATABASE_DIR=databases/dev_databases \
DIFF_JSON_PATH=data/bird_dev_data/raw/bird_dev.json \
TOOL_MODE=default \
PROMPT_TEMPLATE=default \
INCLUDE_COLUMN_COMMENTS=1 \
INCLUDE_FEWSHOTS=1 \
INCLUDE_STATS=1 \
INCLUDE_NULLABILITY=1 \
EXAMPLE_NUM=3 \
bash scripts/launch_inference.sh
```

For BIRD test submission, use test mode and the raw test input. Test mode writes
`predict_test.json` and skips local EX scoring because gold SQL is not provided:

```bash
MODEL_PATH=/path/to/model-or-checkpoint \
BIRD_MODE=test \
BUILD_PROMPTS_AT_RUNTIME=1 \
RAW_INPUT_FILE=/path/to/test.json \
DATABASE_DIR=/path/to/test_databases \
MEANINGS_FILE=/path/to/column_meaning.json \
OUTPUT_DIR=outputs/test_submission_run \
bash scripts/launch_inference.sh
```

Main outputs:

- `predict_dev.json`: official BIRD prediction format.
- `prediction_details.jsonl`: full completions and extracted SQL.
- `eval_results.jsonl`: per-example BIRD execution results.
- `eval_summary.json` / `eval_summary.md`: EX accuracy by difficulty and database.
- `run_report.md` / `per_example_report.csv`: operational debugging report.

The local evaluator follows BIRD EX semantics: execute predicted and gold SQL on SQLite and compare raw result sets with `set(pred_rows) == set(gold_rows)`.

## pass@k

```bash
python3 scripts/run_passk_bird.py \
  --model_name_or_path /path/to/model-or-checkpoint \
  --raw_input_file data/bird_dev_data/raw/bird_dev-few-shot.json \
  --build_prompts_at_runtime \
  --tool_mode default \
  --skill_headers cycle \
  --database_dir databases/dev_databases \
  --diff_json_path data/bird_dev_data/raw/bird_dev.json \
  --output_dir outputs/passk/old_dev/model \
  --num_generations 16 \
  --temperature 0.8 \
  --overwrite
```

The pass@k script samples `num_generations` candidates per example, scores every candidate with BIRD EX, and reports both:

- `pass_at_k_estimated`: combinatorial estimated pass@k from `n` samples and `c` correct samples.
- `prefix_pass_at_k`: whether the first `k` sampled candidates contain at least one correct SQL.

## Self-Consistency

```bash
python3 scripts/run_self_consistency_bird.py \
  --model_name_or_path /path/to/model-or-checkpoint \
  --raw_input_file data/bird_dev_data/raw/bird_dev-few-shot.json \
  --build_prompts_at_runtime \
  --tool_mode default \
  --skill_headers cycle \
  --database_dir databases/dev_databases \
  --diff_json_path data/bird_dev_data/raw/bird_dev.json \
  --output_dir outputs/self_consistency/old_dev/model \
  --num_generations 16 \
  --temperature 0.7 \
  --overwrite
```

Self-consistency generates multiple candidates, executes them, discards failed or empty-result candidates, groups the rest by raw execution result set, and selects the majority result. Ties break by earliest sample index and then shorter SQL.

When `--skill_headers cycle` is enabled, sampled candidates cycle through the prompt prefixes in `prompts_suffix_idea.py`: default, decompose-first, direct-coder, explore-heavy, and conservative. Candidate JSONL outputs include `skill_id` and `skill_name`.

## Submission Readiness Checklist

- Validate that the dev/test input order and prediction keys match the official BIRD release being submitted.
- Record whether the run uses old dev, updated dev, or test; submissions should explicitly identify updated-dev usage when applicable.
- Produce both raw `predict_dev.json`/`predict_test.json` and an audit bundle with run config, prompt config, model path, decoding parameters, and SQL extraction failures.
- Keep pass@k and self-consistency candidate counts in the report; BIRD’s single-trained-model track categorizes self-consistency by number of candidates.
- Treat prompt features as config: column definitions, few-shots, stats/examples, nullability, tool inclusion, and specific tool allowlists.

## BIRD Test Submission Notes

The public BIRD site points submitters to the Google Docs submission guideline and asks teams to contact `bird.bench23@gmail.com` for test evaluation. The guideline asks for a concise code package, a detailed README with commands, `requirements.txt`, model weights or API keys as applicable, and dev-set predicted SQLs for reproducibility.

The BIRD test input layout is described as analogous to dev:

- `test_databases/`
- `test_tables.json`
- `column_meaning.json`
- `test.json`, with the same shape as dev but no gold SQL (`SQL` is empty)

For test submission, make sure the inference path never depends on gold SQL being present. The produced SQL file should follow the BIRD prediction convention used by the official evaluator: `predict_test.json` maps each example key to `SQL\t----- bird -----\tdb_id`.

Example `predict_test.json` entry:

```json
{
  "0": "SELECT COUNT(*) FROM players;\t----- bird -----\tnba_data"
}
```
