# NL2SQL BIRD Submission Pipeline

This branch is focused on BIRD dev/test inference, candidate sampling, self-consistency, and submission artifact generation. Training code has been removed; raw train/dev data is intentionally left in place for prompt construction, few-shot retrieval, and reproducibility checks.

## BIRD Data Notes

The current public BIRD dev release is the updated `dev_20251106` split. Local status:

- `data/bird_dev_data/raw/bird_dev.json` is the older dev file and is not the same content as the updated public `dev_20251106` file.
- `data/bird_dev_data/raw/dev_20251106.json` has the same rows as the public Hugging Face `birdsql/bird_sql_dev_20251106` file, but local row order differs from the official file.
- For leaderboard-oriented runs, prefer `dev_20251106` inputs and keep `question_id`/official ordering checks explicit.

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
  --output outputs/dev-20251106-schema.jsonl
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
INPUT_FILE=outputs/dev-20251106-schema.jsonl \
DATABASE_DIR=databases/dev_databases \
DIFF_JSON_PATH=data/bird_dev_data/raw/dev_20251106.json \
INFERENCE_BACKEND=vllm \
TEMPERATURE=0.0 \
MAX_PROMPT_LENGTH=34000 \
MAX_NEW_TOKENS=8000 \
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
  --input_file outputs/dev-20251106-schema-bare-tool.jsonl \
  --database_dir databases/dev_databases \
  --diff_json_path data/bird_dev_data/raw/dev_20251106.json \
  --output_dir outputs/passk/dev_20251106/model \
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
  --input_file outputs/dev-20251106-schema-bare-tool.jsonl \
  --database_dir databases/dev_databases \
  --diff_json_path data/bird_dev_data/raw/dev_20251106.json \
  --output_dir outputs/self_consistency/dev_20251106/model \
  --num_generations 16 \
  --temperature 0.7 \
  --overwrite
```

Self-consistency generates multiple candidates, executes them, discards failed or empty-result candidates, groups the rest by raw execution result set, and selects the majority result. Ties break by earliest sample index and then shorter SQL.

## Submission Readiness Checklist

- Validate that the dev/test input order and prediction keys match the official BIRD release being submitted.
- Record whether the run uses old dev or `dev_20251106`; new submissions should explicitly identify updated dev usage.
- Produce both raw `predict_dev.json`/`predict_test.json` and an audit bundle with run config, prompt config, model path, decoding parameters, and SQL extraction failures.
- Keep pass@k and self-consistency candidate counts in the report; BIRD’s single-trained-model track categorizes self-consistency by number of candidates.
- Treat prompt features as config: column definitions, few-shots, stats/examples, nullability, tool inclusion, and specific tool allowlists.
