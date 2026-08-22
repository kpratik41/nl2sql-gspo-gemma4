# Dev-set predictions from the two submitted checkpoints

These are the self-consistency outputs our pipeline produced on the **BIRD dev
set** (1534 questions), in the official submission format
(`SQL\t----- bird -----\tdb_id`, keyed by question id). They are included so the
reported dev numbers can be verified without re-running generation.

| file | model | dev EX |
| --- | --- | ---: |
| `dev_predictions_sft_rl.json` | `pratikkakkar/gemma-4-31b-it-bird-sft-rl` | **74.51%** (1143/1534) |
| `dev_predictions_rl_only.json` | `pratikkakkar/gemma-4-31b-it-bird-rl` | 73.86% (1133/1534) |

Both were produced by `bash run.sh` at its default settings: pass@16 sampling at
temperature 1.2, followed by a majority vote over execution result sets.

To score them:

```bash
python scripts/eval_bird_ex.py \
  --predictions predictions/dev_predictions_sft_rl.json \
  --gold data/bird_dev_data/raw/bird_dev.json \
  --database_dir databases/dev_databases
```

Two of the 1534 dev gold queries do not finish inside the 30-second
`--meta_time_out`. Under BIRD semantics they are unscorable and count as wrong
for every system, so the practical ceiling on dev is 1532.

**These are dev predictions, not test predictions.** The test-set file is
produced by running the pipeline against the test inputs, and is written to
`outputs/bird_test_pipeline/self_consistency/predict_test.json`.
