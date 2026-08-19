# Place the BIRD test inputs here

This directory is intentionally shipped empty. The pipeline reads the test split
from here, so copy the three files provided with the test release into this
directory before running:

```text
data/bird_test_data/raw/test.json             questions; the "SQL" field is empty
data/bird_test_data/raw/test_tables.json      schema description
data/bird_test_data/raw/column_meaning.json   per-column descriptions (REQUIRED)
```

The SQLite databases go in `databases/test_databases/` at the repository root.

`scripts/run_bird_test_pipeline.sh` checks for all of these during preflight and
stops with a clear message naming the missing file, before loading the model.

A fourth file, `test-few-shot.json`, is **generated** here by stage 0 of the
pipeline; you do not need to supply it.
