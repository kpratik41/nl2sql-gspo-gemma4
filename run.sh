#!/usr/bin/env bash
# One-touch BIRD test-set run.
#
#   bash run.sh
#
# That is the whole submission: it takes the raw test inputs and produces the
# file to submit. Every parameter already defaults to the validated setting, so
# nothing needs to be passed or edited.
#
# Inputs expected under the repository root (supplied by the BIRD team):
#   data/bird_test_data/raw/test.json
#   data/bird_test_data/raw/column_meaning.json
#   databases/test_databases/
#
# Produces:
#   outputs/bird_test_pipeline/self_consistency/predict_test.json   <- submit this
#   outputs/bird_test_pipeline/run_manifest.tsv                     <- what was produced
#   outputs/bird_test_pipeline/pipeline.log                         <- full log
#
# The run is resumable. Rerunning this same command after an interruption skips
# finished stages and resumes generation from the candidate it reached; it does
# not repeat completed work.
#
# MODEL_PATH defaults to the primary submitted checkpoint
# (pratikkakkar/gemma-4-31b-it-bird-sft-rl). To score the second submitted model,
# pass it as the first argument:
#   bash run.sh pratikkakkar/gemma-4-31b-it-bird-rl   # the RL-only checkpoint
#   bash run.sh /path/to/local/weights                # local weights
#
# Any other setting is overridden through the environment, e.g.
#   NUM_GENERATIONS=1 TEMPERATURE=0.0 bash run.sh     # quick single-sample pass
#   RUN_ROOT=outputs/my_run bash run.sh               # write elsewhere
# See README.md, "Configuration", for the full list.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

# A leading non-flag argument is the checkpoint. Taking it positionally keeps the
# common case to one word, while MODEL_PATH from the environment still wins if
# the caller set it explicitly.
if [[ $# -gt 0 && "$1" != -* ]]; then
  export MODEL_PATH="$1"
  shift
fi

exec bash scripts/run_bird_test_pipeline.sh "$@"
