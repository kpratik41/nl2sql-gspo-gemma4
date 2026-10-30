# Gemma 4 NL2SQL RL: native verl FSDP2, FlexAttention and CP=2

The objective is correct, memory-efficient long-context full-parameter RL for
`google/gemma-4-31B-it`. The current recipe uses native verl FSDP2, FlexAttention
and Ulysses CP=2, with an IEEE FP32 actor and BF16 vLLM rollout. FFPA remains an
optional optimization requiring its own numerical/backward validation.

Start with [the current runtime guide](src/nl2sql_verl/README.md),
[the dated work log](WORK_LOG.md), and
[the latest committed experiment report](src/nl2sql_verl/reports/screened1039-8k-2026-10-30.md).
The report is a timestamped snapshot, not a live job-status claim.

## Entry points

- Environment: root `requirements.txt` and
  `bash src/nl2sql_verl/setup_fsdp_env.sh` (pinned verl checkout included).
- Latest screened 8K recipe: `configs/dapo_screened_8k_16gpu.yaml`,
  `run_screened_pipeline.py` and `run_screened_training.py` under
  `src/nl2sql_verl/`. Read the guide before launching: data, databases, cluster
  topology and validation evidence are required on the destination machine.
- Current actor: `fsdp_engine.py`, `fsdp_attention.py`, `fsdp_projection.py`.
- Current native tools/rewards: `native_agent_loop.py`, `native_tools.py`,
  `reward_entry_native.py`; shared task/consensus helpers remain dependencies.
- Historical implementations: [legacy boundaries](src/nl2sql_verl/legacy/README.md).
  Original import and command paths remain as compatibility entry points.
- Historical replay reports: both `benchmarks/summarize_replay.py` and
  `benchmarks/summarize_layer_replay.py` call one schema-aware renderer.

## CPU checks and handoff

With the pinned source and runtime dependencies available:

```bash
VERL_CONFIG_DIR="$VERL_SOURCE/verl/trainer/config" \
  python -m pytest -q src/nl2sql_verl/tests
```

Set `NL2SQL_TEST_MODEL=/path/to/local/gemma/snapshot` to run the CPU native-tool
round-trip test with a locally cached tokenizer instead of the historical host path.
Optional AutoModel and unavailable reference-data checks may skip. Native import
and configuration checks do not allocate GPUs or certify distributed training.
A new cluster still needs numerical, capacity and end-to-end validation; old
readiness hashes must not be edited to bypass it.

A committed-source handoff is made with `git archive --format=zip <commit>`.
It contains tracked code, documents and evidence, including clearly separated
legacy code. Downloaded data, databases, model checkpoints and environments are
separate migration artifacts; they are not in that ZIP.
