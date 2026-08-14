# Qwen3.8 Eval Plan

## Current Compatibility Notes

- Model: `Qwen/Qwen3.8-27B`
- Local snapshot: `/home/ec2-user/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- Local env: `transformers 5.8.1`, `vllm 0.19.1`
- vLLM recipe says Qwen3.8-27B needs `transformers >= 5.8.0`; current env satisfies this.
- vLLM recipe shows `--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder`.
- Local vLLM also has `qwen3_xml`; because the shipped chat template is XML-style, smoke both parsers before full eval.
- The shipped template can render empty historical `<think></think>` blocks when empty reasoning is preserved. The Qwen runner on this branch passes `chat_template_kwargs: {"enable_thinking": false, "preserve_thinking": false}` by default to avoid that prompt drift.

## Recommended Run Order

1. Run a 20-example smoke with `qwen3_coder`.
2. Run the same 20-example smoke with `qwen3_xml`.
3. Compare EX, pred SQL extraction/execution, zero-tool/no-tool behavior, forced-final count, and avg tool calls.
4. Run all 89 `california_schools` examples with the better parser.
5. If California is sane, run the full 1534 examples.

## Commands

### 20-example smoke, qwen3_coder

```bash
mkdir -p logs/qwen38_27b_smoke20_tp2_shards4_temp0_openai_tool_qwen3_coder
screen -dmS qwen38_smoke20_coder bash -lc 'cd /home/ec2-user/consensus/nl2sql-gspo-gemma4 && \
  TOTAL=20 \
  RUN_ROOT=outputs/inference/dev/old-dev-schema-tool-unpatched/Qwen3.8-27B/smoke20_tp2_shards4_temp0_openai_tool_qwen3_coder \
  LOG_DIR=logs/qwen38_27b_smoke20_tp2_shards4_temp0_openai_tool_qwen3_coder \
  TOOL_CALL_PARSER=qwen3_coder \
  CONCURRENCY_PER_SHARD=4 \
  scripts/qwen/run_qwen38_27b_olddev_tp2_shards4.sh \
  > logs/qwen38_27b_smoke20_tp2_shards4_temp0_openai_tool_qwen3_coder/orchestrator.log 2>&1'
```

### 20-example smoke, qwen3_xml

```bash
mkdir -p logs/qwen38_27b_smoke20_tp2_shards4_temp0_openai_tool_qwen3_xml
screen -dmS qwen38_smoke20_xml bash -lc 'cd /home/ec2-user/consensus/nl2sql-gspo-gemma4 && \
  TOTAL=20 \
  RUN_ROOT=outputs/inference/dev/old-dev-schema-tool-unpatched/Qwen3.8-27B/smoke20_tp2_shards4_temp0_openai_tool_qwen3_xml \
  LOG_DIR=logs/qwen38_27b_smoke20_tp2_shards4_temp0_openai_tool_qwen3_xml \
  TOOL_CALL_PARSER=qwen3_xml \
  CONCURRENCY_PER_SHARD=4 \
  scripts/qwen/run_qwen38_27b_olddev_tp2_shards4.sh \
  > logs/qwen38_27b_smoke20_tp2_shards4_temp0_openai_tool_qwen3_xml/orchestrator.log 2>&1'
```

### California schools, selected parser

Replace `qwen3_coder` with the selected parser if needed.

```bash
mkdir -p logs/qwen38_27b_california_schools_tp2_shards4_temp0_openai_tool_qwen3_coder
screen -dmS qwen38_cali_coder bash -lc 'cd /home/ec2-user/consensus/nl2sql-gspo-gemma4 && \
  TOTAL=89 \
  INPUT_FILE=outputs/old-dev-schema-tool-unpatched-california_schools.jsonl \
  RUN_ROOT=outputs/inference/dev/old-dev-schema-tool-unpatched/Qwen3.8-27B/california_schools_tp2_shards4_temp0_openai_tool_qwen3_coder \
  LOG_DIR=logs/qwen38_27b_california_schools_tp2_shards4_temp0_openai_tool_qwen3_coder \
  TOOL_CALL_PARSER=qwen3_coder \
  CONCURRENCY_PER_SHARD=4 \
  scripts/qwen/run_qwen38_27b_olddev_tp2_shards4.sh \
  > logs/qwen38_27b_california_schools_tp2_shards4_temp0_openai_tool_qwen3_coder/orchestrator.log 2>&1'
```

### Full 1534, selected parser

Replace `qwen3_coder` with the selected parser if needed.

```bash
mkdir -p logs/qwen38_27b_full1534_tp2_shards4_temp0_openai_tool_qwen3_coder
screen -dmS qwen38_full1534_coder bash -lc 'cd /home/ec2-user/consensus/nl2sql-gspo-gemma4 && \
  TOTAL=1534 \
  INPUT_FILE=outputs/old-dev-schema-tool-unpatched.jsonl \
  RUN_ROOT=outputs/inference/dev/old-dev-schema-tool-unpatched/Qwen3.8-27B/full1534_tp2_shards4_temp0_openai_tool_qwen3_coder \
  LOG_DIR=logs/qwen38_27b_full1534_tp2_shards4_temp0_openai_tool_qwen3_coder \
  TOOL_CALL_PARSER=qwen3_coder \
  CONCURRENCY_PER_SHARD=4 \
  scripts/qwen/run_qwen38_27b_olddev_tp2_shards4.sh \
  > logs/qwen38_27b_full1534_tp2_shards4_temp0_openai_tool_qwen3_coder/orchestrator.log 2>&1'
```
