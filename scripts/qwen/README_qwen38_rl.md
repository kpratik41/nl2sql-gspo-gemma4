# Qwen3.8-27B RL Runbook

Adapting the Gemma-4 GSPO/DAPO RL stack to `Qwen/Qwen3.8-27B`.

`Qwen3.8-27B` reports `model_type: qwen3_5`, `architectures:
[Qwen3_5ForConditionalGeneration]` — it reuses the Qwen3.5 architecture and has
no dedicated `qwen3_8.py` in vLLM, which is why a wheel predating the model
release serves it correctly.

## What differs from Gemma

| | Gemma-4-31B | Qwen3.8-27B |
|---|---|---|
| Tool-call syntax | `call:name{k:v}` | `<tool_call><function=name><parameter=k>v</parameter></function></tool_call>` |
| Tool response | `<\|tool_response>response:...` | `<tool_response>...</tool_response>` (role `tool`) |
| Attention | dense | hybrid: 16 `full_attention` + 48 `linear_attention` (64 layers) |
| FSDP wrap class | `Gemma4TextDecoderLayer` | `Qwen3_5DecoderLayer` |
| Reasoning | — | `<think>` opened every turn unless `enable_thinking=false` |
| Sampling | temp 1.2 | temp 1.0, top_p 0.95, top_k 20 |

### The bug this fixes

The trainer only dispatches tools when a completion carries a structured
`tool_calls` list, built by parsing raw text. That parser was Gemma-only, so a
Qwen rollout produced **zero** parsed calls: the tool loop never ran and the
model would train with tools silently disabled. `src/nl2sql_gspo/tool_dialects.py`
isolates the format behind a dialect; Gemma stays the default.

A second, worse trap: `AutoModelForCausalLM` resolves Qwen3.8 to the text-only
`Qwen3_5ForCausalLM`, which expects `model.*` while the checkpoint stores
`model.language_model.*`. That load *succeeds with warnings* while randomly
initializing 850 of 851 parameters. `model_utils.resolve_auto_model_class`
now selects on `config.architectures` instead of a try/except chain.

```bash
PYTHONPATH=src:. .venv/bin/python scripts/qwen/verify_qwen_tool_dialect.py   # 30 checks
```

## Environment

```bash
bash scripts/setup_qwen38_env.sh       # torch 2.10.0, transformers 5.12.1, trl 1.4.0, vllm 0.19.1
bash scripts/setup_qwen38_kernels.sh   # causal-conv1d + flash-linear-attention
```

`requirements.txt` pins the direct deps; `requirements-lock.txt` is a full
`pip freeze` for a byte-exact rebuild. Constraints: `transformers >= 5.8.0`,
`vllm >= 0.17.0`. **No special vLLM commit or Dockerfile is needed** — 0.19.1
serves this model; the day-0 Docker images circulating are for other Qwen variants.

### THE gotcha: keep `.venv/bin` on PATH

FlashInfer's GDN prefill kernel is JIT-compiled with `ninja`. Calling
`.venv/bin/python` directly — without activating the venv — leaves `.venv/bin`
off `PATH`, and the JIT fails:

```
RuntimeError: Worker failed with error '[Errno 2] No such file or directory: 'ninja''
```

EngineCore dies and **every** sample returns `generation_error` with 0 tool
calls and 0% accuracy — while the process still **exits 0**. This cost a full
eval run here. All scripts in `scripts/qwen/` now export `PATH` themselves;
if you invoke anything by hand, do the same or `source .venv/bin/activate`.

Alternative: `--gdn-prefill-backend triton` skips the FlashInfer JIT entirely
(that is what the pass@16 eval runs used). TRL's `vllm-serve` parser does not
expose that flag, so on the RL server path keep ninja on PATH.

## Attention: nothing to force

Measured on this box, with **no** `VLLM_ATTENTION_BACKEND` set:

```
Using FLASH_ATTN attention backend out of potential backends:
  ['FLASH_ATTN', 'FLASHINFER', 'TRITON_ATTN', 'FLEX_ATTENTION']
Using FlashAttention version 3
Using FlashInfer GDN prefill kernel
```

- **Rollouts (vLLM)** — FA3 auto-selected for the 16 full-attention layers. The
  wheel ships both `_vllm_fa2_C.abi3.so` and `_vllm_fa3_C.abi3.so`, and
  `is_fa_version_supported` reports True for both. Version choice is made by
  `get_flash_attn_version(head_size=...)` from GPU capability + head_dim; there
  is **no `VLLM_FLASH_ATTN_VERSION` knob in vLLM 0.19.1**. To pin FA2 you would
  have to select a different backend entirely via `VLLM_ATTENTION_BACKEND`.
  FA2 and FA3 are both numerically fine; FA3 is faster on Hopper.
- **Rollouts, linear-attention layers** — FlashInfer GDN (or Triton/FLA). These
  48 layers never use FlashAttention at all.
- **Training (transformers)** — `ATTN_IMPLEMENTATION=sdpa`. torch SDPA dispatches
  to its own FA2 backend; verified at Qwen3.8's exact shapes
  (24 q / 4 kv heads, head_dim=256, H200). The `flash-attn` **package is not
  needed** and cannot be built here (host CUDA 13.2 vs torch cu128) — see
  `scripts/setup_flash_attn.sh` for the full diagnosis.

## Step 1 — eval smoke (do this before RL)

```bash
bash scripts/qwen/run_qwen38_eval_smoke.sh
```

Verified on this box: **EX 70.00% (14/20), 37 tool calls, 1.85 calls/example**,
`stop_reason_counts {'finished': 18, 'max_tool_rounds': 2}`,
`tool_name_counts {'sqlite_query': 36, 'sqlite_peek': 1}`.

Sanity checks: `tool_call_count_total > 0` and `stop_reason_counts` not
dominated by `generation_error`.

## Step 2 — data

The system prompt is embedded in the training JSONL, so Qwen needs its own build
(the Gemma prompt teaches `call:name{...}`, and the consensus variant explicitly
forbids `<tool_call>` — the exact syntax Qwen requires):

```bash
.venv/bin/python scripts/data_generation/build_tool_dataset.py \
  --input  outputs/old-dev-schema-bare.jsonl \
  --output outputs/old-dev-schema-bare-tool-qwen.jsonl \
  --prompt-template default_qwen
```

## Step 3 — RL smoke (server mode)

```bash
# terminal 1: rollout server on GPUs 6,7
bash scripts/qwen/launch_qwen38_vllm.sh
# terminal 2: trainer on GPUs 0-5
bash scripts/qwen/run_rl_smoke_qwen38.sh
```

4 steps, 2 generations, forced eval + save. **Check `[tool-parse]` in the log:
`attached_native_tool_calls` must be non-zero.** Zero means rollouts are training
without tools.

Prefix caching is safe under RL: TRL calls `reset_prefix_cache()` after every
weight sync (`trl/generation/vllm_generation.py`), so rollouts never reuse KV
computed under stale policy weights.

`VLLM_MODE=colocate` runs vLLM in the training process instead (no server).

## In-memory async vs. server, for RL

The pass@16 script builds an `AsyncLLMEngine` in-process against **static**
weights. It has no weight-sync path, so it cannot drive RL directly: rollouts
would keep sampling from the initial checkpoint forever and the run would look
healthy while learning nothing. RL needs trainer→engine weight sync plus prefix
cache invalidation, which is what TRL's `vllm-serve` (server) and `colocate`
provide.

The *engine tuning* transfers fully, and is on by default in vLLM V1: prefix
caching, chunked prefill, CUDA graphs (`FULL_AND_PIECEWISE`), inductor compile.
Per-run `VLLM_CACHE_ROOT` / `TORCHINDUCTOR_CACHE_DIR` isolation is worth keeping
— concurrent engines race on a shared compile cache.

## Open decisions

- **Thinking mode.** The template opens `<think>` every assistant turn unless
  `enable_thinking=false`; the validated evals ran with it off. Thinking tokens
  count against `max_completion_length` and the DAPO length penalty.
- **`--gdn-prefill-backend`.** FlashInfer (default, JIT on first run) vs Triton
  (no JIT). The eval ran fine on FlashInfer once ninja was on PATH.
