# Gemma 4 benchmark results

Updated: 2026-10-22 12:59 EDT.

**Current result:** standalone attention checks pass through 40,960 tokens, with linear measured kernel working memory. Full-model Flex/SDPA numerical agreement has a blocker; its latency measurements are execution diagnostics. CP2, backward, and RL remain untested.

## Scope

Executed on one RTX PRO 6000 Blackwell Server Edition (SM120), with BF16, batch size one, and CP=1. Attention inputs use Gemma-shaped head counts and dimensions; these are standalone kernels, not AutoModel or verl integration. Model tests load the real cached Gemma 4 31B checkpoint, use no KV cache, and request last-position logits. No backward, optimizer, distributed CP, or RL claim is made.

## Environment and method

- Python 3.13.12; PyTorch 2.10.0+cu130; Triton 3.6.0; Transformers 5.18.0; FFPA 0.2.4; CuTeDSL 4.6.2.
- Isolated `.venv-bench` overlays `/opt/pytorch/lib/python3.13/site-packages` through a `.pth` file; the existing project environment was left intact. This environment depends on that base installation.
- Model snapshot: `842da3794eaa0b77d5f08bae87a17459d91ff475`, loaded locally without downloading weights.
- Every backend/length runs in a fresh subprocess. Attention uses five warmups and twenty measured repetitions; model tests use two and five. Median/p95 wall latency includes synchronization. Separate CUDA-event medians are in JSON.
- Attention memory is incremental peak allocated memory above live Q/K/V and mask allocations. The mask remains outside the timed attention operation; mask construction time and total allocated/reserved peaks are saved separately. Model timing includes its mask path.
- Initial mask construction was outside the attention memory counters. The eager mask builder can allocate quadratic temporary tensors; the kernel-only linear result must not be read as proof of linear end-to-end memory. Separate mask measurements are recorded below when available.
- Each attention case checks all short-reference outputs against independent FP32 eager attention, then samples queries at the start, around the 1,024-token window boundary, and at the tail against all keys. Tolerances: absolute/relative 0.02. Inputs are synthetic random tensors with explicit scale `head_dim**-0.5`; this is not a model-level semantic parity test.
- Profiler runs are separate from timed/memory measurement. JSON contains actual CUDA kernel names, attention operators, allocator retries, cold latency, and numerical errors.
- GPU device usage/driver information and dependency snapshots are in `benchmark_results/environment/`. The overlay has unresolved optional-package requirements and inherited dependency conflicts listed in `pip-check.txt`; it is a measured kernel environment, not a validated AutoModel/verl training installation.

## Measured results

| Scope | Backend | Tokens | Median ms | p95 ms | Incremental peak MiB | Total peak allocated GiB |
|---|---|---:|---:|---:|---:|---:|
| attention/global | ffpa | 1,024 | 0.591 | 0.652 | 72.12 | 0.117 |
| attention/global | flex | 1,024 | 0.451 | 0.473 | 32.25 | 0.078 |
| attention/global | ffpa | 2,048 | 1.577 | 1.610 | 144.25 | 0.227 |
| attention/global | flex | 2,048 | 1.438 | 1.514 | 64.50 | 0.149 |
| attention/global | ffpa | 4,096 | 4.596 | 4.640 | 288.50 | 0.446 |
| attention/global | flex | 4,096 | 5.094 | 5.117 | 129.00 | 0.290 |
| attention/global | ffpa | 8,192 | 15.279 | 15.344 | 577.00 | 0.884 |
| attention/global | flex | 8,192 | 19.318 | 19.329 | 258.00 | 0.572 |
| attention/global | ffpa | 16,384 | 54.530 | 56.431 | 1154.00 | 1.760 |
| attention/global | flex | 16,384 | 75.194 | 75.237 | 516.00 | 1.137 |
| attention/global | ffpa | 32,768 | 205.460 | 206.844 | 2308.00 | 3.512 |
| attention/global | flex | 32,768 | 296.428 | 297.587 | 1032.00 | 2.267 |
| attention/global | ffpa | 40,960 | 321.363 | 322.470 | 2885.00 | 4.388 |
| attention/global | flex | 40,960 | 441.637 | 444.233 | 1290.00 | 2.832 |
| attention/sliding | flex | 1,024 | 0.221 | 0.234 | 16.25 | 0.055 |
| attention/sliding | flex | 2,048 | 0.466 | 0.495 | 32.50 | 0.102 |
| attention/sliding | flex | 4,096 | 0.956 | 0.980 | 65.00 | 0.196 |
| attention/sliding | flex | 8,192 | 1.909 | 1.933 | 130.00 | 0.385 |
| attention/sliding | flex | 16,384 | 3.815 | 3.831 | 260.00 | 0.762 |
| attention/sliding | flex | 32,768 | 7.612 | 7.626 | 520.00 | 1.517 |
| attention/sliding | flex | 40,960 | 9.509 | 9.538 | 650.00 | 1.894 |
| model | eager | 128 | 73.290 | 73.323 | 28.55 | 58.287 |
| model | flex | 128 | 74.544 | 74.625 | 26.52 | 58.285 |
| model | sdpa | 128 | 71.652 | 71.742 | 31.48 | 58.289 |
| model | flex | 1,024 | 225.958 | 226.434 | 210.78 | 58.465 |
| model | sdpa | 1,024 | 244.340 | 250.596 | 251.52 | 58.504 |
| model | flex | 4,096 | 934.208 | 934.450 | 843.22 | 59.082 |
| model | sdpa | 4,096 | 1201.348 | 1202.097 | 1018.18 | 59.253 |

## Interpretation

- Global flex: measured incremental-memory fit `-0.000 + 0.031494 × tokens` MiB, R²=1.000000, over 1,024–40,960 tokens. This supports linear working-memory growth over the measured range; it does not establish whole-model scaling or linear compute.
- Global ffpa: measured incremental-memory fit `-0.000 + 0.070435 × tokens` MiB, R²=1.000000, over 1,024–40,960 tokens. This supports linear working-memory growth over the measured range; it does not establish whole-model scaling or linear compute.
- At 1,024 tokens, FFPA/Flex speedup is 0.76× (greater than one means FFPA is faster).
- At 2,048 tokens, FFPA/Flex speedup is 0.91× (greater than one means FFPA is faster).
- At 4,096 tokens, FFPA/Flex speedup is 1.11× (greater than one means FFPA is faster).
- At 8,192 tokens, FFPA/Flex speedup is 1.26× (greater than one means FFPA is faster).
- At 16,384 tokens, FFPA/Flex speedup is 1.38× (greater than one means FFPA is faster).
- At 32,768 tokens, FFPA/Flex speedup is 1.44× (greater than one means FFPA is faster).
- At 40,960 tokens, FFPA/Flex speedup is 1.37× (greater than one means FFPA is faster).

### Full-model numerical comparison

- 1,024 tokens, final-position logits: max absolute Flex/SDPA difference 6.890625, relative L2 0.276524. **Model-level numerical parity is blocked:** this difference is too large to accept as ordinary BF16 noise. Input SHA256 hashes match. Timings establish execution only; isolate layer outputs/masks and compare with eager before proceeding to longer context or training.
- 4,096 tokens, final-position logits: max absolute Flex/SDPA difference 8.406250, relative L2 0.214461. **Model-level numerical parity is blocked:** this difference is too large to accept as ordinary BF16 noise. Input SHA256 hashes match. Timings establish execution only; isolate layer outputs/masks and compare with eager before proceeding to longer context or training.

### Short eager-reference diagnostic

- At 128 identical-input tokens, sdpa versus eager final-position logits: max absolute difference 1.562500, relative L2 0.031147.
- At 128 identical-input tokens, flex versus eager final-position logits: max absolute difference 2.109375, relative L2 0.030104.

## Failures and implementation findings

- Default D512 Flex tile selection failed at compilation: requested 200,704 bytes of shared memory versus a 101,376-byte hardware limit. Explicit `BLOCK_M=32`, `BLOCK_N=32`, `num_stages=1` passed. The original failure is preserved in `benchmark_results/global/attention_global_flex_1024.json`; final comparisons use these fixed smaller tiles, not a fully autotuned Flex optimum.
- Exploratory results under `benchmark_results/global/` and `flex_tuning_1024.json` are excluded from final comparisons. A tuning probe briefly overlapped the exploratory sweep; final runs execute sequentially.
- FFPA was explicitly forced to its CuTeDSL backend with GQA enabled. Inspect the recorded kernel names to verify execution; the SM120 card uses the generic SM80-family FFPA kernel path in this package.
- Initial model results in `benchmark_results/model/` used the global RNG after model loading and showed large output differences. They are excluded; the controlled rerun confirmed the discrepancy persists with identical inputs. Final model runs use an independent generator seeded to 42 and record matching input SHA256 hashes before comparison.
- Full-model FFPA is deliberately not substituted with a generic HF monkeypatch: AutoModel registration and model dispatch still need their own validated environment.

## Separate causal-mask construction

| Mode | Tokens | Cold construction s | Peak allocated MiB | Live mask MiB |
|---|---:|---:|---:|---:|
| compiled | 1,024 | 2.684 | 0.01 | 0.00 |
| compiled | 16,384 | 24.563 | 128.50 | 0.25 |
| compiled | 4,096 | 2.693 | 0.03 | 0.02 |
| compiled | 40,960 | 43.765 | 131.13 | 1.57 |
| eager | 1,024 | 0.683 | 10.00 | 0.00 |
| eager | 16,384 | 0.665 | 2560.12 | 0.25 |
| eager | 4,096 | 0.668 | 160.01 | 0.02 |
| eager | 40,960 | 0.670 | 16000.78 | 1.57 |

At 40,960 tokens, eager construction peaked at 15.63 GiB versus 131.13 MiB with compiled construction. The final live mask was about 1.57 MiB. Compiled-mask peak time/memory still grows with shape and is not a proof of asymptotically linear mask storage. Cold compiled-mask time includes compilation. Training/packed masks need separate validation. These allocations are separate from the attention table.


## Reproduction

Run `bash benchmarks/setup_env.sh` to recreate the overlay on the same PyTorch base (or set `BENCH_BASE_PYTHON`). Exact overlay pins are in `benchmarks/requirements-overlay.txt`; inherited base packages are recorded in the environment snapshot. This preserves the measured dependency limitations rather than resolving a new stack. Then run from the repository root:

```bash
.venv-bench/bin/python benchmarks/gemma_attention.py --output benchmark_results/global_final
.venv-bench/bin/python benchmarks/gemma_attention.py --layer sliding --backends flex --output benchmark_results/sliding_final
.venv-bench/bin/python benchmarks/gemma_attention.py --kind model --backends sdpa flex --lengths 1024 4096 --warmup 2 --repeats 5 --timeout 900 --output benchmark_results/model_final
.venv-bench/bin/python benchmarks/gemma_attention.py --kind model --backends eager sdpa flex --lengths 128 --warmup 1 --repeats 3 --output benchmark_results/model_reference
.venv-bench/bin/python benchmarks/mask_memory.py
.venv-bench/bin/python benchmarks/summarize.py
```

Use `--help` for shorter sweeps and model-path overrides. Preserve original result directories when collecting a new run. The orchestrator stops increasing length for a backend after a failure/timeout; inspect JSON/log files, not only its console output.

## Next gates

1. Resolve a fully pinned AutoModel environment and reuse its exact dense/varlen FFPA entry points; verify model-level output agreement before claiming model FFPA speedups.
2. Validate model forward at longer lengths with controlled output allocations. Investigate any model-loader/kernel failures below before extending context.
3. Resolve model-level Flex/SDPA numerical disagreement with short eager-reference, per-layer hidden-state, mask, and attention-output comparisons. Keep model performance results diagnostic until this passes.
4. Add cached generation as a separate test, then attention/model backward checks with numerical references.
5. Run real CP=2 parity and communication tests when two GPUs are available. A single GPU cannot establish this result.
6. Integrate the validated path into verl; SQL rewards and task scripts follow the systems gates.

Follow-up same-weight teacher-forced and real-QKV FP32 investigation: [GEMMA4_MODEL_PARITY.md](GEMMA4_MODEL_PARITY.md). BF16 eager is a candidate reference, not exact ground truth.

Roadmap: [GEMMA4_LONG_CONTEXT_RL.md](GEMMA4_LONG_CONTEXT_RL.md). Raw measurements: [benchmark_results](benchmark_results/).

![Measured global attention memory, latency, and speedup](benchmark_results/global_attention.png)
