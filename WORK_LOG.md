# Chronological work log: Gemma 4 NL2SQL training

This log records completed work, failed checks, decisions and pending work in
chronological order. New results are appended; a later passing result does not
retroactively validate an earlier experiment. Dates/times below are UTC unless
an older source explicitly uses America/New_York. Early entries are reconstructed
from Git history and the linked reports; current experiment times are artifact
completion times. Large logs, datasets, model weights and checkpoints remain local.

The objective is full-parameter Gemma 4 31B NL2SQL RL in verl, preserving Gemma's
exact dense global attention and sliding attention, with practical near-linear
attention memory and CP2. Dense global attention compute remains quadratic.
FFPA is optional; correctness takes priority over backend choice.

## 2026-09-28 — branch initialized

- Initialized the `verl` branch (`5ee6c6c`). The prior NL2SQL task recipe remains
  available on `consensus`; its existence is separate from a working verl port.

## 2026-10-08 — architecture and implementation planning

- Documented the full-parameter long-context RL objective, framework choices,
  Gemma context-parallel restrictions, and resource/recipe references.
- Investigated NeMo AutoModel rather than assuming a generic model could use
  arbitrary TP/CP layouts. CP counts GPU ranks, not nodes.
- Established incremental validation: kernels, model probabilities, distributed
  gradients, optimizer/checkpoints, rollout synchronization, then task training.
- Commits: `6b7ac51`, `732092c`, `1bbc72e`, `6101465`.
- Details: [long-context plan](GEMMA4_LONG_CONTEXT_RL.md).

## 2026-10-22 — attention benchmarks and numerical isolation

- Benchmarked the cached Gemma snapshot
  `842da3794eaa0b77d5f08bae87a17459d91ff475`. Early tests used the SM120
  environment, not the later eight-B200 allocation.
- Flex and FFPA attention working memory grew approximately linearly through
  40,960 tokens. FFPA was slower at short lengths and faster at long lengths
  in that environment; these are not B200 training-throughput claims.
- Default D512 Flex tiles exceeded shared-memory capacity. Explicit 32×32
  forward tiles and smaller backward tiles provided an executable baseline.
- Eager mask construction produced large quadratic temporaries: at 40,960
  tokens it peaked near 15.63 GiB, versus about 131 MiB for compiled construction.
  Kernel-only memory measurements therefore cannot establish end-to-end scaling.
- Full-model eager/SDPA/Flex outputs differed substantially even with identical
  weights and tokens. BF16 eager was not treated as a definitive oracle.
- Fixed-input FP32 decoder replay matched FP32 SDPA math exactly. Optimized
  BF16 layer outputs were comparable to BF16 eager. Forty-two causality checks
  and six sliding-window isolation checks passed; this did not certify CP or
  whole-model backward.
- Changing sequence/GEMM shape on one GPU reproduced important discrepancies;
  the first observed divergence was in layer-0 Q projection. cuBLASLt and
  disabling reduced-precision/split-K BF16 reductions mitigated some errors.
- Commits: `fbca939`, `c4f2a9f`.
- Evidence: [benchmarks](GEMMA4_BENCHMARK_RESULTS.md),
  [model parity](GEMMA4_MODEL_PARITY.md),
  [layer replay](GEMMA4_LAYER_REPLAY_RESULTS.md), `benchmark_results/`.

## 2026-10-23 — four-GPU CP experiments, consensus port and BIRD downloads

- Compared TP2/CP1 against TP2/CP2 with model-owned NeMo Gemma sharding.
  The 128-token probability check failed its maximum-token gate (0.356054);
  the 1,024-token case passed (0.069377). Top-1 agreement alone concealed
  differences relevant to RL probability ratios.
- Forward-only capacity with all-token vocabulary-sharded logits reached
  104,448 tokens for TP2/CP1 and 199,680 for TP2/CP2; the next 1,024-token
  increment OOMed. These are not training limits.
- Ported the consensus task into verl and moved the implementation under
  `src/nl2sql_verl`. Source provenance is retained under its `consensus/` directory.
- Established that the active consensus recipe uses GRPO group centering,
  batch-wide reward standard deviation and token-level policy ratios despite
  GSPO-related names. Sequence-level sampling correction is a different option.
- Retained three active tools: `bm25_search_sqlite`, `sqlite_peek`, `sqlite_query`.
  The optional `consensus_at_1` tool was not exposed by the active source trainer.
- Ported seven rewards with weights: format 0.2, execution 0.5, result 2.0,
  table linking 0.5, column linking 0.5, nonnull 0.1, length penalty 0.1.
  BIRD result matching retains the original raw-row set comparison semantics.
- Downloaded/extracted all 69 train and 11 dev SQLite databases, approximately
  33.90 GB, with archive CRC validation and read-only opening checks. Downloaded
  official train questions contain 9,428 rows; that is not the selected RL corpus.
  The official dev ZIP annotation date differs from consensus's corrected dev
  annotations, so annotation provenance must be explicit.
- Commits: `e95b4cf`, `43770a6`.
- Evidence: [recipe/task documentation](src/nl2sql_verl/README.md),
  [download manifest](benchmark_results/bird_data_download/manifest.json),
  `benchmark_results/tp2_capacity/`, `benchmark_results/verl_recipe/`.

## 2026-10-26 — B200 environment and integration investigation

- Continued in the eight-B200 environment. The later pinned stack is PyTorch
  2.10.0+cu130, Transformers 5.18.0, vLLM 0.19.1+cu130, verl
  `d430157d15516d89e0cc853fd10fcd21d9234c44`, NeMo AutoModel
  `65866fbce209938cd153f2bc26f01fa0d34cc61e`.
- Developed the B200 task/runtime checks and isolated rollout/refit failures.
  The first complete rollout/update cycle produced corrupted outputs and zero
  learning signal; successful process execution was not counted as learning.
- Kept workspace handoff archives ignored (`2590742`). Detailed B200 outcomes
  below were finalized during the following UTC day; older notes may date these
  sessions using America/New_York.

## 2026-10-27 — B200 pilots and the validated CP1 control

### Before 05:13 — integration fixes, FFPA diagnosis and pilot results

- vLLM Gemma4 level-2 sleep/refit corrupted output despite all 1,189 exported
  tensors matching the source weights. Fixed `v_norm.weight` constants were
  ordinary tensors outside parameters/buffers. Level-1 sleep/refit reproduced
  baseline outputs/logprobs; the recipe uses this validated workaround.
- The parser now accepts complete native tool calls and consensus-style bare
  `call:` lines, while preserving sampled IDs and rejecting incomplete calls.
- FFPA standalone forward/backward works on B200. A specific D512 SM100 ring
  backward shortcut zeroed a one-local-key row even when other CP ranks
  contributed keys to the global softmax. A diagnostic correction restored
  the missing boundary term and passed tested short CP2 checks. It was not
  installed as the production backend and does not validate packed FFPA.
- Flex full-model BF16 CP comparisons also failed, independently of FFPA.
  Promoting attention alone did not fix those full-model gradient differences.
- Full FP32 actor controls passed the selected-gradient gates, including a
  TP4/CP2 versus TP2/CP1 check at 128 and 1,025 tokens. This did not validate BF16.
- Mixed wheel CUDA 13.0 and system CUDA 13.2 BLAS libraries caused worker failures.
  `worker_python.sh` restores wheel-library precedence after Ray shell startup.
- A repaired four-GPU pilot completed two updates in 1,666.5 seconds. An eight-row
  diagnostic dev subset changed from 2/8 to 7/8 correct. This small subset and
  two updates do not establish general learning improvement.
- An eight-GPU FP32 TP4/CP2 pilot initially used 16 dev examples, then resumed
  with 32. It was stopped at the user's request after completed step 6, retaining
  checkpoints 5 and 6. Recorded training time was 3,753.8 seconds, not the full
  two-hour target. This pilot still used the old 8,192-token filter and small
  rollout batches; it is distinct from the proposed production recipe.

| Completed step | Correct / 32 | Dev mean reward | Train mean reward |
|---:|---:|---:|---:|
| 1, resumed baseline | 17 | 2.25720 | — |
| 2 | 19 | 2.57573 | 3.51488 |
| 3 | 22 | 2.94086 | 3.13984 |
| 4 | 21 | 2.74262 | 3.64028 |
| 5 | 20 | 2.70843 | 3.14673 |
| 6 | 21 | 2.82248 | 3.14375 |

The series fluctuates; training batches differ between steps. Evidence:
`outputs/b200_mini/`, `outputs/ffpa_backward_investigation/`,
`outputs/b200_two_hour/`, `outputs/b200_two_hour_eval32/stopped_summary.json`.

### 05:00–05:13 — CP1 recipe validation completed

- Selected all 6,601 filtered-corpus examples; excluded the 9,428-row corpus.
  Removed dataset length filtering, including the previous 383 exclusions.
- Measured prompts are 4,402–10,786 tokens with the actual tool template.
  Set allowances to 11,264 prompt + 2,048 response = 13,312 total tokens.
- Selected a fixed 640-row dev subset across all eleven dev databases, containing
  the prior 32. Eval is greedy, one generation per prompt.
- Agreed production batch: 32 accepted prompt groups × 16 generations = 512
  trajectories, GRPO, native verl V1 DAPO result filtering, LR 1e-6,
  clipping 0.2/0.28, one PPO pass, no KL, save every five updates, eval every ten,
  retain the latest two completed full checkpoints. Default length: 100 updates.
- BF16 TP4 actor microbatch checks failed. TP8 is unsupported by the pinned
  adapter because global layers have four KV heads. TP2/DP4/CP1 passed;
  removing DP alone had not resolved the numerical issue.
- TP2/DP4/CP1 full-model microbatch checks: worst selected-gradient relative L2
  0.267% on BIRD and 0.381% on synthetic data. The independent HF BF16
  microbatch-consistency control was 0.288%; it was not a cross-implementation
  equality claim.
- A real 13,312-token Adam update peaked at 117.3 GiB allocated / 164.7 GiB
  reserved per GPU. Compute is BF16; master weights/Adam/reductions are FP32.
- The base-model smoke completed 32 accepted trajectories, native DAPO,
  tool rewards, one optimizer update, rollout refresh, a ~346-GiB full checkpoint
  and 640-row evaluation. Reward 3.11745, grad norm 3.00863, dev 422/640
  (65.9375%), exit 0, 847.5 seconds. One update is not a learning trend.
- CPU tests: 31; native DAPO tests: 12. Committed as `d31b350`.
- This is a validated CP1 control. The 100-update production job was not launched.
- Evidence: [CP1 validation summary](benchmark_results/verl_cp1/validation_summary.json),
  `outputs/cp1_recipe/`, [CP1 recipe](src/nl2sql_verl/configs/cp1.yaml).

### 05:23 onward — native verl FSDP2 and Ulysses CP investigation

The requested four stages are: native FSDP2 CP1 control; CP2 gradient checks;
warm attention/full-model memory scaling; real optimizer/save/restore and
GRPO/DAPO integration. The candidate actor topology is TP1 × CP2 × DP4 on eight
GPUs. Rollouts independently use TP4 × DP2 on the same allocation in alternating
phases. There are four CP pairs; each pair shares one sequence partition.

- Stock Gemma initialization failed because generic verl patching queried an
  ambiguous global KV-head attribute. Gemma has heterogeneous global/sliding
  head counts. A repository-local compatibility guard bypasses that unsuitable
  patch only for supported Gemma paths and validates each layer's divisibility.
- The attention adapter uses upstream HF Flex attention/mask construction plus
  verl's differentiable Ulysses collectives. It does not use NeMo TP projections
  or ring-softmax backward. Installed libraries are not edited.
- Masks preserve causal, sliding-window, document and suffix-padding semantics.
  CP slices inputs before embedding; next-token labels are shifted globally.
- Gradient probes execute the full 31B model, comparing 20 selected parameter
  tensors across layers 0, 1, 5 and 59. They are not all-parameter gradient audits.
  Short fixtures are SQL-context tails; full BIRD fixtures contain 5,088 and
  5,252 tokens. Gates remain relative L2 <3%, cosine >0.999, loss delta <0.01,
  maximum response-token logprob delta <0.1.

| UTC completion | Experiment | Result / limitation |
|---|---|---|
| 05:38 | Native FSDP2 CP1, BF16 SDPA math versus independent HF | Exact loss/logprobs; worst selected gradient 0.181%; pass |
| 05:41 | Initial custom Flex CP1 versus HF SDPA math | 89.35% gradient difference, max logprob 3.96; fail |
| 05:47 | FP32-attention-only variant versus HF SDPA math | 295.17% gradient difference; fail |
| 05:49–05:52 | Captured pretrained Q/K/V and upstream HF Flex control | Standalone attention is close to FP32 math; upstream HF Flex also differs from HF SDPA math in full-model BF16. This is not proof that either full-model gradient is correct |
| 05:57 | HF wrapper and HF masks, native FSDP2 CP1 versus HF Flex | Exact loss/logprobs; worst gradient 0.205%; pass |
| 06:00 | Same path, CP2 | Exact loss/logprobs; worst gradient 3.63%; fail |
| 06:08 | Matched 256-token padding buckets, CP2 | Exact loss/logprobs; worst gradient 4.49%; fail |
| 06:12 | Matched FP32 attention on both sides, otherwise BF16 | Exact loss/logprobs; worst gradient 3.23%; fail |
| 06:16 | FP32-weight/residual + BF16-autocast reference | Required Q/K/V dtype alignment: RoPE made Q/K FP32 while V remained BF16 |
| 06:18 | That AMP path, CP2, short fixtures | Exact loss/logprobs; worst gradient 1.35%; pass only on these short fixtures |
| 06:23 | AMP CP1, complete BIRD examples | Exact loss/logprobs; worst gradient ~0.000007%; pass |
| 06:26 | AMP CP2, two packed complete examples | Worst gradient 29.03%, max logprob 0.375; fail |
| 06:30 | AMP CP2, complete examples processed separately | Exact loss/logprobs but worst gradient 8.60%; fail; packing is not the only issue |
| 06:33 | FP32 matmul outputs with lower-precision Tensor Core arithmetic, BF16 Flex | Independent HF reference completed; CP2 comparison pending at this entry |

Additional findings and implementation corrections:

- Early harness attempts accidentally cleared accumulated gradients on context
  exit and mismatched checkpoint-wrapper parameter names. Those attempts are
  invalid evidence; corrected comparisons above preserve accumulation and
  normalize names. Gates were not loosened.
- Identical Q/K/V with the same HF mask give identical wrapper output. Equivalent
  mask representations can still change a BF16 output by one ULP. Full-model
  amplification remains a numerical investigation, not a proven complete cause.
- Captured real-Q/K/V CP2 backward checks passed the 3% operator gate (worst BF16
  1.247%, FP32-attention 0.167%) on tested odd-length padded fixtures. Operator
  correctness alone did not predict full-model parity.
- A single-GPU FP32-weight reference initially OOMed because autocast retained
  BF16 weight casts. Disabling autocast caching and releasing parameter gradients
  after accumulation allowed the independent full-model reference to fit.
- The FP32-attention-only 32K memory sweep was stopped after short cases because
  that candidate failed the numerical gate and was substantially slower.
- PyTorch `float32_matmul_precision='medium'` was profiled on this B200 stack:
  the observed forward/backward GEMM kernels were **TF32 Tensor Core kernels**
  with FP32 outputs. Do not label this diagnostic BF16 linear compute.
- CPU suite reached 33 passing tests. The native checkpoint-completion adapter
  requires every rank's model, optimizer and extra-state files plus trainer data
  before publishing/pruning; committed as `eddd66f` at 06:22.

The BF16 Flex/Ulysses attention sweep, including mask-construction peaks, passed
with finite forward/backward gradients. Geometry is 32 Q heads, 4 KV heads,
D512; these are attention-only peaks, not full-model/Adam residency:

| Global tokens | CP1 peak MiB/GPU | CP2 peak MiB/GPU |
|---:|---:|---:|
| 4,096 | 1,696.57 | 928.32 |
| 8,192 | 3,393.19 | 1,856.69 |
| 16,384 | 6,786.63 | 3,713.63 |
| 32,768 | 13,574.26 | 7,428.26 |

This supports practical near-linear memory over 4K–32K. Compact mask metadata
still has a quadratic component; dense attention compute remains quadratic.
Evidence: `outputs/fsdp_cp_recipe/`; compact tracked observations are under
`benchmark_results/verl_fsdp_cp/` as results are recorded.

### 06:35 onward — SSH publishing restored

- HTTPS pushes failed because this environment lacked Git HTTPS credentials;
  the connected GitHub integration had also rejected repository writes.
- Found an existing SSH key already authorized for `kpratik41`. Verified the
  server key against GitHub's published Ed25519 host key, configured this
  repository's SSH command, and changed `origin` to the SSH URL.
- Successfully pushed `2590742..eddd66f` to `origin/verl`. Local commits were no
  longer stranded. New reviewed work continues to be committed and pushed.
- Generated a separate Ed25519 pair as requested. Public key:
  `/home/ubuntu/.ssh/id_ed25519_nl2sql_github.pub`; private counterpart stays
  outside Git. The new key is not yet registered on GitHub; current pushes use
  the already-authorized `/home/ubuntu/.ssh/id_ed25519_github`.

### Pending at this entry

1. Finish the TF32/FP32-output CP2 comparison; repeat CP1 and packing/accumulation
   controls for any precision policy selected for production.
2. Measure warm full-model forward/backward/Adam memory at 4K, 8K, 16K, 32K for
   CP1/CP2. Test the production 13,312-token boundary and proposed microbatch.
3. Execute a full native FSDP checkpoint round trip with exact checks of all
   model/Adam tensors, scheduler and CPU/CUDA RNG, after deliberate perturbation.
4. Run two actual GRPO/DAPO updates with tools, reward components, rollout weight
   refresh, complete checkpoints and 640-row baseline/final validation.
5. Finalize the native CP2 precision/microbatch recipe only after these gates.
   The new launcher is fail-closed; no native-FSDP sustained RL job has started.
6. Keep recording results and pending items here, committing reviewed code and
   compact evidence. Leave unrelated local TP-backward diagnostics and archive
   files out of these commits.

### 06:37 — FP32-output/TF32 packed comparison

- The CP2 two-example packed test completed. Maximum response logprob delta was
  0.0851 and loss delta 0.00328, but the worst selected gradient difference was
  100.17% with poor early-layer cosine similarity. It fails the gradient gate.
- The same precision policy is being tested with examples processed separately
  to isolate packing. No production readiness has been declared.
- A repeated SSH push succeeded (`Everything up-to-date`); the chronological log
  and compact experiment evidence are now being committed alongside the gated
  experimental FSDP/Ulysses code. Optimizer/memory/restore/RL gates remain pending.
