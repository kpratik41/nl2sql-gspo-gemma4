# Main objective: Gemma 4 31B CP=2 + FFPA (or validated efficient attention)
# for long-context full-parameter NL2SQL RL in verl. CP reduces per-rank
# sequence pressure; efficient attention avoids quadratic score storage.
# Gates: forward memory/speed -> numerical/semantic parity -> real CP2
# -> backward/optimizer/resume -> verl RL -> NL2SQL. FFPA is optional.
# This file: Preserve measured evidence and correctness limits for the CP2/efficient-attention roadmap.
# See GEMMA4_LONG_CONTEXT_RL.md for the objective and complete roadmap.
import json
from pathlib import Path
j=json.loads(Path('benchmark_results/layer_replay/results.json').read_text())
rows=j['replays'];sem=j['semantics']
assert len(rows)==66 and len(sem)==72
for length in [128,1024,1025]:
 for layer in [0,5,6,59]:
  assert len({r['input_sha256'] for r in rows if r['length']==length and r['layer']==layer})==1
lines=['# Fixed-input Gemma decoder replay','', 'Date: 2026-10-22 (America/New_York).','',
'## Outcome','',
f'Completed {len(rows)} replay cases and {len(sem)} semantic checks. Semantic failures: {sum(not s["pass"] for s in sem)}. FP32 eager and FP32 SDPA math matched exactly in recorded outputs. BF16 differences remain small locally in these examples; that does not resolve the previously observed full-model logprob discrepancy.', '',
'## Method and reference limits','',
'- One cached BF16 Gemma 4 31B checkpoint; SQL fixture from the previous investigation; no cache, dropout, backward, or CP. Actual checkpoint/model scaling is retained.',
'- Capture complete hidden states, positional embeddings, and eager masks from the same full-model eager forward, at layers 0/6 (sliding) and 5/59 (global), at lengths 128/1024/1025.',
'- Clone each decoder block and replay identical captured inputs. Promote checkpoint BF16 weights and captured inputs to FP32 for the reference. This isolates arithmetic starting from the same quantized checkpoint and inputs; it is not a full-FP32 model forward or a higher-precision original checkpoint.',
'- Compare raw attention output before output projection, projected attention output, and complete decoder output across all positions/features. FP32 reference normalization and positional-embedding operations are recomputed within the layer using the captured positional tensors promoted to FP32.',
'- Reference: eager FP32. Independent control: SDPA forced to its math backend in FP32. Candidates: eager/SDPA/Flex BF16, plus forced CuTeDSL FFPA on global layers.',
'- Replay Flex uses a block mask converted from the exact eager additive mask, with fixed 32x32 tiles and one stage. It bypasses the HF model mask builder intentionally to isolate arithmetic. Thus it does not validate native Flex mask construction or AutoModel FFPA registration.',
'- Every backend for each layer/length uses the same input hash. No BF16 pass threshold was specified for this run; the errors below are descriptive, not a retrospective acceptance criterion.', '',
'## Worst relative L2 error across tested layers and lengths','', '| Backend | Raw attention | Projected attention | Complete decoder |','|---|---:|---:|---:|']
for backend in ['eager_fp32','sdpa_math_fp32','eager_bf16','sdpa_bf16','flex_bf16','ffpa_bf16']:
 rr=[r for r in rows if r['backend']==backend]
 lines.append('| '+backend+' | '+' | '.join(f"{max(r['errors'][stage]['relative_l2'] for r in rr):.8f}" for stage in ['attention','projected','decoder'])+' |')
lines += ['', '## Complete-decoder error by case','', '| Tokens | Layer | Backend | Relative L2 | Max absolute |','|---:|---:|---|---:|---:|']
for r in rows:
 e=r['errors']['decoder'];lines.append(f"| {r['length']} | {r['layer']} | {r['backend']} | {e['relative_l2']:.8f} | {e['max_abs']:.8f} |")
lines += ['', '## Semantic invariants','',
'- Decoder causality: add 1 to every hidden-state feature in the second half of the sequence; require the first-half decoder output to remain unchanged within 1e-5 absolute. Run for every replay case (66 checks). This checks each replayed block, not the complete model.',
'- Sliding-window exclusion: at 1025 positions, perturb key/value position 0 while examining the last query at 1024, whose 1024-token window excludes that key. Check eager, SDPA, and Flex on layers 0/6 (6 checks), with 1e-5 absolute tolerance.',
f'- Observed maximum semantic difference: {max(s["max_abs"] for s in sem):.8f}. All {len(sem)} checks passed.', '',
'## Interpretation and remaining gates','',
'Controlled local replay found no semantic violation in these cases and no large per-block error unique to an optimized backend. This supports accumulated finite-precision drift as a plausible contributor to full-model disagreement. It does not prove that this is the only cause, nor that observed logprob discrepancies are acceptable for RL.', '',
'Next compare native HF Flex masks against eager masks on identical Q/K/V, including padding and window boundaries. The replay deliberately shared masks, so model-specific mask construction remains an untested source of differences. Then measure accumulation over increasing consecutive layer counts on the same input, with selective FP32 normalization/attention controls. Reuse held-out natural text/SQL documents for probability comparisons after local controls pass; define logprob/KL acceptance bounds before that evaluation. Backward and real CP2 remain separate gates.', '',
'## Reproduction','', '```bash', '.venv-bench/bin/python benchmarks/layer_replay.py', '.venv-bench/bin/python benchmarks/summarize_layer_replay.py', '```','',
'Raw results and hashes: [benchmark_results/layer_replay/results.json](benchmark_results/layer_replay/results.json). Final execution log: [layer_replay_final.log](benchmark_results/layer_replay_final.log). Earlier logs retain mask-conversion harness failures; only the complete final run supplies the results above.', '',
'Prior investigation: [GEMMA4_MODEL_PARITY.md](GEMMA4_MODEL_PARITY.md).','']
# Keep the objective at the top when regenerating the report.
lines[1:1] = ['', '## Main objective and why we are doing this', '', 'Enable **CP=2 context parallelism** and **FFPA, or another validated memory-efficient attention backend**, for **`google/gemma-4-31B-it`**, with **verl** as the target framework for eventual **full-parameter NL2SQL RL**. CP=2 distributes one long sequence across two GPU ranks to reduce per-rank sequence/activation pressure. Efficient attention avoids materializing quadratic attention-score tensors; FFPA is a candidate for faster eligible global-attention operations. We want measured capacity and speed improvements while preserving model semantics, token probabilities, and gradients. Global-attention compute/latency is not expected to become linear merely because working memory is approximately linear.', '', 'We currently have one GPU, so we validate local behavior first and design the distributed recipe for a future two-node allocation with an explicit GPUs-per-node setting. CP=2 counts GPU ranks, not nodes. Standalone kernels or successful forward execution do not establish distributed training correctness. Existing TRL/ZeRO-3 code on `consensus` is prior task work; NL2SQL integration comes after the systems checks.', '', '**Incremental validation sequence:**', '', '1. Establish a reproducible environment and model snapshot; benchmark single-GPU attention and full-model forward latency/memory across context lengths, with and without FFPA. Measure mask construction and cached generation separately.', '2. Resolve numerical discrepancies with identical inputs/weights, teacher-forced token probabilities, FP32 references, fixed-input layer replay, and causal/window/padding/document-boundary checks. This is why we are investigating precision and masks before scaling further.', '3. Validate real CP=2 forward correctness, per-rank memory, communication, and speed on separate GPUs; compare efficient-attention backends under the same allocation and effective batch.', '4. Validate attention backward, model gradients, activation checkpointing, optimizer steps/state memory, and save/resume; increase context incrementally toward the provisional 40,960-total-token target.', '5. Integrate the validated AutoModel/Gemma path into verl, including CP token mapping, loss normalization, actor/reference probabilities, rollout weight synchronization, and one complete resumable RL update.', '6. Add NL2SQL data, executable rewards, and held-out database evaluation; measure complete RL iteration performance before choosing a production recipe.', '', '**Success means correct long-context training with CP=2 and an efficient attention path. FFPA is optional:** retain a correct FlexAttention or other efficient baseline if FFPA is unsupported or offers an unfavorable speed/memory tradeoff. Record performance observations separately from correctness acceptance. Detailed roadmap: [GEMMA4_LONG_CONTEXT_RL.md](GEMMA4_LONG_CONTEXT_RL.md).']
Path('GEMMA4_LAYER_REPLAY_RESULTS.md').write_text('\n'.join(lines))
