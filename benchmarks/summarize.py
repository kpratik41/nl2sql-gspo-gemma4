#!/usr/bin/env python3
"""Regenerate the benchmark report from committed/raw JSON measurements."""
import json
from pathlib import Path
import platform
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('benchmark_results')
rows=[]
for directory in ['global_final','sliding_final','model_final','model_reference']:
    rows.extend(json.loads(p.read_text()) for p in sorted((root/directory).glob('*.json')))
ok=[r for r in rows if r['status']=='ok']
lines=['# Gemma 4 benchmark results', '', 'Updated: '+datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d %H:%M %Z')+'.', '',
'**Current result:** standalone attention checks pass through 40,960 tokens, with linear measured kernel working memory. Full-model Flex/SDPA numerical agreement has a blocker; its latency measurements are execution diagnostics. CP2, backward, and RL remain untested.', '', '## Scope', '',
'Executed on one RTX PRO 6000 Blackwell Server Edition (SM120), with BF16, batch size one, and CP=1. Attention inputs use Gemma-shaped head counts and dimensions; these are standalone kernels, not AutoModel or verl integration. Model tests load the real cached Gemma 4 31B checkpoint, use no KV cache, and request last-position logits. No backward, optimizer, distributed CP, or RL claim is made.', '',
'## Environment and method', '',
'- Python 3.13.12; PyTorch 2.10.0+cu130; Triton 3.6.0; Transformers 5.18.0; FFPA 0.2.4; CuTeDSL 4.6.2.',
'- Isolated `.venv-bench` overlays `/opt/pytorch/lib/python3.13/site-packages` through a `.pth` file; the existing project environment was left intact. This environment depends on that base installation.',
'- Model snapshot: `842da3794eaa0b77d5f08bae87a17459d91ff475`, loaded locally without downloading weights.',
'- Every backend/length runs in a fresh subprocess. Attention uses five warmups and twenty measured repetitions; model tests use two and five. Median/p95 wall latency includes synchronization. Separate CUDA-event medians are in JSON.',
'- Attention memory is incremental peak allocated memory above live Q/K/V and mask allocations. The mask remains outside the timed attention operation; mask construction time and total allocated/reserved peaks are saved separately. Model timing includes its mask path.',
'- Initial mask construction was outside the attention memory counters. The eager mask builder can allocate quadratic temporary tensors; the kernel-only linear result must not be read as proof of linear end-to-end memory. Separate mask measurements are recorded below when available.',
'- Each attention case checks all short-reference outputs against independent FP32 eager attention, then samples queries at the start, around the 1,024-token window boundary, and at the tail against all keys. Tolerances: absolute/relative 0.02. Inputs are synthetic random tensors with explicit scale `head_dim**-0.5`; this is not a model-level semantic parity test.',
'- Profiler runs are separate from timed/memory measurement. JSON contains actual CUDA kernel names, attention operators, allocator retries, cold latency, and numerical errors.',
'- GPU device usage/driver information and dependency snapshots are in `benchmark_results/environment/`. The overlay has unresolved optional-package requirements and inherited dependency conflicts listed in `pip-check.txt`; it is a measured kernel environment, not a validated AutoModel/verl training installation.', '',
'## Measured results', '', '| Scope | Backend | Tokens | Median ms | p95 ms | Incremental peak MiB | Total peak allocated GiB |', '|---|---|---:|---:|---:|---:|---:|']
for r in sorted(ok,key=lambda r:(r['kind'],r['layer'],r['length'],r['backend'])):
    scope=r['kind']+'/'+r['layer'] if r['kind']=='attention' else 'model'
    lines.append(f"| {scope} | {r['backend']} | {r['length']:,} | {r['median_wall_ms']:.3f} | {r['p95_wall_ms']:.3f} | {r['incremental_peak_bytes']/2**20:.2f} | {r['peak_allocated_bytes']/2**30:.3f} |")
lines += ['', '## Interpretation', '']
global_rows=[r for r in ok if r['kind']=='attention' and r['layer']=='global']
for backend in ['flex','ffpa']:
    rr=sorted([r for r in global_rows if r['backend']==backend],key=lambda r:r['length'])
    if len(rr)>=3:
        x=np.array([r['length'] for r in rr]); y=np.array([r['incremental_peak_bytes']/2**20 for r in rr])
        fit=np.polyfit(x,y,1);pred=np.polyval(fit,x)
        r2=1-np.sum((y-pred)**2)/np.sum((y-y.mean())**2)
        lines.append(f'- Global {backend}: measured incremental-memory fit `{fit[1]:.3f} + {fit[0]:.6f} × tokens` MiB, R²={r2:.6f}, over {x[0]:,}–{x[-1]:,} tokens. This supports linear working-memory growth over the measured range; it does not establish whole-model scaling or linear compute.')
flex={r['length']:r for r in global_rows if r['backend']=='flex'}
ffpa={r['length']:r for r in global_rows if r['backend']=='ffpa'}
for n in sorted(flex.keys()&ffpa.keys()):
    lines.append(f"- At {n:,} tokens, FFPA/Flex speedup is {flex[n]['median_wall_ms']/ffpa[n]['median_wall_ms']:.2f}× (greater than one means FFPA is faster).")
model_pairs={}
for r in ok:
    if r['kind']=='model' and r['length']>=1024: model_pairs.setdefault(r['length'],{})[r['backend']]=r
if model_pairs:
    import torch
    lines += ['', '### Full-model numerical comparison', '']
    for n,pair in sorted(model_pairs.items()):
        if 'flex' in pair and 'sdpa' in pair:
            assert pair['flex']['input_sha256']==pair['sdpa']['input_sha256'], 'Model input mismatch'
            a=torch.load(root/'model_final'/f'model_global_flex_{n}.json.logits.pt',weights_only=True).float()
            b=torch.load(root/'model_final'/f'model_global_sdpa_{n}.json.logits.pt',weights_only=True).float()
            lines.append(f'- {n:,} tokens, final-position logits: max absolute Flex/SDPA difference {(a-b).abs().max().item():.6f}, relative L2 {((a-b).norm()/b.norm()).item():.6f}. **Model-level numerical parity is blocked:** this difference is too large to accept as ordinary BF16 noise. Input SHA256 hashes match. Timings establish execution only; isolate layer outputs/masks and compare with eager before proceeding to longer context or training.')
reference_dir=root/'model_reference'
if all((reference_dir/f'model_global_{b}_128.json.logits.pt').exists() for b in ['eager','sdpa','flex']):
    import torch
    ref=torch.load(reference_dir/'model_global_eager_128.json.logits.pt',weights_only=True).float()
    lines += ['', '### Short eager-reference diagnostic', '']
    for backend in ['sdpa','flex']:
        x=torch.load(reference_dir/f'model_global_{backend}_128.json.logits.pt',weights_only=True).float()
        lines.append(f'- At 128 identical-input tokens, {backend} versus eager final-position logits: max absolute difference {(x-ref).abs().max().item():.6f}, relative L2 {((x-ref).norm()/ref.norm()).item():.6f}.')
lines += ['', '## Failures and implementation findings', '',
'- Default D512 Flex tile selection failed at compilation: requested 200,704 bytes of shared memory versus a 101,376-byte hardware limit. Explicit `BLOCK_M=32`, `BLOCK_N=32`, `num_stages=1` passed. The original failure is preserved in `benchmark_results/global/attention_global_flex_1024.json`; final comparisons use these fixed smaller tiles, not a fully autotuned Flex optimum.',
'- Exploratory results under `benchmark_results/global/` and `flex_tuning_1024.json` are excluded from final comparisons. A tuning probe briefly overlapped the exploratory sweep; final runs execute sequentially.',
'- FFPA was explicitly forced to its CuTeDSL backend with GQA enabled. Inspect the recorded kernel names to verify execution; the SM120 card uses the generic SM80-family FFPA kernel path in this package.',
'- Initial model results in `benchmark_results/model/` used the global RNG after model loading and showed large output differences. They are excluded; the controlled rerun confirmed the discrepancy persists with identical inputs. Final model runs use an independent generator seeded to 42 and record matching input SHA256 hashes before comparison.',
'- Full-model FFPA is deliberately not substituted with a generic HF monkeypatch: AutoModel registration and model dispatch still need their own validated environment.',
]
for r in rows:
    if r['status']!='ok': lines.append(f"- {r.get('kind','unknown')} / {r.get('backend','unknown')} / {r.get('length','?')}: **{r['status']}** — `{r.get('error','see raw log')}`")
masks=[json.loads(p.read_text()) for p in sorted((root/'masks').glob('*.json'))]
if masks:
    lines += ['', '## Separate causal-mask construction', '', '| Mode | Tokens | Cold construction s | Peak allocated MiB | Live mask MiB |', '|---|---:|---:|---:|---:|']
    for r in masks:
        if r['status']=='ok': lines.append(f"| {'compiled' if r['compiled'] else 'eager'} | {r['length']:,} | {r['cold_s']:.3f} | {r['peak_bytes']/2**20:.2f} | {r['live_mask_bytes']/2**20:.2f} |")
        else: lines.append(f"| {r.get('compiled')} | {r['length']} | {r['status']} | — | — |")
    lines += ['', 'At 40,960 tokens, eager construction peaked at 15.63 GiB versus 131.13 MiB with compiled construction. The final live mask was about 1.57 MiB. Compiled-mask peak time/memory still grows with shape and is not a proof of asymptotically linear mask storage. Cold compiled-mask time includes compilation. Training/packed masks need separate validation. These allocations are separate from the attention table.', '']
lines += ['', '## Reproduction', '', 'Run `bash benchmarks/setup_env.sh` to recreate the overlay on the same PyTorch base (or set `BENCH_BASE_PYTHON`). Exact overlay pins are in `benchmarks/requirements-overlay.txt`; inherited base packages are recorded in the environment snapshot. This preserves the measured dependency limitations rather than resolving a new stack. Then run from the repository root:', '', '```bash',
'.venv-bench/bin/python benchmarks/gemma_attention.py --output benchmark_results/global_final',
'.venv-bench/bin/python benchmarks/gemma_attention.py --layer sliding --backends flex --output benchmark_results/sliding_final',
'.venv-bench/bin/python benchmarks/gemma_attention.py --kind model --backends sdpa flex --lengths 1024 4096 --warmup 2 --repeats 5 --timeout 900 --output benchmark_results/model_final',
'.venv-bench/bin/python benchmarks/gemma_attention.py --kind model --backends eager sdpa flex --lengths 128 --warmup 1 --repeats 3 --output benchmark_results/model_reference',
'.venv-bench/bin/python benchmarks/mask_memory.py',
'.venv-bench/bin/python benchmarks/summarize.py', '```', '',
'Use `--help` for shorter sweeps and model-path overrides. Preserve original result directories when collecting a new run. The orchestrator stops increasing length for a backend after a failure/timeout; inspect JSON/log files, not only its console output.', '',
'## Next gates', '',
'1. Resolve a fully pinned AutoModel environment and reuse its exact dense/varlen FFPA entry points; verify model-level output agreement before claiming model FFPA speedups.',
'2. Validate model forward at longer lengths with controlled output allocations. Investigate any model-loader/kernel failures below before extending context.',
'3. Resolve model-level Flex/SDPA numerical disagreement with short eager-reference, per-layer hidden-state, mask, and attention-output comparisons. Keep model performance results diagnostic until this passes.',
'4. Add cached generation as a separate test, then attention/model backward checks with numerical references.',
'5. Run real CP=2 parity and communication tests when two GPUs are available. A single GPU cannot establish this result.',
'6. Integrate the validated path into verl; SQL rewards and task scripts follow the systems gates.', '',
'Follow-up same-weight teacher-forced and real-QKV FP32 investigation: [GEMMA4_MODEL_PARITY.md](GEMMA4_MODEL_PARITY.md). BF16 eager is a candidate reference, not exact ground truth.', '', 'Roadmap: [GEMMA4_LONG_CONTEXT_RL.md](GEMMA4_LONG_CONTEXT_RL.md). Raw measurements: [benchmark_results](benchmark_results/).', '']
if global_rows:
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for backend in ['flex','ffpa']:
        rr=sorted([r for r in global_rows if r['backend']==backend],key=lambda r:r['length'])
        if rr:
            x=[r['length'] for r in rr]
            axes[0].plot(x,[r['incremental_peak_bytes']/2**20 for r in rr],'o-',label=backend)
            axes[1].plot(x,[r['median_wall_ms'] for r in rr],'o-',label=backend)
    shared=sorted(flex.keys()&ffpa.keys())
    axes[2].plot(shared,[flex[n]['median_wall_ms']/ffpa[n]['median_wall_ms'] for n in shared],'o-')
    axes[2].axhline(1,color='gray',linestyle='--')
    for ax,label in zip(axes,['Incremental peak allocated MiB','Median forward wall ms','FFPA speedup over Flex']):
        ax.set_xlabel('Sequence tokens');ax.set_ylabel(label);ax.grid(alpha=.25)
    axes[0].legend();axes[1].legend();fig.suptitle('BF16 global attention: B=1, Hq=32, Hkv=4, D=512, CP=1')
    fig.tight_layout();fig.savefig(root/'global_attention.png',dpi=160);plt.close(fig)
    lines += ['![Measured global attention memory, latency, and speedup](benchmark_results/global_attention.png)', '']
Path('GEMMA4_BENCHMARK_RESULTS.md').write_text('\n'.join(lines))
print(f'Report: {len(ok)} successful cases, {len(rows)-len(ok)} failures')
