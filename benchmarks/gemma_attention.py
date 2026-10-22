#!/usr/bin/env python3
# Main objective: Gemma 4 31B CP=2 + FFPA (or validated efficient attention)
# for long-context full-parameter NL2SQL RL in verl. CP reduces per-rank
# sequence pressure; efficient attention avoids quadratic score storage.
# Gates: forward memory/speed -> numerical/semantic parity -> real CP2
# -> backward/optimizer/resume -> verl RL -> NL2SQL. FFPA is optional.
# This file: Measure single-GPU forward latency/memory and kernel correctness before real CP2 or backward tests.
# See GEMMA4_LONG_CONTEXT_RL.md for the objective and complete roadmap.
"""Fresh-process BF16 Gemma-shaped attention/model forward benchmarks.

Does not implement AutoModel dispatch or distributed CP. Results identify that scope.
"""
import argparse
import importlib.metadata as md
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker', action='store_true')
    p.add_argument('--kind', choices=['attention', 'model'], default='attention')
    p.add_argument('--backend', choices=['flex', 'ffpa', 'sdpa', 'eager'], default='flex')
    p.add_argument('--layer', choices=['global', 'sliding'], default='global')
    p.add_argument('--length', type=int, default=1024)
    p.add_argument('--lengths', type=int, nargs='+', default=[1024,2048,4096,8192,16384,32768,40960])
    p.add_argument('--backends', nargs='+', default=['flex','ffpa'])
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--repeats', type=int, default=20)
    p.add_argument('--timeout', type=int, default=600)
    p.add_argument('--output', default='benchmark_results/attention')
    p.add_argument('--model-path', default=os.path.expanduser('~/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475'))
    return p


def worker(a):
    import torch
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    torch.manual_seed(42)
    torch.set_float32_matmul_precision('highest')
    result = vars(a).copy()
    result.update(status='starting', cp=1, seed=42, dtype='bfloat16', torch=torch.__version__, cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability())
    result['versions'] = {}
    for name in ['transformers','ffpa-attn','triton','nvidia-cutlass-dsl','quack-kernels']:
        try: result['versions'][name] = md.version(name)
        except md.PackageNotFoundError: result['versions'][name] = None
    try:
        if a.kind == 'attention':
            if a.backend == 'eager': raise ValueError('eager is supported only for short model reference tests')
            d, hkv = (512,4) if a.layer == 'global' else (256,16)
            result.update(flex_kernel_options={'BLOCK_M':32,'BLOCK_N':32,'num_stages':1},head_dim=d, query_heads=32, kv_heads=hkv, scale=d**-0.5,
                          scope='synthetic attention kernel; not model dispatch', use_cache=False)
            def inputs(s):
                return [torch.randn(1,h,s,d,device='cuda',dtype=torch.bfloat16) for h in [32,hkv,hkv]]
            def mask(b,h,q,k):
                return (q >= k) & ((q-k < 1024) if a.layer == 'sliding' else True)
            if a.backend == 'ffpa':
                if a.layer != 'global': raise ValueError('FFPA comparison targets global D512 only')
                from ffpa_attn import ffpa_attn_func, CuTeDSLBackend
                def call(q,k,v,bm):
                    return ffpa_attn_func(q,k,v,is_causal=True,scale=d**-0.5,enable_gqa=True,backend=CuTeDSLBackend())
            elif a.backend == 'flex':
                compiled = torch.compile(flex_attention, dynamic=False)
                def call(q,k,v,bm):
                    return compiled(q,k,v,block_mask=bm,enable_gqa=True,scale=d**-0.5,kernel_options={'BLOCK_M':32,'BLOCK_N':32,'num_stages':1})
            else:
                def call(q,k,v,bm):
                    if a.layer == 'sliding': raise ValueError('SDPA diagnostic is global-only')
                    return torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True,enable_gqa=True,scale=d**-0.5)
            # Independent FP32 eager short reference; never allocate long SxS scores.
            sq,sk,sv = inputs(128)
            small_mask = create_block_mask(mask,1,1,128,128,device='cuda') if a.backend=='flex' else None
            with torch.inference_mode():
                actual = call(sq,sk,sv,small_mask)
                kk = sk.float().repeat_interleave(32//hkv,dim=1)
                vv = sv.float().repeat_interleave(32//hkv,dim=1)
                scores = (sq.float() @ kk.transpose(-1,-2)) * d**-0.5
                pos=torch.arange(128,device='cuda')
                scores.masked_fill_(~mask(0,0,pos[:,None],pos[None,:]),float('-inf'))
                ref=scores.softmax(-1) @ vv
                error=(actual.float()-ref)
                result['parity']={'max_abs':error.abs().max().item(),'relative_l2':(error.norm()/ref.norm()).item(), 'atol':0.02,'rtol':0.02}
                torch.testing.assert_close(actual.float(),ref,atol=0.02,rtol=0.02)
            del sq,sk,sv,actual,kk,vv,scores,pos,ref,error
            q,k,v=inputs(a.length)
            start=time.perf_counter()
            bm=create_block_mask(mask,1,1,a.length,a.length,device='cuda') if a.backend=='flex' else None
            torch.cuda.synchronize()
            result['mask_build_s']=time.perf_counter()-start
            operation=lambda: call(q,k,v,bm)
        else:
            if a.backend=='ffpa': raise ValueError('Full-model FFPA requires AutoModel registration; not implemented by this HF baseline')
            from transformers import AutoModelForImageTextToText
            start=time.perf_counter()
            model=AutoModelForImageTextToText.from_pretrained(a.model_path,local_files_only=True,dtype=torch.bfloat16,device_map='cuda',attn_implementation='flex_attention' if a.backend=='flex' else a.backend).eval()
            torch.cuda.synchronize()
            result['load_s']=time.perf_counter()-start
            result.update(scope='HF full-model no-cache forward, last-position logits',use_cache=False,logits_to_keep=1,model_revision=Path(a.model_path).name)
            ids=torch.randint(100,20000,(1,a.length),device='cuda',generator=torch.Generator(device='cuda').manual_seed(42))
            import hashlib
            result['input_sha256']=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()
            operation=lambda: model(input_ids=ids,use_cache=False,logits_to_keep=1,**({'kernel_options':{'BLOCK_M':32,'BLOCK_N':32,'num_stages':1}} if a.backend=='flex' else {})).logits
        with torch.inference_mode():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            start=time.perf_counter()
            out=operation()
            torch.cuda.synchronize()
            result['cold_s']=time.perf_counter()-start
            result['cold_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
            if a.kind == 'model':
                if not torch.isfinite(out).all(): raise RuntimeError('Nonfinite model logits')
                result['logits_top5']=out[0,-1].topk(5).indices.tolist()
                Path(a.output).parent.mkdir(parents=True,exist_ok=True)
                torch.save(out.detach().cpu(),a.output+'.logits.pt')
            if a.kind == 'attention':
                # Check sampled queries against all keys, including window and tail boundaries.
                positions=sorted(set([0,min(1023,a.length-1),min(1024,a.length-1),a.length-1]))
                qi=torch.tensor(positions,device='cuda')
                ki=torch.arange(a.length,device='cuda')
                refs=[]
                for h in range(32):
                    scores=(q[0,h,qi].float() @ k[0,h//(32//hkv)].float().T)*d**-0.5
                    scores.masked_fill_(~mask(0,0,qi[:,None],ki[None,:]),float('-inf'))
                    refs.append(scores.softmax(-1) @ v[0,h//(32//hkv)].float())
                ref=torch.stack(refs)
                actual=out[0,:,qi,:].float()
                result['sampled_full_length_parity']={'positions':positions,'max_abs':(actual-ref).abs().max().item(),'relative_l2':((actual-ref).norm()/ref.norm()).item()}
                torch.testing.assert_close(actual,ref,atol=0.02,rtol=0.02)
                del ref,actual,refs,scores,qi,ki
            del out
            for _ in range(a.warmup):
                out=operation()
                torch.cuda.synchronize()
                del out
            torch.cuda.synchronize()
            base=torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            times=[]
            events=[]
            retries=torch.cuda.memory_stats().get('num_alloc_retries',0)
            for _ in range(a.repeats):
                e0,e1=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                t=time.perf_counter()
                e0.record()
                out=operation()
                e1.record()
                torch.cuda.synchronize()
                times.append((time.perf_counter()-t)*1000)
                events.append(e0.elapsed_time(e1))
                del out
            result.update(baseline_allocated_bytes=base,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),incremental_peak_bytes=torch.cuda.max_memory_allocated()-base,
                          median_wall_ms=statistics.median(times),p95_wall_ms=sorted(times)[math.ceil(.95*len(times))-1],median_cuda_ms=statistics.median(events),wall_samples_ms=times,
                          tokens_per_s=a.length/(statistics.median(times)/1000),allocation_retries=torch.cuda.memory_stats().get('num_alloc_retries',0)-retries)
            # Profile a separate pass, after capturing timing/memory measurements.
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                out=operation()
                torch.cuda.synchronize()
                del out
            result['cuda_kernels']=sorted(set(e.name for e in prof.events() if e.device_type==torch.autograd.DeviceType.CUDA))
            result['attention_ops']=sorted(set(e.key for e in prof.key_averages() if any(x in e.key.lower() for x in ['attention','ffpa','softmax','bmm'])))
            result['status']='ok'
    except Exception as exc:
        result.update(status='oom' if isinstance(exc,torch.OutOfMemoryError) else 'error',error=repr(exc),traceback=traceback.format_exc())
    path=Path(a.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['kind','backend','layer','length','status','median_wall_ms','incremental_peak_bytes','error'] if k in result}),flush=True)
    return 0 if result['status']=='ok' else 1


def main():
    p=parser()
    a=p.parse_args()
    if a.repeats < 1 or a.warmup < 0 or a.length < 1 or any(n < 1 for n in a.lengths):
        p.error('repeats and lengths must be positive; warmup must be nonnegative')
    if a.worker: return worker(a)
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    failed=False
    for backend in a.backends:
        for length in a.lengths:
            target=root/f'{a.kind}_{a.layer}_{backend}_{length}.json'
            cmd=[sys.executable,__file__,'--worker','--kind',a.kind,'--layer',a.layer,'--backend',backend,'--length',str(length),'--warmup',str(a.warmup),'--repeats',str(a.repeats),'--model-path',a.model_path,'--output',str(target)]
            log=target.with_suffix('.log')
            with log.open('w') as f:
                try: r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=a.timeout)
                except subprocess.TimeoutExpired:
                    target.write_text(json.dumps({'status':'timeout','backend':backend,'length':length,'timeout_s':a.timeout})+'\n'); print(target,'timeout',flush=True);failed=True;break
            data=json.loads(target.read_text()) if target.exists() else {'status':'crash','returncode':r.returncode}
            if not target.exists(): target.write_text(json.dumps(data)+'\n')
            print(target,data['status'],data.get('median_wall_ms',data.get('error','')),flush=True)
            if data['status']!='ok':
                failed=True
                break
    return int(failed)

if __name__=='__main__': sys.exit(main())
