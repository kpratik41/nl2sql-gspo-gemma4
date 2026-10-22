#!/usr/bin/env python3
"""Measure mask construction separately from attention in fresh processes."""
import argparse,json,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--worker',action='store_true');p.add_argument('--length',type=int,default=1024)
p.add_argument('--compiled',action='store_true');p.add_argument('--output',default='benchmark_results/masks')
a=p.parse_args()
if not a.worker:
    Path(a.output).mkdir(parents=True,exist_ok=True)
    for mode in [False,True]:
        for n in [1024,4096,16384,40960]:
            path=Path(a.output)/f'mask_{"compiled" if mode else "eager"}_{n}.json'
            cmd=[sys.executable,__file__,'--worker','--length',str(n),'--output',str(path)]+(['--compiled'] if mode else [])
            with path.with_suffix('.log').open('w') as log:
                try: r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=180)
                except subprocess.TimeoutExpired: path.write_text(json.dumps({'status':'timeout','length':n,'compiled':mode}));continue
            print(path,r.returncode,flush=True)
else:
    import torch
    from torch.nn.attention.flex_attention import create_block_mask
    def mask(b,h,q,k): return q>=k
    result={'length':a.length,'compiled':a.compiled,'torch':torch.__version__}
    try:
        torch.cuda.init();torch.cuda.empty_cache();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter()
        bm=create_block_mask(mask,1,1,a.length,a.length,device='cuda',_compile=a.compiled)
        torch.cuda.synchronize()
        result.update(status='ok',cold_s=time.perf_counter()-start,baseline_bytes=base,peak_bytes=torch.cuda.max_memory_allocated(),live_mask_bytes=torch.cuda.memory_allocated()-base)
    except Exception as exc: result.update(status='error',error=repr(exc))
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
