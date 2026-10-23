"""All-token forward capacity: ascending sweep, then OOM bisection to 1024 tokens.

Each bisection probe has fresh ranks; unsupported-kernel/timeout failures are
reported separately and are never labeled memory limits.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path('benchmark_results/tp2_capacity')
maximum = 262144
granularity = 1024
coarse = [8192, 16384, 32768, 40960, 65536, 98304, 131072, 196608, maximum]
summary = {}


def run(cp, lengths, label):
    directory = root/f'tp2_cp{cp}'
    for length in lengths:
        for path in directory.glob(f'length{length}_rank*.json'):
            path.unlink()
    log = root/f'tp2_cp{cp}_{label}.log'
    command = ['.venv-bench/bin/torchrun', '--standalone', '--nnodes=1',
               f'--nproc-per-node={2*cp}', 'benchmarks/tp_cp_forward.py',
               '--cp', str(cp), '--lengths', *map(str, lengths)]
    print(f'START cp={cp} lengths={lengths} log={log}', flush=True)
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '0,1' if cp == 1 else '0,1,2,3'}
    with log.open('w') as stream:
        try:
            result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=2400)
        except subprocess.TimeoutExpired:
            return [], {'status': 'timeout', 'log': str(log)}
    successful = []
    for length in lengths:
        files = [directory/f'length{length}_rank{rank}.json' for rank in range(2*cp)]
        if all(p.exists() for p in files):
            successful.append(length)
            print(f'PASS cp={cp} tokens={length} max_peak_GiB={max(json.loads(p.read_text())["peak_allocated_GiB"] for p in files):.3f}', flush=True)
        else:
            failures = [json.loads(p.read_text()) for p in directory.glob(f'length{length}_rank*_failure.json')]
            status = 'oom' if any(r['status'] == 'oom' for r in failures) else 'error'
            return successful, {'status': status, 'tokens': length, 'ranks': failures, 'log': str(log), 'returncode': result.returncode}
    if result.returncode:
        return successful, {'status': 'error_after_forward', 'returncode': result.returncode, 'log': str(log)}
    return successful, None


for cp in (1, 2):
    successes, failure = run(cp, [128]+coarse, 'coarse')
    best = max(successes, default=0)
    entry = {'tp': 2, 'cp': cp, 'batch_size': 1, 'gradients': False,
             'kv_cache': False, 'all_token_logits': True, 'vocabulary_sharded': True,
             'context_limit': maximum, 'largest_success_tokens': best, 'failure': failure}
    if failure and failure['status'] == 'oom':
        upper = failure['tokens']
        while upper-best > granularity:
            midpoint = ((best+upper)//2)//granularity*granularity
            if midpoint <= best:
                break
            fits, probe_failure = run(cp, [128, midpoint], f'probe_{midpoint}')
            if midpoint in fits:
                best = midpoint
            elif probe_failure and probe_failure['status'] == 'oom':
                upper = midpoint
            else:
                entry['probe_error'] = probe_failure
                break
            entry.update(largest_success_tokens=best, first_failed_upper_tokens=upper)
            summary[f'tp2_cp{cp}'] = entry
            (root/'capacity_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        entry.update(largest_success_tokens=best, first_failed_upper_tokens=upper, granularity_tokens=granularity)
    elif not failure and best == maximum:
        entry['limit_type'] = 'configured model context reached; memory exhaustion not established'
    summary[f'tp2_cp{cp}'] = entry
    (root/'capacity_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('RESULT', json.dumps(entry), flush=True)
