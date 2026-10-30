"""Launch/monitor the detached twelve-shard evaluation; never starts training."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shlex
import subprocess
import time

from .evaluate_full_dev import write_json

REPO = Path(__file__).resolve().parents[2]


def remote(node, command):
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', node,
                             shlex.join(command)], capture_output=True, text=True, timeout=90)
    if result.returncode:
        raise RuntimeError(f'{node}: {result.stderr.strip()}')
    return result.stdout


def launch(root, assignments):
    for item in assignments:
        shard, node = item['shard'], item['node']
        unit = f'nl2sql-dev-eval-s{shard:02d}'
        remote(node, ['mkdir', '-p', str(root)])
        active = remote(node, ['systemctl', '--user', 'show', unit, '--property=ActiveState', '--value']).strip()
        if active in ('active', 'activating'):
            print(f'{node} {unit}: already active', flush=True)
            continue
        command = ['systemd-run', '--user', '--unit=' + unit, '--collect',
                   '--property=KillMode=control-group', '--property=TimeoutStopSec=20',
                   '--property=WorkingDirectory=' + str(REPO),
                   '--property=StandardOutput=append:' + str(root / f'shard-{shard:02d}.log'),
                   '--property=StandardError=append:' + str(root / f'shard-{shard:02d}.log'),
                   '--setenv=CUDA_VISIBLE_DEVICES=' + item['gpus'],
                   '--setenv=VLLM_WORKER_MULTIPROC_METHOD=spawn', '--setenv=HF_HUB_OFFLINE=1',
                   '--setenv=OMP_NUM_THREADS=2', str(REPO / 'src/nl2sql_verl/worker_python.sh'),
                   '-m', 'src.nl2sql_verl.evaluate_full_dev', '--output-dir', str(root / f'shard-{shard:02d}'),
                   '--shard', str(shard)]
        print(remote(node, command).strip() or f'{node} {unit}: launched', flush=True)


def snapshot(root, assignments):
    script = '''import json,pathlib,subprocess,sys
root=pathlib.Path(sys.argv[1]); result=[]
for shard in json.loads(sys.argv[2]):
 p=root/f'shard-{shard:02d}'/'heartbeat.json'
 row=json.loads(p.read_text()) if p.exists() else {'shard':shard,'state':'starting','completed':0}
 unit=f'nl2sql-dev-eval-s{shard:02d}'
 q=subprocess.run(['systemctl','--user','show',unit,'--property=ActiveState','--value'],capture_output=True,text=True)
 row['service']=q.stdout.strip()
 if row['state']!='completed' and row['service'] not in ('active','activating'):row['state']='failed'
 result.append(row)
print(json.dumps(result))
'''
    def one(node):
        indices = [a['shard'] for a in assignments if a['node'] == node]
        rows = json.loads(remote(node, ['python3', '-c', script, str(root), json.dumps(indices)]))
        for row in rows:
            row['node'] = node
        return rows
    with ThreadPoolExecutor(3) as executor:
        rows = [r for group in executor.map(one, sorted({a['node'] for a in assignments})) for r in group]
    total = sum(r['completed'] for r in rows)
    counts = {k: sum(r.get('summary', {}).get(k, 0) for r in rows)
              for k in ['format_pass', 'execution_pass', 'result_pass', 'actual_tool_executions',
                        'tool_errors', 'budget_caps', 'turn_limit_reached']}
    status = {'state': 'completed' if all(r['state'] == 'completed' for r in rows) else
              ('failed' if any(r['state'] == 'failed' for r in rows) else 'running'),
              'updated_unix': time.time(), 'completed': total, 'target': 1534,
              'counts': counts, 'shards': sorted(rows, key=lambda r: r['shard'])}
    final = root / 'bird_official/summary.json'
    if final.exists():
        report = json.loads(final.read_text())
        status['bird_ex'] = {k: report[k] for k in ['correct', 'total', 'accuracy_percent', 'timeout_seconds']}
    write_json(root / 'status.json', status)
    return status


def collect(root, row):
    shard = row['shard']
    if row['node'] != 'h200-node0':
        subprocess.run(['rsync', '-a', f"{row['node']}:{root}/shard-{shard:02d}/",
                        str(root / f'shard-{shard:02d}') + '/'], check=True, timeout=90)
        subprocess.run(['rsync', '-a', f"{row['node']}:{root}/shard-{shard:02d}.log",
                        str(root / f'shard-{shard:02d}.log')], check=True, timeout=90)


def main(args):
    root = Path(args.output_dir).resolve()
    assignments = json.loads((root / 'assignments.json').read_text())
    if args.action == 'launch':
        launch(root, assignments)
        return
    collected = set()
    while True:
        status = snapshot(root, assignments)
        print(json.dumps({k: status[k] for k in ['state', 'completed', 'target', 'counts', 'bird_ex'] if k in status}), flush=True)
        if args.action == 'status':
            for row in status['shards']:
                print(f"{row['node']} shard {row['shard']:02d}: {row['state']} {row['completed']}/{row.get('target', '?')}")
            return
        for row in status['shards']:
            if row['state'] == 'completed' and row['shard'] not in collected:
                collect(root, row)
                collected.add(row['shard'])
        if status['state'] == 'completed':
            subprocess.run([str(REPO / 'src/nl2sql_verl/worker_python.sh'), '-m',
                            'src.nl2sql_verl.evaluate_full_dev', '--merge', '--output-dir', str(root)],
                           cwd=REPO, check=True)
            return
        if status['state'] == 'failed':
            raise RuntimeError('An eval shard failed; see status.json and its shard log. Completed results are retained.')
        time.sleep(30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['launch', 'status', 'watch'])
    parser.add_argument('--output-dir', required=True)
    main(parser.parse_args())
