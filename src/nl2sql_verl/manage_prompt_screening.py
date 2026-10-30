"""Detached supervision and collection of 12 TP2 screening shards; no training."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import time
import uuid

from .manage_full_dev_eval import REPO, remote, collect


def write_json(path, value):
    # Status queries and the persistent monitor can run simultaneously.
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def launch(root, assignments, only_shard=None):
    for item in assignments:
        i, node = item['shard'], item['node']
        if only_shard is not None and i != only_shard:
            continue
        unit = f'nl2sql-screen-s{i:02d}'
        remote(node, ['mkdir', '-p', str(root)])
        state = remote(node, ['systemctl', '--user', 'show', unit, '--property=ActiveState', '--value']).strip()
        if state in ('active', 'activating'):
            print(f'{node} {unit}: already running', flush=True)
            continue
        command = ['systemd-run', '--user', '--unit=' + unit, '--collect',
            '--property=KillMode=control-group', '--property=TimeoutStopSec=20',
            '--property=WorkingDirectory=' + str(REPO),
            '--property=StandardOutput=append:' + str(root / f'shard-{i:02d}.log'),
            '--property=StandardError=append:' + str(root / f'shard-{i:02d}.log'),
            '--setenv=CUDA_VISIBLE_DEVICES=' + item['gpus'], '--setenv=VLLM_WORKER_MULTIPROC_METHOD=spawn',
            '--setenv=HF_HUB_OFFLINE=1', '--setenv=OMP_NUM_THREADS=2', '--setenv=TOKENIZERS_PARALLELISM=false',
            str(REPO / 'src/nl2sql_verl/worker_python.sh'), '-m', 'src.nl2sql_verl.screen_training_prompts',
            '--output-dir', str(root / f'shard-{i:02d}'), '--shard', str(i)]
        print(remote(node, command).strip() or f'{node} {unit}: launched', flush=True)


def snapshot(root, assignments):
    script = '''import json,pathlib,subprocess,sys
root=pathlib.Path(sys.argv[1]);result=[]
for i in json.loads(sys.argv[2]):
 p=root/f'shard-{i:02d}'/'heartbeat.json'
 r=json.loads(p.read_text()) if p.exists() else {'state':'starting','shard':i,'completed':0,'groups_completed':0}
 q=subprocess.run(['systemctl','--user','show',f'nl2sql-screen-s{i:02d}','--property=ActiveState','--value'],capture_output=True,text=True)
 r['service']=q.stdout.strip()
 if r['state']!='completed' and r['service'] not in ('active','activating'):r['state']='failed'
 result.append(r)
print(json.dumps(result))
'''
    def one(node):
        rows = json.loads(remote(node, ['python3', '-c', script, str(root),
                        json.dumps([a['shard'] for a in assignments if a['node'] == node])]))
        return [r | {'node': node} for r in rows]
    with ThreadPoolExecutor(3) as pool:
        rows = [r for group in pool.map(one, sorted({a['node'] for a in assignments})) for r in group]
    histogram = {str(n): sum(r.get('solve_count_histogram', {}).get(str(n), 0) for r in rows) for n in range(17)}
    finished_groups = sum(histogram.values())
    state = 'completed' if all(r['state'] == 'completed' for r in rows) else (
            'failed' if any(r['state'] == 'failed' for r in rows) else 'running')
    completed = sum(r['completed'] for r in rows)
    status = {'state': state, 'updated_unix': time.time(), 'completed_rollouts': completed,
        'target_rollouts': 6601 * 16, 'completed_prompts': finished_groups, 'target_prompts': 6601,
        'solve_count_histogram': histogram, 'heterogeneous_completed': sum(histogram[str(i)] for i in range(1, 16)),
        'all_incorrect_completed': histogram['0'], 'all_correct_completed': histogram['16'],
        'pass_at_16_completed_percent': 100 * (finished_groups - histogram['0']) / finished_groups if finished_groups else None,
        'correct_rollouts': sum(r.get('counts', {}).get('result_pass', 0) for r in rows),
        'budget_caps': sum(r.get('counts', {}).get('response_budget_reached', 0) for r in rows),
        'tool_errors': sum(r.get('counts', {}).get('tool_error_count', 0) for r in rows),
        'shards': sorted(rows, key=lambda r: r['shard'])}
    write_json(root / 'status.json', status)
    return status


def main(args):
    root = Path(args.output_dir).resolve()
    assignments = json.loads((root / 'assignments.json').read_text())
    if args.action == 'launch':
        launch(root, assignments, args.only_shard)
        return
    collected = set()
    last_backup = 0.
    while True:
        try:
            status = snapshot(root, assignments)
        except (RuntimeError, subprocess.TimeoutExpired) as error:
            if args.action != 'watch':
                raise
            print(f'Transient monitor error: {error}', flush=True)
            time.sleep(30)
            continue
        compact = {k: v for k, v in status.items() if k not in ['shards', 'solve_count_histogram']}
        print(json.dumps(compact), flush=True)
        if args.action == 'status':
            for row in status['shards']:
                print(f"{row['node']} shard{row['shard']:02d}: {row['state']} {row['completed']}/{row.get('target', '?')}")
            if (root / 'selection.json').exists():
                print('Selection completed: ' + str(root / 'selected_train.parquet'))
            return
        for row in status['shards']:
            if row['state'] == 'completed' and row['shard'] not in collected:
                try:
                    collect(root, row)
                    collected.add(row['shard'])
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                    print(f'Will retry collection: {error}', flush=True)
        # Back up in-progress worker-node append-only results every five minutes.
        if time.monotonic() - last_backup >= 300:
            for row in status['shards']:
                if row['node'] != 'h200-node0' and row['state'] == 'running':
                    target = root / f"shard-{row['shard']:02d}"
                    target.mkdir(exist_ok=True)
                    try:
                        subprocess.run(['rsync', '-a', '--append-verify', '--exclude=*.tmp',
                            f"{row['node']}:{target}/", str(target) + '/'], check=True, timeout=120)
                    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                        print(f'Will retry incremental backup: {error}', flush=True)
            last_backup = time.monotonic()
        if status['state'] == 'completed' and len(collected) == 12:
            subprocess.run([str(REPO / '.venv-verl/bin/python'), '-m', 'src.nl2sql_verl.screen_training_prompts',
                            '--select', '--output-dir', str(root)], cwd=REPO, check=True)
            print('Selected training parquet ready; training was NOT started.', flush=True)
            return
        if status['state'] == 'failed':
            raise RuntimeError('Screening shard failed; completed rollouts are preserved. See status.json and shard log.')
        time.sleep(30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['launch', 'watch', 'status'])
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--only-shard', type=int)
    args = parser.parse_args()
    main(args)
