"""Read-only terminal views for the supervised H200 run."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import subprocess
import time
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO/'outputs/h200_training'
ANSI = re.compile(r'\x1b\[[0-9;]*m')


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def tail(path, size=131072):
    try:
        with path.open('rb') as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell()-size))
            return stream.read().decode(errors='replace')
    except OSError:
        return ''


def current_run():
    runs = [ROOT/mode for mode in ('smoke', 'pilot', 'train') if (ROOT/mode/'heartbeat.json').exists()]
    return max(runs, key=lambda p:(p/'heartbeat.json').stat().st_mtime) if runs else None


def logs():
    identity, offset = None, 0
    while True:
        run = current_run()
        path = run/'training.log' if run else None
        if path and path.exists():
            key = (str(path), path.stat().st_ino)
            if identity != key:
                identity = key
                print(f'\n=== Following {path} ===', flush=True)
                previous = tail(path)
                print('\n'.join(previous.splitlines()[-35:]), flush=True)
                offset = path.stat().st_size
            with path.open('rb') as stream:
                stream.seek(offset if path.stat().st_size >= offset else 0)
                new = stream.read()
                offset = stream.tell()
            if new:
                print(new.decode(errors='replace'), end='', flush=True)
        time.sleep(1)


def status():
    print('H200 run monitor — '+datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d %H:%M:%S %Z'))
    pipeline = read_json(REPO/'outputs/h200_checks/pipeline_status.json')
    print('Pipeline:', pipeline.get('stage', 'unknown'))
    if pipeline.get('error'):
        print('Pipeline error:', pipeline['error'])
    run = current_run()
    if run is None:
        print('No run heartbeat yet.')
        return
    heartbeat, ended = read_json(run/'heartbeat.json'), read_json(run/'status.json')
    records = []
    for line in tail(run/'metrics.jsonl', 2*1024*1024).splitlines():
        try:
            records.append(json.loads(line))
        except ValueError:
            pass
    completed = max((r.get('step', 0) for r in records), default=0)
    if ended:
        phase = 'completed' if ended.get('exit_code') == 0 else 'FAILED'
    elif not records:
        phase = 'initialization / initial evaluation (see live log)'
        if 'trainer.val_before_train=false' in read_json(run/'command.json'):
            phase = 'initialization / first training cycle; base evaluation already completed in smoke'
    else:
        phase = 'training cycle: rollout generation, scoring/filtering, actor computation'
        backup = Path(f'/home/ubuntu/nl2sql-checkpoints/dapo-fsdp-cp2-16gpu-{run.name}')
        saved = max((int(p.parent.name.removeprefix('global_step_'))
                     for p in backup.glob('global_step_*/nl2sql_model_only_complete.json')), default=0)
        if saved > completed:
            phase = f'step-{saved} model saved; evaluation/metrics pending'
    print(f'Run: {run.name} | Phase: {phase}')
    print(f'Last logged optimizer update: {completed} | Microbatch: {heartbeat.get("microbatch")} | W&B: {heartbeat.get("wandb_mode")}')
    print(f'Heartbeat age: {time.time()-(run/"heartbeat.json").stat().st_mtime:.0f}s | Elapsed: {heartbeat.get("elapsed_seconds",0)/60:.1f} min')
    if ended:
        print('Exit:', ended)
    log = ANSI.sub('', tail(run/'training.log'))
    reports = re.findall(r'[^\n]*pending: \d+, running: \d+, finished: \d+, failure: \d+[^\n]*', log)
    if reports:
        print('Last sampler report (can precede actor work):\n '+reports[-1])
    val = next((r for r in records if r.get('step') == 0), {})
    accuracy = val.get('data', {}).get('val-core/nl2sql_bird/acc/mean@1')
    if accuracy is not None:
        print(f'Initial 640-example evaluation complete; accuracy: {accuracy:.2%}')
    else:
        reference = read_json(run/'initial_evaluation_reference.json')
        accuracy = reference.get('metrics', {}).get('val-core/nl2sql_bird/acc/mean@1')
        if accuracy is not None:
            print(f'Base evaluation reused from smoke: {reference["rows"]} examples; accuracy {accuracy:.2%}')
    urls = re.findall(r'https://\S+/runs/[A-Za-z0-9_-]+', log)
    if urls:
        print('W&B:', urls[-1])
    print('Log:',run/'training.log')
    print('\nTraining is owned by systemd; detaching/exiting this monitor does not stop it.')
    print('tmux windows: 0 status | 1 logs | 2 GPUs | 3 Ray | 4 supervisor')


def gpus():
    query = ['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu', '--format=csv,noheader']
    for label, command in [('node0 — 8 training GPUs',query),
                           ('node1 — 8 training GPUs',['ssh','-o','BatchMode=yes','-o','ConnectTimeout=3','h200-node1',*query])]:
        print(label+' (index, memory used/total, compute utilization)',flush=True)
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=10)
            print(result.stdout or result.stderr,flush=True)
        except subprocess.TimeoutExpired:
            print('GPU query timed out',flush=True)
    print('Node2: outside this 16-GPU job.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('view', choices=['status','logs','gpus','ray'])
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    if args.view == 'logs':
        return logs()
    while True:
        if args.watch:
            print('\033[2J\033[H',end='',flush=True)
        if args.view == 'status':
            status()
        elif args.view == 'gpus':
            gpus()
        else:
            subprocess.run([str(REPO/'.venv-verl/bin/ray'),'status','--address=172.31.11.186:6379'],cwd=REPO)
            print('\nRay reservations are logical resources; use the GPU window for physical compute use.')
        if not args.watch:
            break
        time.sleep(5)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
