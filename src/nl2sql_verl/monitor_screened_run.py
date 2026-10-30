"""Terminal status for the screened1039/8K job, without changing training."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import time
from zoneinfo import ZoneInfo

from .monitor_h200 import read_json, tail, ANSI

ROOT = Path(__file__).resolve().parents[2] / 'outputs/h200_training/screened1039-8k'
EXPORTS = Path('/opt/dlami/nvme/nl2sql/model-exports/screened1039-8k')


def status():
    run = ROOT / 'train'
    records = []
    for line in tail(run / 'metrics.jsonl', 4 * 1024**2).splitlines():
        try:
            records.append(json.loads(line))
        except ValueError:
            pass
    step = max((r.get('step', 0) for r in records), default=0)
    log = ANSI.sub('', tail(run / 'training.log', 2 * 1024**2))
    ended = read_json(run / 'status.json')
    heartbeat = read_json(run / 'heartbeat.json')
    checkpoints = sorted(int(p.parent.name.removeprefix('global_step_'))
                         for p in EXPORTS.glob('global_step_*/nl2sql_model_only_complete.json'))
    if ended:
        phase = 'completed' if ended.get('exit_code') == 0 else 'FAILED'
    elif records:
        phase = 'training cycle: rollouts, DAPO filtering, actor computation'
        if checkpoints and checkpoints[-1] > step:
            phase = f'update{checkpoints[-1]} saved; evaluation/metrics pending'
    elif 'all initialize finished, ready to fit' in log:
        phase = 'step0 greedy baseline evaluation'
    else:
        phase = 'initializing actor and rollout servers'
    print(datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d %H:%M:%S %Z'))
    print(f'Phase: {phase} | Completed optimizer updates: {step}/100')
    print('1039 training prompts | 16 generations | 32 accepted prompts/update')
    print('16 GPUs (node0+node1) | FSDP2/CP2/DP8 | async vLLM TP4 x4')
    print('8192 cumulative response tokens | trainT1.2 | evalT0 at0,5,10,...')
    print(f'Elapsed: {heartbeat.get("elapsed_seconds", 0)/60:.1f} minutes')
    print(f'Model-only checkpoints retained: {checkpoints or "none yet"} (all retained on NVMe)')
    validation = next((r['data'] for r in records if r.get('step') == 0), {})
    accuracy = validation.get('val-core/nl2sql_bird/acc/mean@1')
    if accuracy is not None:
        print(f'Step0 result accuracy (640 questions): {accuracy:.2%}')
    urls = re.findall(r'https://[^\s]+/runs/[A-Za-z0-9_-]+', log)
    saved_url = run / 'wandb_url.txt'
    url = saved_url.read_text().strip() if saved_url.exists() else (urls[-1] if urls else '')
    if url:
        print('W&B: ' + url)
    print('Log: ' + str(run / 'training.log'))
    print('Training is owned by systemd; closing tmux does not stop the run.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    try:
        while True:
            if args.watch:
                print('\033[2J\033[H', end='', flush=True)
            status()
            if not args.watch:
                break
            time.sleep(10)
    except KeyboardInterrupt:
        pass
