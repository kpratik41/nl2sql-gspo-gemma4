"""Finish capacity validation, one integration update, then authorized training."""
import json
from pathlib import Path
import subprocess
import time

from .run_screened_training import REPO, ROOT, CHECKS, capacity_ready


def stage(name, **extra):
    ROOT.mkdir(parents=True, exist_ok=True)
    temporary = ROOT / 'pipeline.json.tmp'
    temporary.write_text(json.dumps({'stage': name, 'updated_unix': time.time(), **extra}, indent=2) + '\n')
    temporary.replace(ROOT / 'pipeline.json')
    print(json.dumps({'stage': name, **extra}), flush=True)


def main():
    stage('capacity_validation')
    status = REPO / 'outputs/h200_checks/screened_8k_capacity16/status.json'
    started = time.monotonic()
    while not status.exists():
        if time.monotonic() - started > 5400:
            raise TimeoutError('Capacity check did not finish')
        active = subprocess.check_output(['systemctl', '--user', 'show', 'nl2sql-screened-capacity',
                                         '--property=ActiveState', '--value'], text=True).strip()
        if active not in ('active', 'activating'):
            raise RuntimeError('Capacity check stopped without a passing result')
        time.sleep(10)
    subprocess.run(['rsync', '-a', f'h200-node1:{CHECKS}/capacity_rank*.json', str(CHECKS) + '/'], check=True)
    capacity = capacity_ready()
    stage('capacity_passed', scope=capacity)
    import ray
    ray.init(address='172.31.11.186:6379', ignore_reinit_error=True)
    started = time.monotonic()
    while len([n for n in ray.nodes() if n['Alive']]) != 2:
        if time.monotonic() - started > 300:
            raise TimeoutError('Two-node Ray cluster did not become ready')
        time.sleep(5)
    assert ray.cluster_resources()['GPU'] == 16
    ray.shutdown()
    stage('integration_update')
    subprocess.run([str(REPO / '.venv-verl/bin/python'), '-m', 'src.nl2sql_verl.run_screened_training', 'smoke'],
                   cwd=REPO, check=True)
    stage('training_starting_with_step0_eval')
    subprocess.run([str(REPO / '.venv-verl/bin/python'), '-m', 'src.nl2sql_verl.run_screened_training', 'train'],
                   cwd=REPO, check=True)
    stage('completed')


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        stage('failed', error=f'{type(error).__name__}: {error}')
        raise
