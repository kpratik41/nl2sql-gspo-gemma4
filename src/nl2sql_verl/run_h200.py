"""Supervise the two-node H200 DAPO smoke/pilot/training job."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO/'outputs/h200_training'


def wandb_authenticated():
    """Check the SDK credential store without exposing secrets in job metadata."""
    try:
        import wandb
        api = wandb.Api(timeout=10)
        if not api.api_key:
            return False
        viewer = api.viewer
        if not viewer:
            return False
        marker = Path.home()/'.config/nl2sql/wandb_verified.json'
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({'verified': True})+'\n')
        return True
    except Exception:
        return False


def command(mode, run, microbatch):
    checkpoint = f'/opt/dlami/nvme/nl2sql/checkpoints/dapo-fsdp-cp2-16gpu-{mode}'
    backup = f'/home/ubuntu/nl2sql-checkpoints/dapo-fsdp-cp2-16gpu-{mode}'
    cmd = ['bash', str(REPO/'src/nl2sql_verl/launch_dapo_16gpu.sh'),
           f'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={microbatch}',
           f'trainer.default_local_dir={checkpoint}',
           f'trainer.nl2sql_model_backup_dir={backup}',
           f'trainer.rollout_data_dir={run}/rollouts',
           f'trainer.validation_data_dir={run}/validation',
           f'trainer.experiment_name=h200-dapo-cp2-mb{microbatch}-{mode}',
           'trainer.resume_mode=disable']
    if mode == 'smoke':
        cmd += ['data.train_batch_size=2', 'actor_rollout_ref.actor.ppo_mini_batch_size=2',
                'trainer.total_training_steps=5', 'trainer.logger=[console,file,wandb]']
    elif mode == 'pilot':
        cmd += ['trainer.nl2sql_training_seconds=7200']
    elif mode == 'train':
        # Versioned native prompts differ from the ongoing legacy run: its
        # historical base evaluation is not a valid new-run baseline.
        cmd += ['trainer.val_before_train=true']
    return cmd


def validate_readiness(mode, allow_incomplete_smoke=False):
    ready = json.loads((REPO/'outputs/h200_checks/readiness.json').read_text())
    if not ready.get('numerical_and_memory_ready'):
        raise RuntimeError('H200 numerical/memory gates are incomplete')
    if (mode != 'smoke' and not ready.get('integration_ready')
            and not (mode == 'train' and allow_incomplete_smoke)):
        raise RuntimeError('H200 end-to-end smoke gate is incomplete')
    for path, expected in ready['source_sha256'].items():
        if hashlib.sha256((REPO/path).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'{path} changed since validation')
    for path, expected in ready['dataset_sha256'].items():
        if hashlib.sha256((REPO/path).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'{path} changed since validation')
    return ready


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['smoke', 'pilot', 'train'])
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--allow-incomplete-smoke', action='store_true',
                        help='Explicitly start train after numerical/memory gates before smoke integration completes')
    a = parser.parse_args()
    if a.allow_incomplete_smoke and a.mode != 'train':
        raise ValueError('--allow-incomplete-smoke is only supported for train')
    ready = validate_readiness(a.mode, a.allow_incomplete_smoke)
    run = ROOT/a.mode
    if not a.run:
        if run.exists():
            raise FileExistsError(f'{run} exists; preserve it before starting another job')
        run.mkdir(parents=True)
        (run/'launch_policy.json').write_text(json.dumps({
            'integration_ready': ready.get('integration_ready', False),
            'allow_incomplete_smoke': a.allow_incomplete_smoke,
            'initial_evaluation': 'fresh native-profile evaluation before training',
        }, indent=2)+'\n')
        subprocess.run(['systemd-run', '--user', f'--unit=nl2sql-h200-dapo-{a.mode}',
                        '--collect', f'--property=WorkingDirectory={REPO}',
                        '--property=KillMode=control-group', '--property=TimeoutStopSec=120',
                        str(REPO/'.venv-verl/bin/python'), '-m', 'src.nl2sql_verl.run_h200',
                        a.mode, '--run',
                        *(['--allow-incomplete-smoke'] if a.allow_incomplete_smoke else [])], check=True)
        return
    env = os.environ.copy()
    env.update(VERL_PYTHON='/opt/dlami/nvme/nl2sql/runtime/venv/bin/python',
               VERL_SOURCE='/opt/dlami/nvme/nl2sql/runtime/verl',
               VERL_FILE_LOGGER_PATH=str(run/'metrics.jsonl'),
               NL2SQL_DATABASE_DIR='/opt/dlami/nvme/nl2sql/databases',
               TORCHINDUCTOR_CACHE_DIR='/opt/dlami/nvme/nl2sql/caches/inductor',
               TRITON_CACHE_DIR='/opt/dlami/nvme/nl2sql/caches/triton',
               RAY_ADDRESS='172.31.11.186:6379', OMP_NUM_THREADS='2',
               WANDB_DIR=str(run), WANDB_PROJECT='nl2sql-gemma4-verl')
    # Never put API keys in commands, job metadata, source, or Ray runtime_env.
    # A successful wandb login supplies credentials through the user's SDK store.
    if not wandb_authenticated():
        env['WANDB_MODE'] = 'offline'
        env.pop('WANDB_API_KEY', None)
    cmd = command(a.mode, run, ready['selected_microbatch'])
    (run/'command.json').write_text(json.dumps(cmd, indent=2)+'\n')
    timeout = 7200 if a.mode == 'smoke' else (21600 if a.mode == 'pilot' else None)
    start = time.monotonic()
    with (run/'training.log').open('w', buffering=1) as stream:
        proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, start_new_session=True)
        reason = 'process_exit'
        try:
            while proc.poll() is None:
                (run/'heartbeat.json').write_text(json.dumps({
                    'pid': proc.pid, 'elapsed_seconds': time.monotonic()-start,
                    'state': 'running', 'microbatch': ready['selected_microbatch'],
                    'wandb_mode': env.get('WANDB_MODE', 'online')})+'\n')
                if timeout is not None and time.monotonic()-start > timeout:
                    reason = 'watchdog'
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=120)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                    break
                time.sleep(15)
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
        code = proc.wait()
    (run/'status.json').write_text(json.dumps({
        'exit_code': code, 'elapsed_seconds': time.monotonic()-start,
        'stop_reason': reason, 'selected_microbatch': ready['selected_microbatch'],
        'wandb_mode': env.get('WANDB_MODE', 'online')}, indent=2)+'\n')
    raise SystemExit(code)


if __name__ == '__main__':
    main()
