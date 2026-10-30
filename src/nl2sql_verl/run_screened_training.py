"""Own the authorized screened1039/8K two-node run, with immutable run records."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

REPO = Path(__file__).resolve().parents[2]
CHECKS = REPO / 'outputs/h200_checks/screened_8k_launch'
ROOT = REPO / 'outputs/h200_training/screened1039-8k'


def capacity_ready():
    status = json.loads((REPO / 'outputs/h200_checks/screened_8k_capacity16/status.json').read_text())
    assert status['status'] == 'pass' and status['node_exit_codes'] == [0, 0]
    scope = json.loads((CHECKS / 'capacity.scope.json').read_text())
    assert (scope['world_size'], scope['cp'], scope['response_loss_tokens']) == (16, 2, 8192)
    for rank in range(16):
        report = json.loads((CHECKS / f'capacity_rank{rank}.json').read_text())
        assert report['sp'] == 2 and report['linear_arithmetic'] == 'ieee'
        passes = report['reports'][0]['passes']
        assert len(passes) == 2
        assert all(p['grad_norm'] is None or math.isfinite(p['grad_norm']) for p in passes)
    return scope


def main(args):
    capacity = capacity_ready()
    if args.mode == 'train':
        smoke = json.loads((ROOT / 'smoke/status.json').read_text())
        assert smoke['exit_code'] == 0 and smoke['completed_updates'] == 1
        assert smoke['trajectories'] == 16 and math.isfinite(smoke['grad_norm'])
    import ray
    ray.init(address='172.31.11.186:6379', ignore_reinit_error=True)
    nodes = [n for n in ray.nodes() if n['Alive']]
    assert {n['NodeManagerAddress'] for n in nodes} == {'172.31.11.186', '172.31.2.174'}
    assert ray.cluster_resources()['GPU'] == 16
    ray.shutdown()
    root = ROOT / args.mode
    root.mkdir(parents=True, exist_ok=False)
    if args.mode == 'train':
        from .run_h200 import wandb_authenticated
        if not wandb_authenticated():
            raise RuntimeError('W&B online authentication failed; training was not started')
    manifest = json.loads((REPO / 'outputs/screened_dapo_8k/data/manifest.json').read_text())
    for split, meta in manifest.items():
        assert hashlib.sha256((REPO / f'outputs/screened_dapo_8k/data/{split}.parquet').read_bytes()).hexdigest() == meta['output_sha256']
    command = [str(REPO / 'src/nl2sql_verl/worker_python.sh'), '-m', 'src.nl2sql_verl.train',
        '--config-name', 'dapo_screened_8k_16gpu',
        'actor_rollout_ref.model.path=/opt/dlami/nvme/nl2sql/hf/hub/models--google--gemma-4-31B-it/snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475',
        'ray_kwargs.ray_init.runtime_env.py_executable=' + str(REPO / 'src/nl2sql_verl/worker_python.sh'),
        'trainer.rollout_data_dir=' + str(root / 'rollouts'),
        'trainer.validation_data_dir=' + str(root / 'validation')]
    if args.mode == 'smoke':
        command += ['data.train_batch_size=1', 'actor_rollout_ref.actor.ppo_mini_batch_size=1',
            'trainer.total_training_steps=1', 'trainer.val_before_train=false',
            'trainer.save_freq=-1', 'trainer.test_freq=-1', 'trainer.logger=[console,file]',
            'trainer.experiment_name=screened1039-8k-16gpu-integration',
            'trainer.default_local_dir=' + str(root / 'checkpoints')]
    environment = os.environ.copy()
    environment.update(VERL_PYTHON='/opt/dlami/nvme/nl2sql/runtime/venv/bin/python',
        VERL_SOURCE='/opt/dlami/nvme/nl2sql/runtime/verl',
        VERL_FILE_LOGGER_PATH=str(root / 'metrics.jsonl'),
        NL2SQL_DATABASE_DIR='/opt/dlami/nvme/nl2sql/databases', NL2SQL_LENGTH_MAX='8192',
        NL2SQL_LENGTH_BUFFER='512', RAY_ADDRESS='172.31.11.186:6379',
        OMP_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false',
        TORCHINDUCTOR_CACHE_DIR='/opt/dlami/nvme/nl2sql/caches/inductor',
        TRITON_CACHE_DIR='/opt/dlami/nvme/nl2sql/caches/triton',
        WANDB_MODE='online' if args.mode == 'train' else 'disabled',
        WANDB_DIR=str(root), WANDB_PROJECT='nl2sql-gemma4-verl')
    (root / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
    (root / 'launch_policy.json').write_text(json.dumps({'model': 'original_base', 'selected_prompts': 1039,
        'response_budget': 8192, 'world_size': 16, 'cp': 2, 'rollout_n': 16,
        'retain_all_model_checkpoints': True, 'capacity': capacity,
        'data_manifest': manifest, 'mode': args.mode}, indent=2) + '\n')
    source = list((REPO / 'src/nl2sql_verl').glob('*.py')) + list((REPO / 'src/nl2sql_verl/configs').rglob('*.yaml'))
    (root / 'source_sha256.json').write_text(json.dumps({str(p.relative_to(REPO)):
        hashlib.sha256(p.read_bytes()).hexdigest() for p in source}, indent=2) + '\n')
    started = time.monotonic()
    with (root / 'training.log').open('w', buffering=1) as stream:
        process = subprocess.Popen(command, cwd=REPO, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        while process.poll() is None:
            (root / 'heartbeat.json').write_text(json.dumps({'state': 'running', 'mode': args.mode,
                'pid': process.pid, 'elapsed_seconds': time.monotonic() - started}) + '\n')
            time.sleep(10)
    metrics = [json.loads(line) for line in (root / 'metrics.jsonl').open()] if (root / 'metrics.jsonl').exists() else []
    # File logger stores training/global_step in metric dictionaries; validation
    # at step0 must not be counted as an optimizer update.
    steps = [r.get('step', 0) for r in metrics if r.get('step', 0) > 0]
    last = next((r['data'] for r in reversed(metrics) if r.get('step', 0) > 0), {})
    result = {'exit_code': process.returncode, 'elapsed_seconds': time.monotonic() - started,
              'completed_updates': max(steps, default=0), 'mode': args.mode,
              'grad_norm': last.get('actor/grad_norm'), 'trajectories': last.get('reward/trajectory_count'),
              'actual_tools': last.get('tools/actual_executions')}
    (root / 'status.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    raise SystemExit(process.returncode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['smoke', 'train'])
    main(parser.parse_args())
