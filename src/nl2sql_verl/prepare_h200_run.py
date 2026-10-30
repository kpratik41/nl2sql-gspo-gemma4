"""Run the remaining owned H200 gates, then smoke and the authorized pilot.

Execute under a systemd user service from the repository. Every required gate
must pass before the next stage; an optional microbatch-2 failure selects one.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO/'outputs/h200_checks'
PREFIX = 'stable_ieee'
PYTHON = str(REPO/'.venv-verl/bin/python')
MODEL = '/opt/dlami/nvme/nl2sql/hf/hub/models--google--gemma-4-31B-it/snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475'


def run(command, *, optional=False):
    print('Running:', ' '.join(command), flush=True)
    result = subprocess.run(command, cwd=REPO)
    if result.returncode and not optional:
        raise RuntimeError(f'Stage failed with exit code {result.returncode}')
    return result.returncode


def manifest(name, records):
    path = ROOT/name
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(records, indent=2)+'\n')
    run(['rsync', '-a', str(path), f'h200-node1:{path}'])
    return str(path)


def comparison(label, micro=1):
    kind = 'long' if label.startswith('long') else 'real'
    return {'lengths': [13312, 13312],
            'row_indices': [4837, 2737] if kind == 'long' else [0, 1],
            'reference': str(ROOT/f'full_hf_selected_{kind}.pt'),
            'output': str(ROOT/f'{PREFIX}_{label}'), 'micro': micro}


def probe(name, micro, primary, comparisons, exercises):
    common = [PYTHON, '-m', 'src.nl2sql_verl.run_multinode_probe',
              '--name', name, '--port', '29642', '--timeout', '7200',
              '--module', 'src.nl2sql_verl.tests.fsdp_cp_probe', '--',
              '--mode', 'native', '--compat', '--backend', 'nl2sql_flex_ulysses',
              '--model', MODEL, '--sp', '2', '--micro', str(micro),
              '--bucket', '256', '--fp32-ieee', '--gather-global-key-projection']
    for key in ('lengths', 'row_indices'):
        common += ['--'+key.replace('_', '-'), *map(str, primary[key])]
    common += ['--reference', primary['reference'], '--output', primary['output']]
    if comparisons:
        common += ['--additional-comparisons', manifest(name+'_comparisons.json', comparisons)]
    common += ['--exercise-output', str(ROOT/f'{PREFIX}_mb{micro}_memory'),
               '--exercise-lengths', '13312', '--exercise-accum', '2']
    if exercises:
        common += ['--additional-exercises', manifest(name+'_exercises.json', exercises)]
    return run(common, optional=(micro == 2))


def collect_reports():
    run(['rsync', '-a', f'h200-node1:{ROOT}/{PREFIX}_*_memory_rank*.json', str(ROOT)+'/'])


def state(stage, **extra):
    path = ROOT/'pipeline_status.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps({'stage': stage, 'updated_epoch': time.time(), **extra}, indent=2)+'\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-smoke', action='store_true',
                        help='Revalidate existing numerical/memory evidence and retry integration')
    args = parser.parse_args()
    if args.from_smoke:
        run([PYTHON, '-m', 'src.nl2sql_verl.validate_h200'])
        return smoke_and_pilot()
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=str(REPO/'src/nl2sql_verl/configs'), version_base=None):
        cfg = compose(config_name='dapo_fsdp_cp2_16gpu')
    context = cfg.data.max_prompt_length + cfg.data.max_response_length
    state('baseline_numerical_and_memory')
    primary = {'lengths': [257, 385], 'row_indices': [0, 1],
               'reference': str(ROOT/'full_hf_selected_short.pt'),
               'output': str(ROOT/f'{PREFIX}_short')}
    probe('stable_ieee16_baseline', 1, primary,
          [comparison(label) for label in ('real', 'long', 'real_repeat')],
          [{'lengths': [context], 'output': str(ROOT/f'{PREFIX}_{context}_memory')}])
    collect_reports()
    state('microbatch_two')
    code = probe('stable_ieee16_micro2', 2, comparison('real_micro2', 2),
                 [comparison('long_micro2', 2)], [])
    collect_reports()
    state('validate_readiness', microbatch_two_exit_code=code)
    run([PYTHON, '-m', 'src.nl2sql_verl.validate_h200'])
    smoke_and_pilot()


def smoke_and_pilot():
    state('five_update_smoke')
    run([PYTHON, '-m', 'src.nl2sql_verl.run_h200', 'smoke'])
    status_path = REPO/'outputs/h200_training/smoke/status.json'
    deadline = time.monotonic()+7500
    while not status_path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError('Smoke supervisor did not publish completion status')
        time.sleep(15)
    run([PYTHON, '-m', 'src.nl2sql_verl.validate_h200', '--integration'])
    state('launching_pilot')
    run([PYTHON, '-m', 'src.nl2sql_verl.run_h200', 'pilot'])
    state('pilot_launched', pilot_status='outputs/h200_training/pilot/heartbeat.json')


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        state('failed', error=f'{type(error).__name__}: {error}')
        raise
