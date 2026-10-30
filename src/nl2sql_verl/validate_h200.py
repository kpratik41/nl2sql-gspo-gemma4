"""Gate H200 launches on saved numerical, capacity, data and integration evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import math

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO/'outputs/h200_checks'


def read(name):
    return json.loads((ROOT/name).read_text())


def numerical(name):
    result = read(name+'.json')
    assert result['cp'] == 2 and result['dp'] == 8 and result['static_shapes']
    assert result['attention_arithmetic'] == 'ieee'
    assert result['linear_arithmetic'] == 'ieee' and result['gather_global_key_projection']
    recorded = read(name+'.launch.json')['source_sha256']
    assert all(hashlib.sha256((REPO/p).read_bytes()).hexdigest() == digest
               for p, digest in recorded.items()), 'Probe sources changed after measurement'
    assert result['pass_gradient']
    assert result['loss_abs_difference'] < .01
    assert result['max_logprob_abs_difference'] < .1
    assert len(result['rows']) == 20
    assert all(r['finite'] and r['relative_l2'] < .03 and r['cosine'] > .999 for r in result['rows'])
    return {'worst_gradient_relative_l2': max(r['relative_l2'] for r in result['rows']),
            'max_logprob_abs_difference': result['max_logprob_abs_difference']}


def capacity(prefix, length):
    reports = [read(f'{prefix}_rank{rank}.json') for rank in range(16)]
    trials = []
    for report in reports:
        assert report['sp'] == 2 and report['static_shapes'] and report['attention_arithmetic'] == 'ieee'
        assert report['gather_global_key_projection']
        trial = next((r for r in report['reports'] if r['length'] == length), None)
        assert trial is not None, f'{prefix} rank {len(trials)} lacks a {length}-token capacity trial'
        assert len(trial['passes']) == 2
        assert all(p['grad_norm'] is None or math.isfinite(p['grad_norm']) for p in trial['passes'])
        trials.append(trial)
    return {
        'warm_seconds': max(r['passes'][1]['seconds'] for r in trials),
        'peak_allocated_gib': max(p['peak_allocated_gib'] for r in trials for p in r['passes']),
        'peak_reserved_gib': max(p['peak_reserved_gib'] for r in trials for p in r['passes']),
        'microbatch': trials[0]['microbatch'],
        'length': length}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--precision', choices=['ieee', 'tf32'], default='ieee')
    parser.add_argument('--prefix', default='stable_ieee')
    parser.add_argument('--integration', action='store_true')
    args = parser.parse_args()
    if args.integration:
        ready = read('readiness.json')
        status = json.loads((REPO/'outputs/h200_training/smoke/status.json').read_text())
        assert status['exit_code'] == 0
        from .model_checkpoints import MARKER, FORMAT
        model = Path('/home/ubuntu/nl2sql-checkpoints/dapo-fsdp-cp2-16gpu-smoke/global_step_5')
        metadata = json.loads((model/MARKER).read_text())
        assert metadata['format'] == FORMAT and metadata['step'] == 5
        assert not list(model.glob('optim*'))
        assert (model/'model.safetensors.index.json').is_file()
        metrics_path = REPO/'outputs/h200_training/smoke/metrics.jsonl'
        records = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
        # The file logger wraps each record as {step, data}; verify actual
        # completion metrics rather than relying solely on process exit.
        assert any(r.get('step') == 5 for r in records), 'No step-five metrics'
        final_metrics = [r['data'] for r in records if r.get('step') == 5]
        assert any(any(k.startswith('val-core/') for k in row) for row in final_metrics)
        for step in (0, 5):
            path = REPO/f'outputs/h200_training/smoke/validation/{step}.jsonl'
            assert sum(bool(line.strip()) for line in path.open()) == 640, f'Incomplete validation: {path}'
        assert 50_000_000_000 < metadata['weight_bytes'] < 70_000_000_000
        ready['integration_ready'] = True
        ready['integration_model_export'] = str(model)
        (ROOT/'readiness.json').write_text(json.dumps(ready, indent=2)+'\n')
        print(json.dumps({'integration_ready': True, 'model': str(model)}))
        return
    assert read('collectives.json')['status'] == 'pass'
    assert read('model_only_export.json')['status'] == 'pass'
    for rank in range(16):
        assert read(f'tiny_updates_restore_rank{rank}.json')['pass']
    control = read('full_cp1_short.json')
    assert control['pass_gradient'] and control['linear_arithmetic'] == 'ieee'
    assert control['loss_abs_difference'] < .01 and control['max_logprob_abs_difference'] < .1
    for label in ('real', 'long'):
        control = read(f'full_cp1_{label}.json')
        assert control['linear_arithmetic'] == 'ieee' and control['microbatch'] == 1
    assert args.precision == 'ieee', 'H200 global-key projection requires IEEE FP32'
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=str(REPO/'src/nl2sql_verl/configs'), version_base=None):
        cfg = compose(config_name='dapo_fsdp_cp2_16gpu')
    context = cfg.data.max_prompt_length + cfg.data.max_response_length
    assert context == cfg.actor_rollout_ref.rollout.max_model_len
    prefix = args.prefix
    comparisons = {label: numerical(f'{prefix}_{label}') for label in ('short', 'real', 'long', 'real_repeat')}
    mb1 = capacity(prefix+'_mb1_memory', 13312)
    assert mb1['peak_allocated_gib'] < 127.8 and mb1['peak_reserved_gib'] < 131.8
    context_prefix = prefix+f'_{context}_memory'
    # Retain compatibility with the original 32K evidence; it cannot validate
    # a future configuration whose context limit is larger.
    if context == 32768 and not (ROOT/f'{context_prefix}_rank0.json').exists():
        context_prefix = prefix+'_32k_memory'
    long_memory = capacity(context_prefix, context)
    assert long_memory['peak_reserved_gib'] < 139.8
    selected = 1
    candidate = None
    candidate_error = None
    try:
        candidate_checks = {label: numerical(f'{prefix}_{label}') for label in ('real_micro2', 'long_micro2')}
        candidate = capacity(prefix+'_mb2_memory', 13312)
        if (candidate['peak_allocated_gib'] < 127.8 and candidate['peak_reserved_gib'] < 131.8
                and candidate['warm_seconds'] <= mb1['warm_seconds'] * 1.05):
            selected = 2
        comparisons.update(candidate_checks)
    except (AssertionError, FileNotFoundError, StopIteration) as error:
        candidate_error = type(error).__name__+': '+str(error)
    data_root = (REPO/cfg.data.train_files).parent
    for split in ('train', 'validation', 'dev_full'):
        lengths = json.loads((data_root/f'{split}_prompt_lengths.json').read_text())
        assert max(lengths) <= cfg.data.max_prompt_length
    data_validation = json.loads((data_root/'validation_report.json').read_text())
    assert data_validation['status'] == 'pass' and data_validation['profile'] == 'gemma4_native_v1'
    sources = ['fsdp_engine.py', 'fsdp_attention.py', 'fsdp_projection.py', 'fsdp_runtime.py', 'bounded_runtime.py',
               'model_checkpoints.py', 'checkpoint_retention.py', 'vllm_compat.py', 'task.py',
               'tools.py', 'agent_loop.py', 'train.py', 'run_h200.py', 'worker_python.sh',
               'launch_dapo_16gpu.sh', 'configs/dapo_fsdp_cp2_16gpu.yaml',
               'configs/fsdp_cp2_base.yaml', 'configs/base.yaml',
               'configs/actor/nl2sql_fsdp_actor.yaml']
    sources += ['native_agent_loop.py', 'native_tools.py', 'native_runtime.py', 'reward_entry_native.py',
                'gemma_native_profile.py', 'tool_profiles.py', 'prepare_native_data.py',
                'configs/agent_native.yaml', 'configs/tools_native.yaml']
    paths = ['src/nl2sql_verl/'+p for p in sources]
    datasets = [str((data_root/f'{s}.parquet').relative_to(REPO)) for s in ('train', 'validation')]
    digest = lambda p: hashlib.sha256((REPO/p).read_bytes()).hexdigest()
    ready = {
        'numerical_and_memory_ready': True, 'integration_ready': False,
        'data_profile': 'gemma4_native_v1',
        'selected_precision': args.precision, 'selected_microbatch': selected,
        'selected_accumulation_rounds': 64//selected,
        'numerical_comparisons': comparisons, 'microbatch_1': mb1,
        'microbatch_2': candidate, 'microbatch_2_error': candidate_error,
        'validated_context_length': context,
        'memory_max_context_microbatch_1': long_memory,
        'selection_rule': 'Largest passing microbatch, >=8GiB reserved-memory headroom and <=5% warm-time regression',
        'source_sha256': {p: digest(p) for p in paths},
        'dataset_sha256': {p: digest(p) for p in datasets},
        'numerical_scope': '20 selected gradients against independent HF short, real and long controls',
        'single_gpu_long_hf_reference': 'Selected response logits preserve the loss and fit full long HF references on H200'}
    (ROOT/'readiness.json').write_text(json.dumps(ready, indent=2)+'\n')
    print(json.dumps({k:v for k,v in ready.items() if not k.endswith('sha256')}, indent=2))


if __name__ == '__main__':
    main()
