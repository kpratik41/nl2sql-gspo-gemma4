"""Fail closed unless the persisted FSDP/CP numerical, capacity and restore gates pass."""
import hashlib
import json
import math
from pathlib import Path
from hydra import compose, initialize_config_dir

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / 'outputs/fsdp_cp_recipe'


def read(name):
    return json.loads((ROOT / name).read_text())


def main():
    with initialize_config_dir(config_dir=str(REPO / 'src/nl2sql_verl/configs'), version_base=None):
        cfg = compose(config_name='fsdp_cp2')
    micro = int(cfg.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu)
    assert micro==1 and cfg.actor_rollout_ref.model.override_config.nl2sql_precision=='ieee'
    assert cfg.actor_rollout_ref.model.override_config.nl2sql_static_shapes is True
    assert cfg.actor_rollout_ref.model.override_config.nl2sql_attention_precision=='ieee'
    numerical = {}
    for name in ('native_static_fp32_sp1_full', 'native_static_fp32_sp2_short',
                 'native_static_fp32_sp2_long', 'native_static_fp32_sp2_full',
                 'native_static_fp32_sp2_full_repeat'):
        result = read(name + '.json')
        assert result['fp32_output'] and result['fp32_residual'] and result['attention_fp32'], name
        assert result['linear_arithmetic']=='ieee' and result['static_shapes'], name
        assert result['attention_arithmetic']=='ieee' and not result['canonical_layout'], name
        assert result['cp']*result['dp']==8 and result['microbatch']==micro, name
        launch=read(name+'.launch.json')
        for source in ('src/nl2sql_verl/fsdp_engine.py','src/nl2sql_verl/fsdp_attention.py','src/nl2sql_verl/worker_python.sh'):
            assert hashlib.sha256((REPO/source).read_bytes()).hexdigest()==launch['source_sha256'][source], (name,source)
        assert result['pass_gradient'], name
        assert result['loss_abs_difference'] < .01, name
        assert result['max_logprob_abs_difference'] < .1, name
        assert len(result['rows']) == 20, name
        assert all(r['finite'] and r['relative_l2'] < .03 and r['cosine'] > .999
                   for r in result['rows']), name
        numerical[name] = max(r['relative_l2'] for r in result['rows'])
    attention = {}
    capacity = {}
    for sp in (1, 2):
        rows = read(f'attention_memory_fp32_ieee_static_sp{sp}_rank0.json')
        assert [r['global_tokens'] for r in rows] == [4096, 8192, 16384, 32768]
        assert all(r['finite'] and 'mask_construction_peak_mib' in r for r in rows)
        peaks = [r['peak_allocated_mib'] for r in rows]
        assert all(b/a < 2.3 for a, b in zip(peaks, peaks[1:])), peaks
        attention[f'cp{sp}'] = rows
        reports = [read(f'full_memory_static_fp32_sp{sp}_rank{rank}.json') for rank in range(8)]
        for report in reports:
            assert report['sp'] == sp and report['fp32_residual'] and report['fp32_output']
            assert report['linear_arithmetic']=='ieee' and report['static_shapes']
            assert report['attention_fp32'] and report['attention_arithmetic']=='ieee'
            assert [r['length'] for r in report['reports']] == [4096, 8192, 16384, 32768]
            for item in report['reports']:
                for trial in item['passes']:
                    norm = trial['grad_norm']
                    assert norm is None or (math.isfinite(norm) and norm > 0)
        capacity[f'cp{sp}'] = max(t['peak_allocated_gib'] for r in reports
                                for item in r['reports'] for t in item['passes'])
    for rank in range(8):
        report = read(f'long_update_static_fp32_rank{rank}.json')
        assert report['sp'] == 2 and report['fp32_residual'] and report['fp32_output'] and report['accumulation_rounds'] == 2
        assert report['linear_arithmetic']=='ieee' and report['static_shapes']
        assert report['attention_fp32'] and report['attention_arithmetic']=='ieee'
        assert report['reports'][0]['length'] == 13312
        assert report['reports'][0]['microbatch'] == micro
        restore = read(f'long_update_static_fp32_restore_rank{rank}.json')
        assert restore['pass'] and restore['parameters'] > 1000 and restore['optimizer_tensors'] > 1000
    cpu = (ROOT / 'cpu_tests.log').read_text()
    assert '34 passed' in cpu and 'FAILED ' not in cpu
    manifest = json.loads((REPO / 'outputs/cp1_recipe/data/manifest.json').read_text())
    assert manifest['raw_9428_used'] is False
    dataset_hashes = {}
    for split, count in (('train', 6601), ('validation', 640)):
        path = REPO / f'outputs/cp1_recipe/data/{split}.parquet'
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert manifest['splits'][split]['count'] == count
        assert digest == manifest['splits'][split]['sha256']
        dataset_hashes[str(path.relative_to(REPO))] = digest
    integration = False
    smoke = ROOT / 'smoke'
    if (smoke / 'status.json').exists():
        assert read('smoke/status.json')['exit_code'] == 0
        metrics = {}
        for path in smoke.glob('metrics*.jsonl'):
            for line in path.read_text().splitlines():
                row = json.loads(line)
                metrics.setdefault(row['step'], {}).update(row['data'])
        for step in (1, 2):
            assert metrics[step]['reward/trajectory_count'] == 32
            assert math.isfinite(metrics[step]['actor/grad_norm']) and metrics[step]['actor/grad_norm'] > 0
            checkpoint = Path(f'/opt/dlami/nvme/nl2sql/fsdp-cp2-smoke/global_step_{step}')
            assert json.loads((checkpoint / 'nl2sql_complete.json').read_text())['step'] == step
        for step in (0, 2):
            assert len((smoke / f'validation/{step}.jsonl').read_text().splitlines()) == 640
        integration = True
    sources = [p for p in (REPO / 'src/nl2sql_verl').rglob('*')
               if p.suffix in ('.py', '.yaml', '.sh', '.txt')
               and 'tests' not in p.parts and '__pycache__' not in p.parts]
    result = dict(numerical_and_memory_ready=True, integration_ready=integration,
                  microbatch=micro, accumulation_rounds=512//4//micro,
                  numerical=numerical, attention_memory=attention,
                  full_model_peak_allocated_gib=capacity, dataset_sha256=dataset_hashes,
                  source_sha256={str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(sources)})
    (ROOT / 'readiness.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_sha256', 'attention_memory')}, indent=2))


if __name__ == '__main__':
    main()
