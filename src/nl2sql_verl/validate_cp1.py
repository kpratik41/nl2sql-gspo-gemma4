"""Record readiness only after the CP1 CPU, GPU, and real-training checks pass.

Run the commands documented in README first; this verifies their persisted
results, dataset hashes and a completed end-to-end smoke checkpoint.
"""
import hashlib,json,math,re
from pathlib import Path

REPO=Path(__file__).resolve().parents[2]
ROOT=REPO/'outputs/cp1_recipe'


def main():
    cpu=(ROOT/'cpu_tests.log').read_text();native=(ROOT/'native_dapo_tests.log').read_text()
    assert re.search(r'\b31 passed\b',cpu) and 'FAILED ' not in cpu
    assert re.search(r'\b12 passed\b',native) and 'FAILED ' not in native
    tiny=json.loads((ROOT/'tiny_microbatch_bf16.json').read_text())
    gradient=json.loads((ROOT/'full_microbatch_bf16_tp2dp4_synthetic.json').read_text())
    bird_gradient=json.loads((ROOT/'full_microbatch_bf16_bird_tp2dp4.json').read_text())
    memory=json.loads((ROOT/'full_memory_bf16.json').read_text())
    assert tiny['pass'] and tiny['checkpoint_restored'] and gradient['pass'] and memory['pass']
    assert bird_gradient['pass']
    for result in (tiny,gradient,bird_gradient,memory):
        assert (result['tp'],result['dp'],result['cp'],result['dtype'])==(2,4,1,'bf16')
    assert memory['deferred_gradient_sync'] and len(memory['lengths'])==4
    assert all(r['grad_norm'] is None or (math.isfinite(r['grad_norm']) and r['grad_norm']>0) for r in memory['ranks'])
    smoke=ROOT/'smoke';status=json.loads((smoke/'training.status.json').read_text())
    assert status['exit_code']==0 and status['status']=='pass'
    logs={}
    for p in smoke.glob('metrics*.jsonl'):
        for line in p.read_text().splitlines():
            row=json.loads(line);logs.setdefault(row['step'],{}).update(row['data'])
    step=logs[1];assert step['reward/trajectory_count']==32
    assert math.isfinite(step['actor/grad_norm']) and step['actor/grad_norm']>0
    validation=[json.loads(x) for x in (smoke/'validation/1.jsonl').read_text().splitlines()]
    assert len(validation)==640
    ckpt=Path('/opt/dlami/nvme/nl2sql/cp1-smoke/global_step_1')
    assert json.loads((ckpt/'nl2sql_complete.json').read_text())['step']==1
    assert len(list((ckpt/'actor').glob('rng_rank*.pt')))==8
    manifest=json.loads((ROOT/'data/manifest.json').read_text())
    for split,count in (('train',6601),('validation',640)):
        p=ROOT/f'data/{split}.parquet'
        assert manifest['splits'][split]['count']==count
        assert hashlib.sha256(p.read_bytes()).hexdigest()==manifest['splits'][split]['sha256']
    sources=[p for p in (REPO/'src/nl2sql_verl').rglob('*') if p.suffix in ('.py','.yaml','.sh','.txt') and '__pycache__' not in p.parts and 'tests' not in p.parts]
    sources.append(REPO/'benchmarks/cp_flex_options.py')
    report={'ready':True,'cpu_tests':31,'native_dapo_tests':12,'tiny_microbatch_and_restore':True,
        'full_model_max_gradient_relative_l2':max(r['worst_relative_l2'] for r in gradient['ranks']),
        'full_model_peak_allocated_gib':max(r['allocated_gib'] for r in memory['ranks']),
        'full_model_peak_reserved_gib':max(r['reserved_gib'] for r in memory['ranks']),
        'smoke_accepted_trajectories':32,'smoke_validation_examples':640,
        'smoke_dev_correct':sum(r['acc'] for r in validation),'smoke_train_reward':step['critic/score/mean'],
        'full_model_bird_max_gradient_relative_l2':max(r['worst_relative_l2'] for r in bird_gradient['ranks']),
        'actor_tp':2,'actor_dp':4,'rollout_tp':4,'rollout_dp':2,'cp':1,
        'production_accepted_trajectories_per_update':512,'production_accumulation_rounds':64,
        'production_steps':100,'save_frequency':5,'eval_frequency':10,
        'precision':'BF16 compute; FP32 master weights, optimizer state, and FSDP gradient reductions',
        'source_sha256':{str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)}}
    (ROOT/'readiness.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},indent=2))


if __name__=='__main__':main()
