"""Reconcile versioned data, fixed samples, actor capacity and Ray smoke evidence."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from .prepare_reference_data import file_hash
from .validate_h200 import numerical

REPO=Path(__file__).resolve().parents[2]


def validate(root):
    from hydra import compose, initialize_config_dir
    root=Path(root)
    data=REPO/'outputs/gemma4_native_v1/data'
    manifest=json.loads((data/'manifest.json').read_text())
    report=json.loads((data/'validation_report.json').read_text())
    assert report['status']=='pass' and report['remaining_none_targets']==0
    for split,item in manifest['splits'].items():
        assert file_hash(data/f'{split}.parquet')==item['sha256']
    with initialize_config_dir(config_dir=str(REPO/'src/nl2sql_verl/configs'),version_base=None):
        cfg=compose(config_name='dapo_fsdp_cp2_16gpu')
    assert cfg.data.max_response_length==4096 and cfg.actor_rollout_ref.rollout.max_model_len==34816
    assert cfg.data.train_files=='outputs/gemma4_native_v1/data/train.parquet'
    assert cfg.actor_rollout_ref.model.external_lib=='src.nl2sql_verl.native_runtime'
    cohort={}
    rescoring=json.loads((root/'strict_rescoring.json').read_text())
    assert rescoring['reward_source_sha256']==file_hash('src/nl2sql_verl/reward_entry_native.py')
    for label in ['base','step5']:
        directory=root/label
        provenance=json.loads((directory/'provenance.json').read_text())
        assert provenance['sample_sha256']==file_hash(data/'fixed_validation_sample.json')
        for path,digest in provenance['source_sha256'].items():
            if path.endswith('/reward_entry_native.py'):
                assert rescoring['cohorts'][label]['original_reward_source_sha256']==digest
            else:
                assert file_hash(path)==digest, path
        assert file_hash(directory/'results.jsonl')==rescoring['cohorts'][label]['input_sha256']
        assert file_hash(directory/'strict_results.jsonl')==rescoring['cohorts'][label]['output_sha256']
        summary=json.loads((directory/'strict_summary.json').read_text())
        rows=[json.loads(line) for line in (directory/'strict_results.jsonl').open()]
        assert len(rows)==summary['groups']['all']['count']==256
        assert len({(r['split'],r['index']) for r in rows})==256
        assert summary['contract_pass'] and summary['groups']['all']['responses_with_tools']==256
        assert all(r['response_tokens']<=4096 and r['assistant_tokens']<=4096 for r in rows)
        assert all(t['generated_tokens']<=t['max_new_tokens'] for r in rows for t in r['turns'])
        assert summary['groups']['all']['error_then_success']>=1
        assert summary['format_99_percent_target_met'] and summary['execution_98_percent_target_met']
        cohort[label]=summary['groups']
    capacity=root/'capacity'
    scope=json.loads((capacity/'actor.scope.json').read_text())
    assert scope['world_size']==8 and scope['cp']==2 and scope['response_loss_tokens']==4096
    assert scope['prompt_tokens']>=manifest['splits']['train']['prompt_tokens']['max']
    trials=[]
    for rank in range(8):
        r=json.loads((capacity/f'actor_rank{rank}.json').read_text())
        assert r['sp']==2 and r['linear_arithmetic']=='ieee' and r['gather_global_key_projection']
        trial=r['reports'][0]
        assert trial['length']>=scope['actual_sequence_tokens'] and len(trial['passes'])==2
        assert all(p['grad_norm'] is None or math.isfinite(p['grad_norm']) for p in trial['passes'])
        trials.extend(trial['passes'])
    smoke=root/'strict_smoke'
    status=json.loads((smoke/'status.json').read_text())
    assert status['exit_code']==0
    for path,digest in json.loads((smoke/'source_sha256.json').read_text()).items():
        assert file_hash(REPO/path)==digest,path
    metrics=[json.loads(line) for line in (smoke/'metrics.jsonl').open()]
    final=next(r['data'] for r in metrics if r['step']==1)
    assert math.isfinite(final['actor/grad_norm'])
    expected=status['expected_trajectories']
    assert final['reward/trajectory_count']==expected
    assert final['tools/actual_executions']>0 and final['tools/responses_with_calls']==expected
    traces=[json.loads(line) for line in (smoke/'rollouts/1.tools.jsonl').open()]
    assert len(traces)==expected
    assert all(t['tool_events'] and t['generation_trace'] for t in traces)
    assert sum(t['metrics']['tool_execution_count'] for t in traces)==final['tools/actual_executions']
    old_math={label:numerical('stable_ieee_'+label) for label in ['short','real','long','real_repeat']}
    ready={'validation_status':'pass','data_profile':'gemma4_native_v1',
           'data_version':manifest['dataset_version'],'data_validation':report,
           'fixed_sample':{'distinct_prompts':256,'total_trajectories':512,'groups':cohort},
           'response_budget':4096,'rollout_context':34816,
           'actor_capacity':scope|{'peak_allocated_gib':max(t['peak_allocated_gib'] for t in trials),
                                   'peak_reserved_gib':max(t['peak_reserved_gib'] for t in trials)},
           'unchanged_16gpu_actor_numerical_evidence':old_math,
           'single_node_native_dapo_smoke':status|{'completed_updates':1,'accepted_trajectories':expected,
               'actual_tools':final['tools/actual_executions'],'grad_norm':final['actor/grad_norm']},
           'ready_for_16gpu_native_smoke':True,'full_16gpu_native_training_revalidated':False,
           'remaining_scope':'Full two-node native-profile smoke/eval and launch gates when the active job releases its GPUs',
           'manifest_sha256':file_hash(data/'manifest.json'),
           'sample_sha256':file_hash(data/'fixed_validation_sample.json'),
           'source_sha256':json.loads((smoke/'source_sha256.json').read_text())}
    (root/'readiness.json').write_text(json.dumps(ready,indent=2)+'\n')
    return ready


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence-dir',default='outputs/h200_checks/native_v1')
    result=validate(p.parse_args().evidence_dir)
    print(json.dumps({k:v for k,v in result.items() if k not in ['fixed_sample','source_sha256']},indent=2))
