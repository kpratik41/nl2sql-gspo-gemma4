"""Future native-profile metrics and persistent traces for pinned verl V1."""
import json
import math
from pathlib import Path
import statistics

from . import fsdp_runtime  # noqa: F401; register unchanged actor/checkpoint hooks first
from .checkpoint_retention import atomic_write
from .native_agent_loop import METRICS


def records_from_extra(extra, keys, tags):
    values=extra.tolist() if hasattr(extra,'tolist') else list(extra)
    records=[]
    for fields,key,tag in zip(values,keys,tags,strict=True):
        if tag.get('is_padding',False):
            continue
        reward=fields.get('reward_extra_info',{})
        numeric={name:float(reward[name]) for name in METRICS}
        if not all(math.isfinite(v) for v in numeric.values()):
            raise ValueError('Nonfinite native tool telemetry')
        records.append({'uid':key, 'metrics':numeric,
                        'assistant_text':fields.get('assistant_text',''),
                        'tool_events':fields.get('native_tool_events',[]),
                        'generation_trace':fields.get('native_generation_trace',[])})
    return records


def register():
    from verl.trainer.ppo.v1.trainer_base import PPOTrainer, tq
    if getattr(PPOTrainer,'_nl2sql_native_runtime',False):
        return
    original_metrics=PPOTrainer._compute_metrics
    original_dump=PPOTrainer._log_rollout_data

    def fetch(batch):
        fields=tq.kv_batch_get(keys=batch.keys,partition_id=batch.partition_id,select_fields=['extra_fields'])['extra_fields']
        return records_from_extra(fields,batch.keys,batch.tags)

    def metrics(self,batch,metrics,timing_raw,global_steps,epoch):
        original_metrics(self,batch,metrics,timing_raw,global_steps,epoch)
        records=fetch(batch)
        for name in METRICS:
            values=[r['metrics'][name] for r in records]
            metrics[f'reward/{name}/mean']=statistics.mean(values)
            metrics[f'reward/{name}/std']=statistics.pstdev(values)
        metrics['tools/actual_executions']=sum(r['metrics']['tool_execution_count'] for r in records)
        metrics['tools/execution_errors']=sum(r['metrics']['tool_error_count'] for r in records)
        metrics['tools/responses_with_calls']=sum(r['metrics']['tool_execution_count']>0 for r in records)
        metrics['tools/final_sql_verified_fraction']=statistics.mean(r['metrics']['final_sql_verified_exact'] for r in records)

    def dump(self,batch,timing_raw,rollout_data_dir):
        original_dump(self,batch,timing_raw,rollout_data_dir)
        records=fetch(batch)
        def key(row):
            parts=row['uid'].rsplit('_',2)
            return (parts[0],int(parts[1]),int(parts[2])) if len(parts)==3 else (row['uid'],0,0)
        records.sort(key=key)
        root=Path(rollout_data_dir)
        root.mkdir(parents=True,exist_ok=True)
        atomic_write(root/f'{self.global_steps}.tools.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))

    PPOTrainer._compute_metrics=metrics
    PPOTrainer._log_rollout_data=dump
    PPOTrainer._nl2sql_native_runtime=True


register()
