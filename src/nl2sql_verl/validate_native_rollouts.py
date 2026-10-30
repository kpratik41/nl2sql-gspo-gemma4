"""Fixed-sample native-tool validation on an isolated vLLM TP4 engine."""
import argparse
import asyncio
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

from .prepare_reference_data import file_hash
from .tests.h200_tool_trace import diagnostic_hf_config


async def validate(args):
    import pyarrow.parquet as pq
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
    from verl.experimental.agent_loop.agent_loop import DictConfigWrap, ToolListWrap
    from verl.tools.tool_registry import initialize_tools_from_config
    from verl.utils.dataset.rl_dataset import RLHFDataset
    from verl.workers.rollout.replica import TokenOutput
    from .native_agent_loop import NL2SQLNativeAgentLoop, METRICS
    from .reward_entry_native import compute_score

    root=Path(args.output_dir)
    root.mkdir(parents=True,exist_ok=False)
    data=Path(args.data_dir)
    sample_path=data/'fixed_validation_sample.json'
    selection=json.loads(sample_path.read_text())
    tables={split:pq.read_table(data/f'{split}.parquet').to_pylist() for split in ['train','validation']}
    for split,digest in selection['data_sha256'].items():
        assert file_hash(data/f'{split}.parquet')==digest
    samples=selection['samples']
    assert len(samples)==256
    os.environ['NL2SQL_DATABASE_DIR']=args.database_dir
    os.environ['NL2SQL_LENGTH_MAX']='4096'
    os.environ['NL2SQL_LENGTH_BUFFER']='512'
    tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    engine=AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
        model=args.model,tokenizer=args.model,dtype='bfloat16',tensor_parallel_size=4,
        distributed_executor_backend='mp',max_model_len=34816,max_num_seqs=16,
        max_num_batched_tokens=34816,gpu_memory_utilization=.40,enforce_eager=True,
        enable_prefix_caching=True,hf_overrides=diagnostic_hf_config,language_model_only=True,
        disable_log_stats=True,seed=20261029))
    tools=initialize_tools_from_config('src/nl2sql_verl/configs/tools_native.yaml')
    cfg=OmegaConf.create({'actor_rollout_ref':{'rollout':{
        'full_determinism':False,'prompt_length':30720,'response_length':4096,
        'multi_turn':{'max_user_turns':None,'max_assistant_turns':9,'max_parallel_calls':3,
                      'max_tool_response_length':512,'tool_response_truncate_side':'right','format':'nl2sql_gemma4'}}}})
    metadata={'model':args.model,'model_label':args.label,'sample_sha256':file_hash(sample_path),
              'context':34816,'response_budget':4096,'temperature':1.2,'top_p':.95,'tensor_parallel':4,
              'sampling_seed':20261029,'samples':len(samples),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
              'source_sha256':{str(p):file_hash(p) for p in [Path(__file__),
                  Path('src/nl2sql_verl/native_agent_loop.py'),Path('src/nl2sql_verl/native_tools.py'),
                  Path('src/nl2sql_verl/gemma_native_profile.py'),Path('src/nl2sql_verl/reward_entry_native.py'),
                  Path('src/nl2sql_verl/configs/tools_native.yaml')]}}
    (root/'provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
    results=[]
    semaphore=asyncio.Semaphore(args.concurrency)
    started=time.monotonic()

    class Manager:
        def __init__(self,seed):
            self.seed,self.turns,self.prompt_max=seed,[],0

        async def generate(self,request_id,prompt_ids,sampling_params,**kwargs):
            maximum=min(sampling_params.get('max_tokens',4096),34816-len(prompt_ids))
            assert maximum>0
            self.prompt_max=max(self.prompt_max,len(prompt_ids))
            params=SamplingParams(temperature=sampling_params['temperature'],top_p=sampling_params['top_p'],top_k=-1,
                max_tokens=maximum,stop_token_ids=sampling_params.get('stop_token_ids'),skip_special_tokens=False,
                seed=self.seed+len(self.turns))
            final=None
            async for output in engine.generate({'prompt_token_ids':prompt_ids},params,request_id+'-'+uuid.uuid4().hex):
                final=output
            output=final.outputs[0]
            ids=list(output.token_ids)
            assert len(ids)<=maximum
            self.turns.append({'prompt_tokens':len(prompt_ids),'max_new_tokens':maximum,'generated_tokens':len(ids),
                              'finish_reason':output.finish_reason,'stop_reason':output.stop_reason,
                              'raw_text':tok.decode(ids,skip_special_tokens=False),'token_ids':ids})
            return TokenOutput(token_ids=ids)

    async def run_one(number,item):
        async with semaphore:
            row=tables[item['split']][item['index']]
            assert hashlib.sha256(json.dumps(row['prompt'],ensure_ascii=False).encode()).hexdigest()==item['prompt_sha256']
            manager=Manager(20261029+number*17)
            agent=NL2SQLNativeAgentLoop(trainer_config=DictConfigWrap(cfg),server_manager=manager,tokenizer=tok,
                processor=None,dataset_cls=RLHFDataset,data_config=DictConfigWrap(OmegaConf.create({})),
                hf_model_type='gemma4',tools=ToolListWrap(tools))
            output=await agent.run({'temperature':1.2,'top_p':.95},raw_prompt=row['prompt'],
                extra_info=row['extra_info'],tools_kwargs=row['extra_info']['tools_kwargs'])
            fields=output.extra_fields
            score=await asyncio.to_thread(compute_score,'nl2sql_bird','',row['reward_model']['ground_truth'],fields)
            result=dict(item,number=number,model_label=args.label,scores=score,
                        metrics={k:fields[k] for k in METRICS},response_tokens=len(output.response_ids),
                        assistant_tokens=fields['generated_token_count'],masked_context_tokens=len(output.response_mask)-sum(output.response_mask),
                        maximum_prompt_tokens=manager.prompt_max,turns=manager.turns,
                        tool_events=fields.get('native_tool_events',[]),assistant_text=fields['assistant_text'])
            results.append(result)
            with (root/'results.jsonl').open('a') as stream:
                stream.write(json.dumps(result)+'\n')
            (root/'heartbeat.json').write_text(json.dumps({'completed':len(results),'target':len(samples),
                'elapsed_seconds':time.monotonic()-started})+'\n')
            print(json.dumps({'completed':len(results),'split':item['split'],'index':item['index'],
                'format':score['format'],'execution':score['execution'],'result':score['result'],
                'tools':fields['tool_execution_count'],'tool_errors':fields['tool_error_count'],
                'verified':fields['final_sql_verified_exact'],'capped':fields['response_budget_reached']}),flush=True)

    try:
        await asyncio.gather(*(run_one(number,item) for number,item in enumerate(samples)))
        groups={}
        for split in ['train','validation','all']:
            chosen=[r for r in results if split=='all' or r['split']==split]
            n=len(chosen)
            groups[split]={
                'count':n,'format_pass':sum(r['scores']['format'] for r in chosen),
                'execution_pass':sum(r['scores']['execution'] for r in chosen),'result_pass':sum(r['scores']['result'] for r in chosen),
                'actual_tool_executions':sum(r['metrics']['tool_execution_count'] for r in chosen),
                'tool_errors':sum(r['metrics']['tool_error_count'] for r in chosen),
                'responses_with_tools':sum(r['metrics']['tool_execution_count']>0 for r in chosen),
                'responses_with_successful_sql_check':sum(r['metrics']['sqlite_query_success_count']>0 for r in chosen),
                'final_sql_verified_exact':sum(r['metrics']['final_sql_verified_exact'] for r in chosen),
                'error_then_success':sum(r['metrics']['tool_error_count']>0 and r['metrics']['final_sql_verified_exact']>0 for r in chosen),
                'format_valid_but_zero_actual_tools':sum(r['scores']['format']>0 and r['metrics']['tool_execution_count']==0 for r in chosen),
                'length_capped':sum(r['metrics']['response_budget_reached'] for r in chosen),
                'max_prompt_tokens':max(r['maximum_prompt_tokens'] for r in chosen),
                'max_response_tokens':max(r['response_tokens'] for r in chosen),
            }
            groups[split]['format_percent']=100*groups[split]['format_pass']/n
            groups[split]['execution_percent']=100*groups[split]['execution_pass']/n
        summary={'status':'completed','model_label':args.label,'elapsed_seconds':time.monotonic()-started,
                 'groups':groups,'contract_pass':groups['all']['format_valid_but_zero_actual_tools']==0,
                 'format_99_percent_target_met':groups['all']['format_percent']>=99,
                 'execution_98_percent_target_met':groups['all']['execution_percent']>=98}
        (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(summary,indent=2),flush=True)
    finally:
        engine.shutdown()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True)
    p.add_argument('--label',required=True)
    p.add_argument('--data-dir',default='outputs/gemma4_native_v1/data')
    p.add_argument('--database-dir',default='/opt/dlami/nvme/nl2sql/databases')
    p.add_argument('--output-dir',required=True)
    p.add_argument('--concurrency',type=int,default=16)
    asyncio.run(validate(p.parse_args()))
