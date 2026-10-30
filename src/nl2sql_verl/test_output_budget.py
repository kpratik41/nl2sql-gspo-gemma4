"""Inference-only paired-seed test of cumulative response budgets on hard prompts.

No Ray cluster, actor, optimizer, checkpoint writing, or training is started.
Run one process per budget on independent GPU sets, with the same saved cases.
"""
import argparse
import asyncio
from collections import Counter
import json
import os
from pathlib import Path
import time
import uuid

from .prepare_reference_data import file_hash
from .tests.h200_tool_trace import diagnostic_hf_config


async def run(args):
    from transformers import AutoTokenizer
    from omegaconf import OmegaConf
    from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
    from verl.experimental.agent_loop.agent_loop import DictConfigWrap, ToolListWrap
    from verl.tools.tool_registry import initialize_tools_from_config
    from verl.utils.dataset.rl_dataset import RLHFDataset
    from verl.workers.rollout.replica import TokenOutput
    from .native_agent_loop import NL2SQLNativeAgentLoop, METRICS
    from .reward_entry_native import compute_score

    root=Path(args.output_dir)
    root.mkdir(parents=True,exist_ok=False)
    cases=json.loads(Path(args.cases).read_text())
    assert len(cases)==2
    os.environ.update(NL2SQL_DATABASE_DIR=args.database_dir,NL2SQL_LENGTH_MAX=str(args.budget),NL2SQL_LENGTH_BUFFER='512')
    tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    # Same context/cache setup for both arms; only response budget and its
    # matching instruction differ. Short test prompts do not bind context.
    context=38912
    engine=AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
        model=args.model,tokenizer=args.model,dtype='bfloat16',tensor_parallel_size=4,
        distributed_executor_backend='mp',max_model_len=context,max_num_seqs=16,
        max_num_batched_tokens=context,gpu_memory_utilization=.35,enforce_eager=True,
        enable_prefix_caching=True,hf_overrides=diagnostic_hf_config,language_model_only=True,
        disable_log_stats=True,seed=20261029))
    cfg=OmegaConf.create({'actor_rollout_ref':{'rollout':{
        'full_determinism':False,'prompt_length':30720,'response_length':args.budget,
        'multi_turn':{'max_user_turns':None,'max_assistant_turns':9,'max_parallel_calls':3,
                      'max_tool_response_length':512,'tool_response_truncate_side':'right','format':'nl2sql_gemma4'}}}})
    tools=initialize_tools_from_config('src/nl2sql_verl/configs/tools_native.yaml')
    meta={'mode':'inference_only','model':args.model,'budget':args.budget,'context':context,
          'temperature':1.2,'top_p':.95,'max_assistant_turns':9,'generations_per_prompt':16,
          'cases_sha256':file_hash(args.cases),'seed_formula':'20261029 + case_index * 1000 + rollout_index * 17',
          'source_sha256':{str(p):file_hash(p) for p in [Path(__file__),Path('src/nl2sql_verl/native_agent_loop.py'),
              Path('src/nl2sql_verl/native_tools.py'),Path('src/nl2sql_verl/reward_entry_native.py')]},
          'caveat':'Matched seeds are not guaranteed identical token prefixes across independently scheduled stochastic batches.'}
    (root/'provenance.json').write_text(json.dumps(meta,indent=2)+'\n')
    results=[]
    started=time.monotonic()
    slots=asyncio.Semaphore(16)

    class Manager:
        def __init__(self,seed):
            self.seed,self.turns=seed,[]

        async def generate(self,request_id,prompt_ids,sampling_params,**kwargs):
            maximum=min(sampling_params['max_tokens'],context-len(prompt_ids))
            assert maximum>0
            sp=SamplingParams(temperature=sampling_params['temperature'],top_p=sampling_params['top_p'],top_k=-1,
                max_tokens=maximum,stop_token_ids=sampling_params.get('stop_token_ids'),skip_special_tokens=False,
                seed=self.seed+len(self.turns))
            last=None
            async for output in engine.generate({'prompt_token_ids':prompt_ids},sp,request_id+'-'+uuid.uuid4().hex):
                last=output
            out=last.outputs[0]
            ids=list(out.token_ids)
            assert len(ids)<=maximum
            self.turns.append({'prompt_tokens':len(prompt_ids),'max_new_tokens':maximum,'generated_tokens':len(ids),
                'finish_reason':out.finish_reason,'stop_reason':out.stop_reason,'token_ids':ids,
                'raw_text':tok.decode(ids,skip_special_tokens=False)})
            return TokenOutput(token_ids=ids)

    async def one(case_number,case,index):
        async with slots:
            seed=20261029+case_number*1000+index*17
            manager=Manager(seed)
            messages=[dict(m) for m in case['prompt']]
            old='4,096-token response budget'
            assert messages[0]['content'].count(old)==1
            messages[0]['content']=messages[0]['content'].replace(old,f'{args.budget:,}-token response budget')
            agent=NL2SQLNativeAgentLoop(trainer_config=DictConfigWrap(cfg),server_manager=manager,tokenizer=tok,
                processor=None,dataset_cls=RLHFDataset,data_config=DictConfigWrap(OmegaConf.create({})),
                hf_model_type='gemma4',tools=ToolListWrap(tools))
            output=await agent.run({'temperature':1.2,'top_p':.95},raw_prompt=messages,
                extra_info=case['extra_info'],tools_kwargs=case['extra_info']['tools_kwargs'])
            fields=output.extra_fields
            score=await asyncio.to_thread(compute_score,'nl2sql_bird','',case['ground_truth'],fields)
            row={'case':case['label'],'index':index,'seed':seed,'budget':args.budget,'scores':score,
                 'metrics':{k:fields[k] for k in METRICS},'response_tokens':len(output.response_ids),
                 'assistant_tokens':fields['generated_token_count'],'assistant_text':fields['assistant_text'],
                 'tool_events':fields.get('native_tool_events',[]),'turns':manager.turns,
                 'turn_limit_reached':fields['assistant_generation_count']>=9}
            assert row['response_tokens']<=args.budget
            results.append(row)
            with (root/'results.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n')
            progress={'completed':len(results),'total':32,'case':case['label'],'budget':args.budget,
                      'format':score['format'],'execution':score['execution'],'result':score['result'],
                      'capped':fields['response_budget_reached'],'turn_limit':row['turn_limit_reached'],
                      'tools':fields['tool_execution_count']}
            (root/'heartbeat.json').write_text(json.dumps(progress|{'elapsed_seconds':time.monotonic()-started})+'\n')
            print(json.dumps(progress),flush=True)

    try:
        await asyncio.gather(*(one(n,case,i) for n,case in enumerate(cases) for i in range(16)))
        groups={}
        for label in [c['label'] for c in cases]+['all']:
            selected=[r for r in results if label=='all' or r['case']==label]
            n=len(selected)
            groups[label]={'responses':n,'format_pass':sum(r['scores']['format'] for r in selected),
                'execution_pass':sum(r['scores']['execution'] for r in selected),'result_pass':sum(r['scores']['result'] for r in selected),
                'budget_caps':sum(r['metrics']['response_budget_reached'] for r in selected),
                'turn_limit_reached':sum(r['turn_limit_reached'] for r in selected),
                'format_failures_without_budget_cap':sum(not r['scores']['format'] and not r['metrics']['response_budget_reached'] for r in selected),
                'actual_tools':sum(r['metrics']['tool_execution_count'] for r in selected),
                'responses_with_tools':sum(r['metrics']['tool_execution_count']>0 for r in selected),
                'max_response_tokens':max(r['response_tokens'] for r in selected),
                'mean_response_tokens':sum(r['response_tokens'] for r in selected)/n}
        summary={'status':'completed','mode':'inference_only','budget':args.budget,'groups':groups,
                 'elapsed_seconds':time.monotonic()-started}
        (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(summary,indent=2),flush=True)
    finally:
        engine.shutdown()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True)
    p.add_argument('--cases',required=True)
    p.add_argument('--budget',type=int,choices=[4096,8192],required=True)
    p.add_argument('--database-dir',default='/opt/dlami/nvme/nl2sql/databases')
    p.add_argument('--output-dir',required=True)
    asyncio.run(run(p.parse_args()))
