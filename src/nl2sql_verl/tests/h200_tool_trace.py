"""Isolated node-2 vLLM/native-agent diagnostic, with raw-token and tool traces.

Uses a completed HF export, never joins the training Ray cluster, and never
modifies model weights, training configuration, or the active run.
"""
import argparse
import asyncio
from collections import Counter
import json
import os
from pathlib import Path
import uuid


def diagnostic_hf_config(config):
    from src.nl2sql_verl.vllm_compat import legacy_gemma4_config
    config = legacy_gemma4_config(config)
    # HF exports retain the actor's attention tag; vLLM implements text attention
    # itself, while any HF multimodal submodules must use an installed backend.
    for value in [config, getattr(config, 'text_config', None),
                  getattr(config, 'vision_config', None), getattr(config, 'audio_config', None)]:
        if value is not None:
            value._attn_implementation = 'sdpa'
    return config


async def probe(args):
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
    from verl.experimental.agent_loop.agent_loop import DictConfigWrap, ToolListWrap
    from verl.tools.tool_registry import initialize_tools_from_config
    from verl.utils.dataset.rl_dataset import RLHFDataset
    from verl.workers.rollout.replica import TokenOutput
    from src.nl2sql_verl.agent_loop import NL2SQLToolAgentLoop
    from src.nl2sql_verl.consensus.rewards import make_nl2sql_rewards
    from src.nl2sql_verl.consensus.sql_utils import bird_execute_sql, extract_sql

    os.environ['NL2SQL_DATABASE_DIR'] = args.database_dir
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    if (outdir/'results.jsonl').exists():
        raise FileExistsError('Use a fresh output directory to keep probe trajectories unambiguous')
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
        model=args.model, tokenizer=args.model, dtype='bfloat16',
        tensor_parallel_size=4, distributed_executor_backend='mp',
        max_model_len=32768, max_num_seqs=16, max_num_batched_tokens=32768,
        gpu_memory_utilization=0.35, enforce_eager=True,
        enable_prefix_caching=True, hf_overrides=diagnostic_hf_config, language_model_only=True,
        disable_log_stats=True, seed=20261029,
    ))
    inputs = json.loads(Path(args.inputs).read_text())
    tools = initialize_tools_from_config('src/nl2sql_verl/configs/tools.yaml')
    events = []
    for tool in tools:
        original = tool.execute

        async def traced_execute(instance_id, parameters, _original=original, _name=tool.name, **kwargs):
            response, reward, extra = await _original(instance_id, parameters, **kwargs)
            data = kwargs.get('agent_data')
            events.append({'request_id': data.request_id if data else None, 'name': _name,
                           'parameters': parameters, 'observation': response.text, 'extra': extra})
            return response, reward, extra
        tool.execute = traced_execute

    class Manager:
        def __init__(self, budget, seed):
            self.budget, self.seed, self.turns = budget, seed, []

        async def generate(self, request_id, prompt_ids, sampling_params, **kwargs):
            sp = SamplingParams(
                temperature=sampling_params['temperature'], top_p=sampling_params['top_p'], top_k=-1,
                max_tokens=min(self.budget, 32768-len(prompt_ids)),
                stop_token_ids=sampling_params.get('stop_token_ids'),
                skip_special_tokens=False, seed=self.seed+len(self.turns),
            )
            final = None
            async for output in engine.generate(
                {'prompt_token_ids': prompt_ids}, sp, request_id+'-'+uuid.uuid4().hex
            ):
                final = output
            output = final.outputs[0]
            ids = list(output.token_ids)
            self.turns.append({
                'prompt_length': len(prompt_ids), 'generated_length': len(ids),
                'finish_reason': output.finish_reason, 'stop_reason': output.stop_reason,
                'special_token_counts': dict(Counter(tok.convert_ids_to_tokens(t) for t in ids if t in tok.all_special_ids)),
                'token_ids': ids, 'raw_text': tok.decode(ids, skip_special_tokens=False),
            })
            return TokenOutput(token_ids=ids)

    results = []
    async def run_one(example, budget, protocol, seed):
        manager = Manager(budget, seed)
        cfg = OmegaConf.create({'actor_rollout_ref': {'rollout': {
            'full_determinism': False, 'prompt_length': 30720, 'response_length': budget,
            'multi_turn': {'max_user_turns': None, 'max_assistant_turns': 9,
                           'max_parallel_calls': 3, 'max_tool_response_length': 512,
                           'tool_response_truncate_side': 'right', 'format': 'nl2sql_gemma4'},
        }}})
        data = OmegaConf.create({'apply_chat_template_kwargs': {}})
        agent = NL2SQLToolAgentLoop(
            trainer_config=DictConfigWrap(cfg), server_manager=manager, tokenizer=tok,
            processor=None, dataset_cls=RLHFDataset, data_config=DictConfigWrap(data),
            hf_model_type='gemma4', tools=ToolListWrap(tools),
        )
        messages = [dict(m) for m in example['prompt']]
        if protocol == 'explicit_native':
            messages[0]['content'] += '''

NATIVE SERIALIZATION CLARIFICATION (takes precedence over earlier syntax examples):
Use the actual Gemma special tokens for every tool call:
<|tool_call>call:sqlite_query{db_id:<|"|>DATABASE_ID<|"|>,sql:<|"|>SELECT ...<|"|>,max_return_rows:10}<tool_call|>
Every string argument, including SQL, must use <|"|> delimiters. Integer and array arguments retain their native types.
Emit <tool_call|> immediately after the argument object. The runtime will stop generation and execute the tool.
Wait for the actual tool observation before describing results. A bare call:... expression does not invoke a tool.
'''
        output = await agent.run({'temperature': 1.2, 'top_p': 0.95},
                                 raw_prompt=messages, tools_kwargs=example['tools_kwargs'])
        text = output.extra_fields['assistant_text']
        sql = extract_sql(text)
        ok, _, err = await asyncio.to_thread(bird_execute_sql, sql, example['db_id'], args.database_dir, 60)
        fmt = make_nl2sql_rewards(args.database_dir)[0]([text])[0]
        result = {
            'example': example['label'], 'db_id': example['db_id'], 'budget': budget,
            'protocol': protocol, 'seed': seed, 'assistant_turns': len(manager.turns),
            'num_turns_metric': output.num_turns, 'format': fmt, 'execution': int(ok),
            'execution_error': err, 'generated_token_count': output.extra_fields['generated_token_count'],
            'response_length': len(output.response_ids), 'tool_reward_count': len(output.extra_fields['tool_rewards']),
            'turns': manager.turns, 'assistant_text': text, 'sql': sql,
        }
        results.append(result)
        with (outdir/'results.jsonl').open('a') as f:
            f.write(json.dumps(result)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k not in ['turns','assistant_text','sql']}), flush=True)

    try:
        # Baseline and two isolated counterfactuals, same checkpoint and seeds.
        for budget, protocol in [(2048, 'current'), (4096, 'current'), (4096, 'explicit_native')]:
            await asyncio.gather(*(run_one(ex, budget, protocol, seed)
                                   for ex in inputs for seed in [120, 121]))
        (outdir/'executed_tools.json').write_text(json.dumps(events, indent=2)+'\n')
        groups = {}
        for budget, protocol in [(2048, 'current'), (4096, 'current'), (4096, 'explicit_native')]:
            selected = [r for r in results if r['budget']==budget and r['protocol']==protocol]
            groups[f'{budget}_{protocol}'] = {
                'n': len(selected), 'format_pass': sum(r['format'] for r in selected),
                'execution_pass': sum(r['execution'] for r in selected),
                'assistant_generations': sum(r['assistant_turns'] for r in selected),
                'tool_reward_entries': sum(r['tool_reward_count'] for r in selected),
                'length_capped': sum(r['response_length']>=budget for r in selected),
                'parsed_special_tool_call_starts': sum(sum(t['special_token_counts'].get('<|tool_call>',0) for t in r['turns']) for r in selected),
            }
        (outdir/'summary.json').write_text(json.dumps(groups, indent=2)+'\n')
        print(json.dumps(groups), flush=True)
    finally:
        (outdir/'executed_tools.json').write_text(json.dumps(events, indent=2)+'\n')
        if hasattr(engine, 'shutdown'):
            engine.shutdown()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--database-dir', required=True)
    p.add_argument('--inputs', required=True)
    p.add_argument('--output-dir', required=True)
    args = p.parse_args()
    asyncio.run(probe(args))


if __name__ == '__main__':
    main()
