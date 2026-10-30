"""Inference-only full-dev evaluation, partitioned across independent TP2 engines.

Each shard writes resumable, per-example results and an atomic heartbeat. The
original versioned parquet stays unchanged; the effective budget instruction is
recorded by hash. Invoke through worker_python.sh for the pinned CUDA runtime.
"""
import argparse
import asyncio
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import time
import uuid

from .prepare_reference_data import file_hash
from .tests.h200_tool_trace import diagnostic_hf_config


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def prompt_hash(messages):
    return hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode()).hexdigest()


def summarize(rows):
    n = len(rows)
    if not n:
        return {'count': 0}
    counts = {key + '_pass': sum(r['scores'][key] for r in rows)
              for key in ('format', 'execution', 'result')}
    return dict(count=n, **counts,
                **{key + '_percent': 100 * value / n for key, value in counts.items()},
                actual_tool_executions=sum(r['metrics']['tool_execution_count'] for r in rows),
                tool_errors=sum(r['metrics']['tool_error_count'] for r in rows),
                responses_with_tools=sum(r['metrics']['tool_execution_count'] > 0 for r in rows),
                final_sql_verified_exact=sum(r['metrics']['final_sql_verified_exact'] for r in rows),
                budget_caps=sum(r['metrics']['response_budget_reached'] for r in rows),
                turn_limit_reached=sum(r['turn_limit_reached'] for r in rows),
                mean_tool_calls=sum(r['metrics']['tool_execution_count'] for r in rows) / n,
                mean_response_tokens=sum(r['response_tokens'] for r in rows) / n,
                max_response_tokens=max(r['response_tokens'] for r in rows),
                max_assistant_tokens=max(r['assistant_tokens'] for r in rows),
                max_prompt_tokens=max(r['maximum_prompt_tokens'] for r in rows))


async def run(args):
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

    assert 0 <= args.shard < args.shards
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows = pq.read_table(args.data).to_pylist()
    assert len(rows) == 1534
    assert [r['extra_info']['reference_index'] for r in rows] == list(range(1534))
    assert all(r['extra_info']['tool_profile'] == 'gemma4_native_v1' for r in rows)
    fewshots = Counter(sum(m['content'].count(f'- Example {i} (db:')
                          for m in r['prompt'] for i in (1, 2, 3)) for r in rows)
    assert fewshots == {3: 1534}, fewshots
    selected = [i for i in range(len(rows)) if i % args.shards == args.shard]
    sources = [Path(__file__), Path('src/nl2sql_verl/native_agent_loop.py'),
               Path('src/nl2sql_verl/native_tools.py'), Path('src/nl2sql_verl/agent_loop.py'),
               Path('src/nl2sql_verl/task.py'), Path('src/nl2sql_verl/consensus/rewards.py'),
               Path('src/nl2sql_verl/tools.py'), Path('src/nl2sql_verl/vllm_compat.py'),
               Path('src/nl2sql_verl/gemma_arguments.py'), Path('src/nl2sql_verl/consensus/sql_utils.py'),
               Path('src/nl2sql_verl/gemma_native_profile.py'),
               Path('src/nl2sql_verl/reward_entry_native.py'),
               Path('src/nl2sql_verl/configs/tools_native.yaml')]
    contract = {'mode': 'inference_only', 'model': args.model,
                'model_revision': Path(args.model).name,
                'model_index_sha256': file_hash(Path(args.model) / 'model.safetensors.index.json'),
                'data_sha256': file_hash(args.data), 'data_rows': len(rows),
                'shard': args.shard, 'shards': args.shards, 'indices': selected,
                'temperature': 0., 'top_p': 1., 'top_k': -1, 'generations_per_prompt': 1,
                'tensor_parallel': 2, 'response_budget': args.budget,
                'context': 30720 + args.budget, 'max_assistant_turns': 9,
                'concurrency': args.concurrency, 'gpu_memory_utilization': args.gpu_memory_utilization,
                'max_num_batched_tokens': 8192, 'enforce_eager': True,
                'prefix_caching': True, 'fewshot_counts': {str(k): v for k, v in fewshots.items()},
                'versions': {name: importlib.metadata.version(name)
                             for name in ['torch', 'vllm', 'transformers', 'ray']},
                'source_sha256': {str(p): file_hash(p) for p in sources}}
    provenance_path = root / 'provenance.json'
    if provenance_path.exists():
        assert json.loads(provenance_path.read_text())['contract'] == contract, 'Resume contract changed'
    else:
        write_json(provenance_path, {'contract': contract, 'host': socket.gethostname(),
                                    'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                                    'started_unix': time.time()})
    results_path = root / 'results.jsonl'
    results = [json.loads(line) for line in results_path.open()] if results_path.exists() else []
    done = {r['reference_index'] for r in results}
    assert len(done) == len(results) and done <= set(selected)
    pending = [i for i in selected if i not in done]
    started = time.monotonic()

    def heartbeat(state, **extra):
        write_json(root / 'heartbeat.json', dict(state=state, shard=args.shard,
                   host=socket.gethostname(), completed=len(results), target=len(selected),
                   updated_unix=time.time(), elapsed_seconds=time.monotonic() - started,
                   summary=summarize(results), **extra))

    heartbeat('initializing')
    if not pending:
        heartbeat('completed')
        return
    os.environ.update(NL2SQL_DATABASE_DIR=args.database_dir,
                      NL2SQL_LENGTH_MAX=str(args.budget), NL2SQL_LENGTH_BUFFER='512',
                      NL2SQL_EXEC_TIMEOUT_S='60')
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    context = contract['context']
    engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
        model=args.model, tokenizer=args.model, dtype='bfloat16', tensor_parallel_size=2,
        distributed_executor_backend='mp', max_model_len=context,
        max_num_seqs=args.concurrency, max_num_batched_tokens=8192,
        enable_chunked_prefill=True, gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True, enable_prefix_caching=True, hf_overrides=diagnostic_hf_config,
        language_model_only=True, disable_log_stats=True, seed=20261029))
    cfg = OmegaConf.create({'actor_rollout_ref': {'rollout': {
        'full_determinism': False, 'prompt_length': 30720, 'response_length': args.budget,
        'multi_turn': {'max_user_turns': None, 'max_assistant_turns': 9, 'max_parallel_calls': 3,
                       'max_tool_response_length': 512, 'tool_response_truncate_side': 'right',
                       'format': 'nl2sql_gemma4'}}}})
    tools = initialize_tools_from_config('src/nl2sql_verl/configs/tools_native.yaml')
    slots = asyncio.Semaphore(args.concurrency)

    class Manager:
        def __init__(self):
            self.turns = []

        async def generate(self, request_id, prompt_ids, sampling_params, **kwargs):
            maximum = min(sampling_params['max_tokens'], context - len(prompt_ids))
            assert maximum > 0
            assert sampling_params['temperature'] == 0.
            sp = SamplingParams(temperature=0., top_p=1., top_k=-1, n=1,
                                max_tokens=maximum, stop_token_ids=sampling_params.get('stop_token_ids'),
                                skip_special_tokens=False, seed=20261029)
            last = None
            async for output in engine.generate({'prompt_token_ids': prompt_ids}, sp,
                                                request_id + '-' + uuid.uuid4().hex):
                last = output
            out = last.outputs[0]
            ids = list(out.token_ids)
            assert len(ids) <= maximum
            self.turns.append({'prompt_tokens': len(prompt_ids), 'max_new_tokens': maximum,
                               'generated_tokens': len(ids), 'finish_reason': out.finish_reason,
                               'stop_reason': out.stop_reason, 'token_ids': ids,
                               'raw_text': tok.decode(ids, skip_special_tokens=False)})
            return TokenOutput(token_ids=ids)

    async def one(index):
        async with slots:
            row = rows[index]
            messages = [dict(m) for m in row['prompt']]
            old = '4,096-token response budget'
            assert messages[0]['content'].count(old) == 1
            messages[0]['content'] = messages[0]['content'].replace(old, f'{args.budget:,}-token response budget')
            manager = Manager()
            agent = NL2SQLNativeAgentLoop(trainer_config=DictConfigWrap(cfg), server_manager=manager,
                tokenizer=tok, processor=None, dataset_cls=RLHFDataset,
                data_config=DictConfigWrap(OmegaConf.create({})), hf_model_type='gemma4', tools=ToolListWrap(tools))
            output = await agent.run({'temperature': 0., 'top_p': 1.}, raw_prompt=messages,
                extra_info=row['extra_info'], tools_kwargs=row['extra_info']['tools_kwargs'])
            fields = output.extra_fields
            score = await asyncio.to_thread(compute_score, 'nl2sql_bird', '',
                                           row['reward_model']['ground_truth'], fields)
            result = {'reference_index': index, 'db_id': row['extra_info']['db_id'], 'shard': args.shard,
                      'source_prompt_sha256': prompt_hash(row['prompt']),
                      'effective_prompt_sha256': prompt_hash(messages), 'scores': score,
                      'metrics': {k: fields[k] for k in METRICS}, 'response_tokens': len(output.response_ids),
                      'assistant_tokens': fields['generated_token_count'],
                      'masked_context_tokens': len(output.response_mask) - sum(output.response_mask),
                      'maximum_prompt_tokens': max(t['prompt_tokens'] for t in manager.turns),
                      'assistant_text': fields['assistant_text'], 'tool_events': fields.get('native_tool_events', []),
                      'turns': manager.turns, 'turn_limit_reached': fields['assistant_generation_count'] >= 9}
            assert result['response_tokens'] <= args.budget
            assert result['metrics']['tool_execution_count'] == len(result['tool_events'])
            with results_path.open('a') as stream:
                stream.write(json.dumps(result) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            results.append(result)
            heartbeat('running', last_index=index)
            print(json.dumps({'completed': len(results), 'target': len(selected), 'index': index,
                              'format': score['format'], 'execution': score['execution'],
                              'result': score['result'], 'tools': fields['tool_execution_count'],
                              'capped': fields['response_budget_reached']}), flush=True)

    tasks = []
    try:
        heartbeat('running')
        tasks = [asyncio.create_task(one(i)) for i in pending]
        await asyncio.gather(*tasks)
        assert len(results) == len(selected)
        write_json(root / 'summary.json', {'status': 'completed', 'summary': summarize(results)})
        heartbeat('completed')
    except BaseException as error:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        heartbeat('failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        engine.shutdown()


def merge(args):
    root = Path(args.output_dir)
    records, provenance = [], []
    for shard in range(args.shards):
        directory = root / f'shard-{shard:02d}'
        assert json.loads((directory / 'heartbeat.json').read_text())['state'] == 'completed'
        provenance.append(json.loads((directory / 'provenance.json').read_text()))
        records.extend(json.loads(line) for line in (directory / 'results.jsonl').open())
    records.sort(key=lambda row: row['reference_index'])
    assert [r['reference_index'] for r in records] == list(range(1534)), 'Missing or duplicate rows'
    first = provenance[0]['contract']
    for item in provenance:
        contract = item['contract']
        assert all(contract[k] == v for k, v in first.items() if k not in {'shard', 'indices'})
    with (root / 'results.jsonl').open('w') as stream:
        for row in records:
            stream.write(json.dumps(row) + '\n')
    summary = {'status': 'completed', 'mode': 'inference_only', 'contract': first,
               'all': summarize(records),
               'by_database': {db: summarize([r for r in records if r['db_id'] == db])
                               for db in sorted({r['db_id'] for r in records})},
               'results_sha256': file_hash(root / 'results.jsonl')}
    write_json(root / 'summary.json', summary)
    print(json.dumps(summary['all'], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--merge', action='store_true')
    parser.add_argument('--model', default='/opt/dlami/nvme/nl2sql/hf/hub/models--google--gemma-4-31B-it/snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475')
    parser.add_argument('--data', default='outputs/gemma4_native_v1/data/dev_full.parquet')
    parser.add_argument('--database-dir', default='/opt/dlami/nvme/nl2sql/databases')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=12)
    parser.add_argument('--budget', type=int, default=8192)
    parser.add_argument('--concurrency', type=int, default=16)
    parser.add_argument('--gpu-memory-utilization', type=float, default=.8)
    arguments = parser.parse_args()
    if arguments.merge:
        merge(arguments)
    else:
        asyncio.run(run(arguments))
