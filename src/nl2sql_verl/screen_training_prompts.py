"""Inference-only 16-sample screening; retain mixed groups and 10% all-correct.

Positions in the saved train parquet are data_index; reference_index preserves
the original source ID (the parquet is already shuffled). No training is run.
"""
import argparse
import asyncio
from collections import Counter
import importlib.metadata
import json
import os
from pathlib import Path
import random
import socket
import time
import uuid

from .evaluate_full_dev import write_json, prompt_hash
from .prepare_reference_data import file_hash
from .tests.h200_tool_trace import diagnostic_hf_config


def select_indices(correct_counts, seed=20261029):
    """Sample exactly floor(N_all_correct / 10) from sorted all-correct IDs."""
    assert all(0 <= n <= 16 for n in correct_counts)
    mixed = [i for i, n in enumerate(correct_counts) if 0 < n < 16]
    perfect = [i for i, n in enumerate(correct_counts) if n == 16]
    easy = sorted(random.Random(seed).sample(perfect, len(perfect) // 10))
    return sorted(mixed + easy), mixed, easy


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

    assert args.temperature > 0 and 0 <= args.shard < 12
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows = pq.read_table(args.data).to_pylist()
    assert len(rows) == 6601
    assert sorted(r['extra_info']['reference_index'] for r in rows) == list(range(6601))
    assert all(r['extra_info']['tool_profile'] == 'gemma4_native_v1' for r in rows)
    assert not any('- Example 1 (db:' in m['content'] for r in rows for m in r['prompt'])
    selected = list(range(args.shard, len(rows), 12))
    sources = [Path(__file__), Path('src/nl2sql_verl/evaluate_full_dev.py'),
               Path('src/nl2sql_verl/native_agent_loop.py'), Path('src/nl2sql_verl/native_tools.py'),
               Path('src/nl2sql_verl/agent_loop.py'), Path('src/nl2sql_verl/task.py'),
               Path('src/nl2sql_verl/consensus/rewards.py'), Path('src/nl2sql_verl/tools.py'),
               Path('src/nl2sql_verl/vllm_compat.py'), Path('src/nl2sql_verl/gemma_arguments.py'),
               Path('src/nl2sql_verl/consensus/sql_utils.py'), Path('src/nl2sql_verl/configs/tools_native.yaml')]
    contract = {'mode': 'inference_only_train_screening', 'model': args.model,
                'model_revision': Path(args.model).name,
                'model_index_sha256': file_hash(Path(args.model) / 'model.safetensors.index.json'),
                'data': args.data, 'data_sha256': file_hash(args.data), 'data_rows': 6601,
                'shard': args.shard, 'shards': 12, 'data_indices': selected,
                'generations_per_prompt': 16, 'temperature': args.temperature, 'top_p': .95, 'top_k': -1,
                'response_budget': 8192, 'context': 38912, 'max_assistant_turns': 9,
                'tensor_parallel': 2, 'concurrency': 32, 'gpu_memory_utilization': .8,
                'max_num_batched_tokens': 8192, 'enforce_eager': True, 'prefix_caching': True,
                'correctness': 'native result reward: raw BIRD row-set equality; 60s SQL timeout; completed final required',
                'seed_formula': '20261029 + data_index * 100000 + rollout_index * 1000 + turn_index',
                'train_fewshots': 0,
                'versions': {p: importlib.metadata.version(p) for p in ['torch', 'vllm', 'transformers', 'ray']},
                'source_sha256': {str(p): file_hash(p) for p in sources}}
    path = root / 'provenance.json'
    if path.exists():
        assert json.loads(path.read_text())['contract'] == contract, 'Resume contract changed'
    else:
        write_json(path, {'contract': contract, 'host': socket.gethostname(),
                          'gpus': os.environ.get('CUDA_VISIBLE_DEVICES'), 'started_unix': time.time()})
    counts, group_n, group_correct, done = Counter(), Counter(), Counter(), set()
    maximum_response = 0

    def account(record):
        nonlocal maximum_response
        key = (record['data_index'], record['rollout_index'])
        assert key not in done and key[0] in selected and 0 <= key[1] < 16
        assert record['reference_index'] == rows[key[0]]['extra_info']['reference_index']
        assert record['scores']['result'] in (0., 1.)
        done.add(key)
        group_n[key[0]] += 1
        group_correct[key[0]] += int(record['scores']['result'])
        for metric in ['format', 'execution', 'result']:
            counts[metric + '_pass'] += record['scores'][metric]
        for metric in ['tool_execution_count', 'tool_error_count', 'response_budget_reached']:
            counts[metric] += record['metrics'][metric]
        counts['response_tokens'] += record['response_tokens']
        maximum_response = max(maximum_response, record['response_tokens'])

    result_path = root / 'results.jsonl'
    if result_path.exists():
        with result_path.open() as stream:
            for line in stream:
                account(json.loads(line))
    resumed = len(done)
    started = time.monotonic()
    last_heartbeat = 0.

    def heartbeat(state, **extra):
        nonlocal last_heartbeat
        hist = Counter(group_correct[i] for i in selected if group_n[i] == 16)
        last_heartbeat = time.monotonic()
        write_json(root / 'heartbeat.json', {'state': state, 'shard': args.shard,
            'completed': len(done), 'target': len(selected) * 16, 'prompts': len(selected),
            'groups_completed': sum(hist.values()), 'solve_count_histogram': dict(hist),
            'updated_unix': time.time(), 'elapsed_seconds': last_heartbeat - started,
            'resumed_rollouts': resumed, 'counts': dict(counts), 'max_response_tokens': maximum_response, **extra})

    heartbeat('initializing')
    if len(done) == len(selected) * 16:
        heartbeat('completed')
        return
    os.environ.update(NL2SQL_DATABASE_DIR=args.database_dir, NL2SQL_LENGTH_MAX='8192',
                      NL2SQL_LENGTH_BUFFER='512', NL2SQL_EXEC_TIMEOUT_S='60')
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
        model=args.model, tokenizer=args.model, dtype='bfloat16', tensor_parallel_size=2,
        distributed_executor_backend='mp', max_model_len=38912, max_num_seqs=32,
        max_num_batched_tokens=8192, enable_chunked_prefill=True, gpu_memory_utilization=.8,
        enforce_eager=True, enable_prefix_caching=True, hf_overrides=diagnostic_hf_config,
        language_model_only=True, disable_log_stats=True, seed=20261029))
    cfg = OmegaConf.create({'actor_rollout_ref': {'rollout': {
        'full_determinism': False, 'prompt_length': 30720, 'response_length': 8192,
        'multi_turn': {'max_user_turns': None, 'max_assistant_turns': 9, 'max_parallel_calls': 3,
                       'max_tool_response_length': 512, 'tool_response_truncate_side': 'right',
                       'format': 'nl2sql_gemma4'}}}})
    tools = initialize_tools_from_config('src/nl2sql_verl/configs/tools_native.yaml')
    slots = asyncio.Semaphore(32)

    class Manager:
        def __init__(self, seed):
            self.seed, self.turns = seed, []

        async def generate(self, request_id, prompt_ids, sampling_params, **kwargs):
            maximum = min(sampling_params['max_tokens'], 38912 - len(prompt_ids))
            assert maximum > 0 and sampling_params['temperature'] == args.temperature
            seed = self.seed + len(self.turns)
            sp = SamplingParams(temperature=args.temperature, top_p=.95, top_k=-1, n=1,
                max_tokens=maximum, stop_token_ids=sampling_params.get('stop_token_ids'),
                skip_special_tokens=False, seed=seed)
            last = None
            async for output in engine.generate({'prompt_token_ids': prompt_ids}, sp,
                                                request_id + '-' + uuid.uuid4().hex):
                last = output
            out = last.outputs[0]
            ids = list(out.token_ids)
            assert len(ids) <= maximum
            self.turns.append({'seed': seed, 'prompt_tokens': len(prompt_ids), 'max_new_tokens': maximum,
                               'generated_tokens': len(ids), 'finish_reason': out.finish_reason,
                               'stop_reason': out.stop_reason, 'token_ids': ids,
                               'raw_text': tok.decode(ids, skip_special_tokens=False)})
            return TokenOutput(token_ids=ids)

    async def one(index, sample):
        async with slots:
            row = rows[index]
            messages = [dict(m) for m in row['prompt']]
            assert messages[0]['content'].count('4,096-token response budget') == 1
            messages[0]['content'] = messages[0]['content'].replace('4,096-token response budget', '8,192-token response budget')
            seed = 20261029 + index * 100000 + sample * 1000
            manager = Manager(seed)
            agent = NL2SQLNativeAgentLoop(trainer_config=DictConfigWrap(cfg), server_manager=manager,
                tokenizer=tok, processor=None, dataset_cls=RLHFDataset,
                data_config=DictConfigWrap(OmegaConf.create({})), hf_model_type='gemma4', tools=ToolListWrap(tools))
            output = await agent.run({'temperature': args.temperature, 'top_p': .95}, raw_prompt=messages,
                extra_info=row['extra_info'], tools_kwargs=row['extra_info']['tools_kwargs'])
            fields = output.extra_fields
            score = await asyncio.to_thread(compute_score, 'nl2sql_bird', '',
                                           row['reward_model']['ground_truth'], fields)
            record = {'data_index': index, 'reference_index': row['extra_info']['reference_index'],
                'rollout_index': sample, 'seed': seed, 'db_id': row['extra_info']['db_id'], 'shard': args.shard,
                'source_prompt_sha256': prompt_hash(row['prompt']), 'effective_prompt_sha256': prompt_hash(messages),
                'scores': score, 'metrics': {k: fields[k] for k in METRICS},
                'response_tokens': len(output.response_ids), 'assistant_tokens': fields['generated_token_count'],
                'assistant_text': fields['assistant_text'], 'tool_events': fields.get('native_tool_events', []),
                'turns': manager.turns}
            assert record['response_tokens'] <= 8192
            assert record['metrics']['tool_execution_count'] == len(record['tool_events'])
            with result_path.open('a') as stream:
                stream.write(json.dumps(record) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            account(record)
            if len(done) % 16 == 0 or time.monotonic() - last_heartbeat >= 10:
                heartbeat('running', last_data_index=index)
                print(json.dumps({'completed': len(done), 'target': len(selected) * 16,
                                  'groups_completed': sum(n == 16 for n in group_n.values()),
                                  'correct_rollouts': counts['result_pass']}), flush=True)

    tasks = []
    try:
        heartbeat('running')
        tasks = [asyncio.create_task(one(i, j)) for i in selected for j in range(16) if (i, j) not in done]
        await asyncio.gather(*tasks)
        assert len(done) == len(selected) * 16
        heartbeat('completed')
    except BaseException as error:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        heartbeat('failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        engine.shutdown()


def select(args):
    import pyarrow as pa
    import pyarrow.parquet as pq
    root = Path(args.output_dir)
    table = pq.read_table(args.data)
    assert table.num_rows == 6601
    rows = table.to_pylist()
    done, totals, correct, contracts = set(), Counter(), Counter(), []
    for shard in range(12):
        directory = root / f'shard-{shard:02d}'
        assert json.loads((directory / 'heartbeat.json').read_text())['state'] == 'completed'
        contract = json.loads((directory / 'provenance.json').read_text())['contract']
        assert contract['data_sha256'] == file_hash(args.data)
        contracts.append(contract)
        with (directory / 'results.jsonl').open() as stream:
            for line in stream:
                row = json.loads(line)
                i, j = row['data_index'], row['rollout_index']
                assert 0 <= i < 6601 and 0 <= j < 16 and i % 12 == shard
                assert (i, j) not in done
                assert row['reference_index'] == rows[i]['extra_info']['reference_index']
                assert row['source_prompt_sha256'] == prompt_hash(rows[i]['prompt'])
                assert row['seed'] == 20261029 + i * 100000 + j * 1000
                assert row['scores']['result'] in (0., 1.)
                done.add((i, j));totals[i] += 1;correct[i] += int(row['scores']['result'])
    assert len(done) == 6601 * 16 and all(totals[i] == 16 for i in range(6601))
    for contract in contracts:
        assert all(contract[k] == v for k, v in contracts[0].items() if k not in {'shard', 'data_indices'})
    hits = [correct[i] for i in range(6601)]
    chosen, mixed, easy = select_indices(hits, args.selection_seed)
    assert chosen and not any(hits[i] == 0 for i in chosen)
    selected_table = table.take(pa.array(chosen, type=pa.int64()))
    output = root / 'selected_train.parquet'
    temporary = output.with_suffix('.tmp.parquet')
    pq.write_table(selected_table, temporary)
    temporary.replace(output)
    # The RL dataset contains original prompts/tools/gold, never screening answers.
    assert pq.read_table(output).equals(selected_table)
    chosen_set, mixed_set, easy_set = set(chosen), set(mixed), set(easy)
    with (root / 'group_scores.jsonl').open('w') as stream:
        for i in range(6601):
            stream.write(json.dumps({'data_index': i, 'reference_index': rows[i]['extra_info']['reference_index'],
                'correct_of_16': hits[i], 'selected': i in chosen_set,
                'reason': 'heterogeneous' if i in mixed_set else ('sampled_all_correct' if i in easy_set else 'excluded')}) + '\n')
    summary = {'status': 'completed', 'prompts': 6601, 'rollouts': len(done),
        'solve_count_histogram': dict(Counter(hits)), 'pass_at_16_count': sum(n > 0 for n in hits),
        'pass_at_16_percent': 100 * sum(n > 0 for n in hits) / 6601,
        'sample_accuracy_percent': 100 * sum(hits) / (6601 * 16),
        'heterogeneous': len(mixed), 'all_incorrect': sum(n == 0 for n in hits),
        'all_correct': sum(n == 16 for n in hits), 'sampled_all_correct': len(easy),
        'selection_seed': args.selection_seed, 'selection_rule': 'all 1..15/16 + floor(0.1 * all_16/16), seeded uniform sample',
        'selected_prompts': len(chosen), 'selected_data_indices': chosen,
        'source_data_sha256': file_hash(args.data), 'selected_parquet_sha256': file_hash(output),
        'screening_contract': contracts[0]}
    write_json(root / 'selection.json', summary)
    write_json(root / 'training_overrides.json', [f'data.train_files={output.resolve()}',
                                               f'data.expected_train_rows={len(chosen)}'])
    print(json.dumps({k: v for k, v in summary.items() if k not in {'selected_data_indices', 'screening_contract'}}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--select', action='store_true')
    parser.add_argument('--model', default='/opt/dlami/nvme/nl2sql/hf/hub/models--google--gemma-4-31B-it/snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475')
    parser.add_argument('--data', default='outputs/gemma4_native_v1/data/train.parquet')
    parser.add_argument('--database-dir', default='/opt/dlami/nvme/nl2sql/databases')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--temperature', type=float, default=1.2)
    parser.add_argument('--selection-seed', type=int, default=20261029)
    arguments = parser.parse_args()
    if arguments.select:
        select(arguments)
    else:
        asyncio.run(run(arguments))
