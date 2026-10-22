import argparse
import asyncio
import hashlib
import json
import re
import time
import urllib.request
from pathlib import Path

from openai import AsyncOpenAI

from .database import Database, atomic_json, json_default
from .tools import ToolRunner

TEMPLATE_KWARGS = {'enable_thinking': True, 'force_nonempty_content': True, 'truncate_history_thinking': False}


def token_count(base_url, model, messages, tools):
    url = base_url.rstrip('/').removesuffix('/v1') + '/tokenize'
    payload = {'model': model, 'messages': messages, 'tools': tools, 'chat_template_kwargs': TEMPLATE_KWARGS}
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=60) as response:
        result = json.load(response)
    return result['count'], result['max_model_len']


def extract_sql(content):
    matches = re.findall(r'<final_answer>\s*<sql_code>(.*?)</sql_code>\s*</final_answer>', content or '', re.S)
    return matches[0].strip() if len(matches) == 1 and matches[0].strip() else None


async def infer(client, base_url, shard, record, args, database):
    messages = [dict(m) for m in record['messages']]
    runner = ToolRunner(database, record['db_id'], stats_profile=record.get('stats_profile', 'baseline'))
    turns, prediction, status = [], None, 'turn_limit'
    start = time.monotonic()
    try:
        for turn in range(args.max_tool_rounds + 1):
            final_turn = turn == args.max_tool_rounds
            if final_turn:
                messages.append({'role': 'user', 'content': 'The tool-round budget is exhausted. Return your best final SQL now as <final_answer><sql_code>SQL</sql_code></final_answer>. No more tool calls.'})
            count, context_limit = await asyncio.to_thread(token_count, base_url, args.model, messages, record['tools'])
            if context_limit != args.max_model_len:
                raise RuntimeError(f'Server context is {context_limit}; expected {args.max_model_len}')
            if count + args.max_tokens > context_limit:
                status = 'filtered_context_length'
                turns.append({'turn': turn, 'prompt_tokens': count, 'reserved_output_tokens': args.max_tokens,
                              'context_limit': context_limit, 'error': 'Insufficient context; no history or output budget was truncated'})
                break
            response = await client.chat.completions.create(
                model=args.model, messages=messages, tools=record['tools'], tool_choice='none' if final_turn else 'auto',
                parallel_tool_calls=False, temperature=0, top_p=1, seed=0, max_tokens=args.max_tokens,
                extra_body={'chat_template_kwargs': TEMPLATE_KWARGS, 'truncate_prompt_tokens': None})
            choice = response.choices[0]
            message = choice.message.model_dump(exclude_none=True)
            turns.append({'turn': turn, 'prompt_tokens_preflight': count, 'finish_reason': choice.finish_reason, 'message': message,
                          'usage': response.usage.model_dump() if response.usage else None})
            # Preserve reasoning_content when supplied: the native template handles prior turns.
            history = {k: v for k, v in message.items() if k in ('role', 'content', 'tool_calls', 'reasoning_content', 'reasoning')}
            messages.append(history)
            if choice.finish_reason == 'length':
                status = 'token_limit'
                break
            calls = message.get('tool_calls') or []
            if calls:
                if final_turn:
                    status = 'tool_limit'
                    break
                for call in calls:
                    try:
                        if len(calls) != 1:
                            raise ValueError('Make at most one tool call per round. None of this round\'s calls were executed; retry with one call.')
                        arguments = json.loads(call['function']['arguments'])
                        result = await asyncio.to_thread(runner.call, call['function']['name'], arguments)
                    except Exception as exc:
                        result = {'error': type(exc).__name__, 'message': str(exc)}
                    tool_message = {'role': 'tool', 'tool_call_id': call['id'], 'name': call['function']['name'],
                                    'content': json.dumps(result, ensure_ascii=False, default=json_default)}
                    messages.append(tool_message)
                    turns[-1].setdefault('tool_results', []).append(tool_message)
                continue
            prediction = extract_sql(message.get('content'))
            if prediction:
                status = 'completed'
                break
            messages.append({'role': 'user', 'content': 'Continue using native tools if needed, or return <final_answer><sql_code>SQL</sql_code></final_answer>.'})
    except Exception as exc:
        status = 'request_error'
        turns.append({'error': type(exc).__name__, 'message': str(exc)})
    return {'question_id': record['question_id'], 'db_id': record['db_id'], 'question': record['question'],
            'prediction': prediction, 'status': status, 'elapsed_seconds': time.monotonic()-start,
            'temperature': 0, 'shard': shard, 'base_url': base_url,
            'turns': turns, 'messages': messages, 'tools': record['tools']}


async def main_async(args):
    all_records = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    records = all_records[:args.limit] if args.limit else all_records
    database = Database()
    for record in records:
        if database.fingerprint(record['db_id']) != record['schema_fingerprint']:
            raise RuntimeError('Database changed after prompt preparation; regenerate prompts')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = {k: v for k, v in vars(args).items() if k != 'resume'}
    config['input_sha256'] = hashlib.sha256(Path(args.input).read_bytes()).hexdigest()
    config['temperature'] = 0
    config['enable_thinking'] = True
    config['chat_template_kwargs'] = TEMPLATE_KWARGS
    config['pipeline_sha256'] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                 for name in ('run.py', 'tools.py', 'database.py')}
    manifest = output / 'config.json'
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        raise RuntimeError('Output directory belongs to a different run configuration')
    atomic_json(manifest, config)
    sems = [asyncio.Semaphore(args.concurrency) for _ in args.base_urls]
    clients = [AsyncOpenAI(base_url=url, api_key='local', timeout=600, max_retries=2) for url in args.base_urls]
    async def worker(index, record):
        shard = index % len(clients)
        path = output / f"{record['question_id']}.json"
        if path.exists():
            if args.resume:
                return
            raise FileExistsError(f'{path} exists; use --resume or a fresh output directory')
        async with sems[shard]:
            result = await infer(clients[shard], args.base_urls[shard], shard, record, args, database)
            atomic_json(path, result)
            print(f"{record['question_id']}: {result['status']} ({len(result['turns'])} turns, {result['elapsed_seconds']:.1f}s)", flush=True)
    try:
        for shard, client in enumerate(clients):
            models = await client.models.list()
            if args.model not in {m.id for m in models.data}:
                raise RuntimeError(f'Requested model is not served on shard {shard}: {args.model}')
            atomic_json(output / f'server_models_{shard}.json', models.model_dump())
            _, context_limit = await asyncio.to_thread(token_count, args.base_urls[shard], args.model, records[0]['messages'], records[0]['tools'])
            if context_limit != args.max_model_len:
                raise RuntimeError(f'Shard {shard} context is {context_limit}, expected {args.max_model_len}')
        await asyncio.gather(*(worker(i, record) for i, record in enumerate(records)))
    finally:
        await asyncio.gather(*(client.close() for client in clients))
    results = [json.loads((output / f"{r['question_id']}.json").read_text()) for r in records]
    filtered = [r for r in results if r['status'] == 'filtered_context_length']
    predictions = {str(r['question_id']): (r['prediction'] or '') + '\t----- bird -----\t' + r['db_id']
                   for r in results if r['status'] != 'filtered_context_length'}
    atomic_json(output / 'predict.json', predictions)
    atomic_json(output / 'filtered.json', [{'question_id': r['question_id'], 'db_id': r['db_id'],
                'reason': r['status'], 'details': r['turns'][-1]} for r in filtered])
    atomic_json(output / 'summary.json', {'samples': len(results), 'completed': sum(r['status'] == 'completed' for r in results),
                'filtered_context_length': len(filtered), 'eligible_for_evaluation': len(results)-len(filtered),
                'tool_calls': sum(len(t.get('message', {}).get('tool_calls', [])) for r in results for t in r['turns'])})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='data/prepared/california_schools.jsonl')
    p.add_argument('--output', default='outputs/california_schools_temp0')
    p.add_argument('--base-urls', nargs='+', default=['http://127.0.0.1:8000/v1', 'http://127.0.0.1:8001/v1'])
    p.add_argument('--model', default='nemotron-lightning-bf16')
    p.add_argument('--concurrency', type=int, default=8, help='Concurrent questions per server shard')
    p.add_argument('--max-tool-rounds', type=int, choices=[8], default=8)
    p.add_argument('--max-model-len', type=int, default=43000)
    p.add_argument('--max-tokens', type=int, default=8192)
    p.add_argument('--limit', type=int)
    p.add_argument('--resume', action='store_true')
    asyncio.run(main_async(p.parse_args()))


if __name__ == '__main__':
    main()
