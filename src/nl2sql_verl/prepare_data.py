"""Normalize existing parquet rows for verl; never generate questions or gold SQL."""
import argparse
import json
from pathlib import Path

from .consensus.prompts import SYSTEM_PROMPT_TEMPLATES
from .consensus.sql_utils import extract_sql
from .consensus.tool_calling import get_tool_definitions, tool_catalog_compact

TOOLS = get_tool_definitions(include_consensus=False)
TOOL_NAMES = [t['function']['name'] for t in TOOLS]


def normalize_record(row, index=0):
    messages = row.get('prompt')
    if messages is None:
        messages = row.get('messages')
    if hasattr(messages, 'tolist'):
        messages = messages.tolist()
    if isinstance(messages, str):
        messages = json.loads(messages)
    if not isinstance(messages, list):
        raise ValueError(f'row {index}: prompt/messages must be a chat-message list')
    extra = dict(row.get('extra_info') or {})
    truth = (row.get('reward_model') or {}).get('ground_truth')
    if isinstance(truth, str) and truth.startswith('{'):
        truth = json.loads(truth)
    truth = truth if isinstance(truth, dict) else {'gold_sql': truth}
    db_id = row.get('db_id') or extra.get('db_id') or truth.get('db_id')
    gold = row.get('gold_sql') or row.get('SQL') or row.get('sql') or truth.get('gold_sql')
    if not gold:
        gold = next((m.get('content') for m in messages if m.get('role') == 'assistant'), '')
    gold = extract_sql(gold)
    if not db_id or not gold:
        raise ValueError(f'row {index}: db_id and gold_sql are required reward metadata')
    # Match consensus: remove gold assistant answers from the generation prompt.
    prompt = [{'role': m['role'], 'content': m['content']} for m in messages
              if m.get('role') in {'system', 'user'}]
    if not any(m['role'] == 'user' for m in prompt):
        raise ValueError(f'row {index}: missing user message')
    if not any(m['role'] == 'system' for m in prompt):
        prompt.insert(0, {'role': 'system', 'content': SYSTEM_PROMPT_TEMPLATES.replace(
            '{TOOL_CATALOG_COMPACT}', tool_catalog_compact()).strip()})
    supplied = row.get('tools')
    if isinstance(supplied, str):
        supplied = json.loads(supplied)
    if supplied is not None:
        names = {t['function']['name'] for t in supplied}
        if names != set(TOOL_NAMES):
            raise ValueError(f'row {index}: expected active consensus tools {TOOL_NAMES}; got {names}')
    extra.update(index=index, db_id=str(db_id), tool_selection=TOOL_NAMES,
                 tools_kwargs={name: {'create_kwargs': {'db_id': str(db_id)}} for name in TOOL_NAMES})
    return {'data_source': 'nl2sql_bird', 'prompt': prompt, 'ability': 'nl2sql',
            'agent_name': 'nl2sql_tool_agent',
            'reward_model': {'style': 'rule', 'ground_truth': json.dumps(
                {'db_id': str(db_id), 'gold_sql': gold}, ensure_ascii=False)},
            'extra_info': extra}


def prepare(source, destination, tokenizer=None, max_prompt_length=None):
    import pyarrow as pa
    import pyarrow.parquet as pq
    accepted, dropped = [], 0
    for batch in pq.ParquetFile(source).iter_batches(batch_size=128):
        for row in batch.to_pylist():
            result = normalize_record(row, len(accepted)+dropped)
            if tokenizer is not None and max_prompt_length is not None:
                ids = tokenizer.apply_chat_template(result['prompt'], tools=TOOLS,
                    tokenize=True, add_generation_prompt=True, return_dict=False)
                if len(ids) > max_prompt_length:
                    dropped += 1
                    continue
            accepted.append(result)
    if not accepted:
        raise ValueError(f'{source}: no usable rows remain')
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(accepted), destination)
    return {'input': str(source), 'output': str(destination), 'kept': len(accepted), 'overlength_dropped': dropped}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model')
    parser.add_argument('--max-prompt-length', type=int, default=2048)
    args = parser.parse_args()
    tokenizer = None
    if args.model:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(json.dumps(prepare(args.input, args.output, tokenizer, args.max_prompt_length)))


if __name__ == '__main__':
    main()
