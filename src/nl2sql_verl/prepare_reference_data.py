"""Build verl parquet from raw 6601 training rows and frozen prompt references.

This intentionally reuses the supplied schemas/few-shots, rather than rerunning
SQLite profiling or BM25 with possibly different results. No rows are filtered.
"""
import argparse
from collections import defaultdict, deque
import hashlib
import json
import math
from pathlib import Path
import random
import re

from .prepare_data import normalize_record
from .tool_profiles import load_tool_profile


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def identity(row):
    return tuple(str(row.get(key, '')).strip()
                 for key in ('db_id', 'question', 'evidence'))


def sql_tokens(sql):
    # Preserve quoted literals/identifiers. Only whitespace outside them and a
    # terminal semicolon may differ between raw SQL and the prepared reference.
    tokens = re.findall(
        r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|`(?:``|[^`])*`|"
        r"\[[^\]]*\]|[\w]+|[^\s]", sql)
    if tokens and tokens[-1] == ';':
        tokens.pop()
    return tokens


def reference_messages(row, profile):
    messages = row['prompt']
    if row.get('messages', messages) != messages:
        raise ValueError('Reference prompt and messages disagree')
    if [m['role'] for m in messages] != ['system', 'user']:
        raise ValueError('Reference must contain exactly system and user messages')
    if row['tools'] != profile.declarations():
        raise ValueError('Reference tool schemas differ from the active task contract')
    if profile.name == 'gemma4' and messages != profile.messages(messages[1]['content']):
        raise ValueError('Gemma reference system prompt changed')
    user = messages[1]['content']
    for tag, key in [('question', 'question'), ('hint', 'evidence'), ('db_id', 'db_id')]:
        match = re.search(fr'<{tag}>(.*?)</{tag}>', user, re.S)
        if match is None or match.group(1).strip() != str(row.get(key, '')).strip():
            raise ValueError(f'Reference metadata does not match <{tag}>')
    return profile.messages(user)


def match_training(raw_rows, reference_rows):
    references = defaultdict(deque)
    for index, row in enumerate(reference_rows):
        references[identity(row)].append((index, row))
    matched = []
    for raw in raw_rows:
        candidates = references[identity(raw)]
        if not candidates:
            raise ValueError(f'Raw training row has no reference match: {identity(raw)}')
        # Preserve duplicate examples, and match gold SQL as well as the question.
        candidate = next((item for item in candidates
                          if sql_tokens(raw['SQL']) == sql_tokens(item[1]['gold_sql'])), None)
        if candidate is None:
            raise ValueError(f'Gold SQL differs for {identity(raw)}')
        candidates.remove(candidate)
        matched.append((raw, *candidate))
    if any(references.values()):
        raise ValueError('Training reference has rows absent from the raw source')
    return matched


def build_record(reference, profile, index, reference_index, split, gold_sql=None):
    messages = reference_messages(reference, profile)
    row = dict(reference, prompt=messages,
               gold_sql=reference['gold_sql'] if gold_sql is None else gold_sql,
               extra_info={'question': reference['question'],
                           'evidence': reference['evidence'], 'split': split,
                           'reference_index': reference_index,
                           'tool_profile': profile.name})
    return normalize_record(row, index)


def summarize_lengths(lengths, limit):
    ordered = sorted(lengths)
    return {'min': ordered[0], 'median': ordered[len(ordered)//2],
            'max': ordered[-1], 'configured_limit': limit,
            'above_configured_limit': sum(n > limit for n in lengths),
            'covering_limit_multiple_1024': math.ceil(ordered[-1]/1024)*1024}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-source', default='data/raw/train-6601.jsonl')
    parser.add_argument('--train-reference', default='outputs/train-6601-schema-bare-tool.jsonl')
    parser.add_argument('--eval-reference', default='outputs/old-dev-schema-tool-unpatched.jsonl')
    parser.add_argument('--output-dir', default='outputs/cp1_recipe/data')
    parser.add_argument('--model', required=True, help='Locally cached tokenizer directory')
    parser.add_argument('--tool-profile', choices=['gemma4', 'custom'], default='gemma4')
    parser.add_argument('--system-prompt-file')
    parser.add_argument('--eval-size', type=int, default=640,
                        help='Training evaluation subset; 0 selects the entire dev set')
    parser.add_argument('--eval-seed', type=int, default=20261026)
    parser.add_argument('--max-prompt-length', type=int, default=11264,
                        help='Report incompatibility without truncating or dropping rows')
    args = parser.parse_args()
    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    profile = load_tool_profile(args.tool_profile, args.system_prompt_file)
    raw = read_jsonl(args.train_source)
    if len(raw) != 6601:
        raise ValueError(f'Expected all 6601 training rows, found {len(raw)}')
    train_reference = read_jsonl(args.train_reference)
    eval_reference = read_jsonl(args.eval_reference)
    matches = match_training(raw, train_reference)
    train = [build_record(ref, profile, i, ref_i, 'train', gold_sql=source['SQL'])
             for i, (source, ref_i, ref) in enumerate(matches)]
    dev = [build_record(ref, profile, i, i, 'dev') for i, ref in enumerate(eval_reference)]
    train_dbs = {r['extra_info']['db_id'] for r in train}
    dev_dbs = {r['extra_info']['db_id'] for r in dev}
    if train_dbs & dev_dbs:
        raise ValueError('Train and dev databases overlap')
    for db in train_dbs | dev_dbs:
        from .task import database_path
        database_path(db)
    if not 0 <= args.eval_size <= len(dev):
        raise ValueError('eval-size must be 0 (all) or a valid dev subset size')
    if args.eval_size and args.eval_size < len(dev):
        indices = list(range(len(dev)))
        random.Random(args.eval_seed).shuffle(indices)
        eval_indices = sorted(indices[:args.eval_size])
    else:
        eval_indices = list(range(len(dev)))

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    lengths = {}
    for split, rows in [('train', train), ('dev_full', dev)]:
        lengths[split] = []
        for i, row in enumerate(rows):
            lengths[split].append(profile.prompt_length(tokenizer, row['prompt']))
            if (i+1) % 1000 == 0:
                print(f'{split}: tokenized {i+1}/{len(rows)}', flush=True)
    lengths['validation'] = [lengths['dev_full'][i] for i in eval_indices]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for split in ('train', 'dev_full', 'validation'):
        if (output/f'{split}.parquet').exists():
            raise FileExistsError(f'{output}/{split}.parquet already exists; choose a new output-dir')
    manifest = {
        'training_source': args.train_source, 'raw_9428_used': False,
        'train_reference': args.train_reference, 'eval_reference': args.eval_reference,
        'input_sha256': {path: file_hash(path) for path in
                         (args.train_source, args.train_reference, args.eval_reference)},
        'tool_profile': profile.name, 'model': str(Path(args.model).resolve()),
        'tokenizer_sha256': {name: file_hash(Path(args.model)/name) for name in
                             ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja')
                             if (Path(args.model)/name).exists()},
        'source_sha256': {name: file_hash(Path(__file__).parent/name) for name in
                         ('prepare_reference_data.py', 'tool_profiles.py', 'prepare_data.py',
                          'consensus/prompts.py', 'consensus/tool_calling.py')},
        'length_filter': False, 'truncated_rows': 0,
        'train_order': 'raw 6601 source order; duplicates retained',
        'dev_selection': 'seeded sample from the supplied unpatched dev reference',
        'dev_seed': args.eval_seed, 'dev_reference_indices': eval_indices,
        'historical_640_subset_reproduced': False,
        'splits': {}}
    for split, rows in [('train', train), ('dev_full', dev),
                        ('validation', [dev[i] for i in eval_indices])]:
        path = output/f'{split}.parquet'
        temporary = path.with_suffix('.parquet.tmp')
        pq.write_table(pa.Table.from_pylist(rows), temporary)
        temporary.replace(path)
        (output/f'{split}_prompt_lengths.json').write_text(json.dumps(lengths[split])+'\n')
        manifest['splits'][split] = {
            'count': len(rows), 'database_count': len({r['extra_info']['db_id'] for r in rows}),
            'sha256': file_hash(path),
            'duplicate_prompt_rows_retained': len(rows)-len({json.dumps(r['prompt']) for r in rows}),
            'prompt_tokens': summarize_lengths(lengths[split], args.max_prompt_length)}
    manifest['fits_configured_prompt_limit'] = all(
        s['prompt_tokens']['above_configured_limit'] == 0 for s in manifest['splits'].values())
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
