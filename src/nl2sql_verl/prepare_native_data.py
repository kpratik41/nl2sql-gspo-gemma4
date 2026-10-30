"""Version existing BIRD parquets with native Gemma instructions and resolved FKs."""
import argparse
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import sqlite3

from .prepare_reference_data import file_hash, summarize_lengths
from .task import database_path
from .tool_profiles import load_tool_profile

VERSION = 'gemma4-native-v1'


def quote_identifier(name):
    return '"'+name.replace('"', '""')+'"'


@lru_cache(maxsize=128)
def implicit_foreign_keys(db_id, database_dir):
    path = database_path(db_id, database_dir)
    fixes, schema = {}, []
    with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as connection:
        for table, ddl in connection.execute("SELECT name,sql FROM sqlite_master WHERE type='table' ORDER BY name"):
            schema.append((table, ddl))
            groups = {}
            for row in connection.execute('PRAGMA foreign_key_list('+quote_identifier(table)+')'):
                groups.setdefault(row[0], []).append(row)
            for group in groups.values():
                group.sort(key=lambda r: r[1])
                if not any(r[4] is None for r in group):
                    continue
                parent = group[0][2]
                primary = sorted((r[5], r[1]) for r in connection.execute(
                    'PRAGMA table_info('+quote_identifier(parent)+')') if r[5])
                if len(primary) != len(group) or any(r[4] is not None for r in group):
                    raise ValueError(f'Cannot resolve implicit FK: {db_id}/{table}/{group}')
                for row, (_, target) in zip(group, primary, strict=True):
                    old = f'{table}.{row[3]}={parent}.None'
                    new = f'{table}.{row[3]}={parent}.{target}'
                    if old in fixes and fixes[old] != new:
                        raise ValueError(f'Ambiguous FK rendering: {old}')
                    fixes[old] = new
    digest = hashlib.sha256(json.dumps(schema, ensure_ascii=False).encode()).hexdigest()
    return fixes, digest


def corrected_user_prompt(user, db_id, database_dir):
    match = re.search(r'(<database_schema>)(.*?)(</database_schema>)', user, re.S)
    if match is None:
        raise ValueError('Missing database_schema block')
    schema = match[2]
    changes = []
    fixes, digest = implicit_foreign_keys(db_id, str(Path(database_dir).resolve()))
    for old, new in fixes.items():
        pattern = re.compile(r'(?<![\w.])'+re.escape(old)+r'(?!\w)')
        schema, count = pattern.subn(lambda _: new, schema)
        if count:
            changes.append({'old': old, 'new': new, 'count': count})
    if re.search(r'\b\w+\.None\b', schema):
        raise ValueError(f'Unresolved implicit FK in {db_id}')
    corrected = user[:match.start(2)]+schema+user[match.end(2):]
    return corrected, changes, digest


def transform(row, profile, database_dir):
    result = deepcopy(row)
    assert [m['role'] for m in row['prompt']] == ['system', 'user']
    db_id = row['extra_info']['db_id']
    user, changes, digest = corrected_user_prompt(row['prompt'][1]['content'], db_id, database_dir)
    result['prompt'] = profile.messages(user)
    result['agent_name'] = profile.agent_name
    result['extra_info'].update(
        tool_profile=profile.name, dataset_version=VERSION,
        source_prompt_sha256=hashlib.sha256(json.dumps(row['prompt'], ensure_ascii=False).encode()).hexdigest(),
        schema_sha256=digest, implicit_fk_corrections=sum(c['count'] for c in changes))
    return result, changes


def prepare(source, output, model, database_dir, prompt_limit=30720):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    profile = load_tool_profile('gemma4_native_v1')
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    manifest = {'dataset_version': VERSION, 'tool_profile': profile.name,
                'parser_format': profile.parser_format, 'agent_name': profile.agent_name,
                'parent': str(source), 'parent_sha256': {}, 'schema_sha256': {},
                'length_filter': False, 'truncated_rows': 0, 'gold_sql_changed': False,
                'evaluation_membership_and_order_changed': False, 'splits': {}}
    changes_by_db = {}
    for split in ['train', 'dev_full', 'validation']:
        src = source/f'{split}.parquet'
        manifest['parent_sha256'][str(src)] = file_hash(src)
        rows = pq.read_table(src).to_pylist()
        transformed, lengths, changed_rows = [], [], 0
        for index, row in enumerate(rows):
            result, changes = transform(row, profile, database_dir)
            transformed.append(result)
            lengths.append(profile.prompt_length(tokenizer, result['prompt']))
            changed_rows += bool(changes)
            db = result['extra_info']['db_id']
            manifest['schema_sha256'][db] = result['extra_info']['schema_sha256']
            if changes:
                changes_by_db[db] = changes
            if (index+1)%1000 == 0:
                print(f'{split}: {index+1}/{len(rows)}', flush=True)
        if max(lengths)>prompt_limit:
            raise ValueError(f'{split}: corrected prompt exceeds {prompt_limit}; max={max(lengths)}')
        path = output/f'{split}.parquet'
        pq.write_table(pa.Table.from_pylist(transformed), path)
        (output/f'{split}_prompt_lengths.json').write_text(json.dumps(lengths)+'\n')
        manifest['splits'][split] = {
            'count': len(rows), 'sha256': file_hash(path), 'fk_corrected_rows': changed_rows,
            'prompt_tokens': summarize_lengths(lengths, prompt_limit)}
    manifest['source_sha256'] = {name: file_hash(Path(__file__).parent/name) for name in
                               ['prepare_native_data.py', 'gemma_native_profile.py', 'tool_profiles.py',
                                'consensus/prompts.py', 'consensus/tool_calling.py']}
    manifest['tokenizer_sha256'] = {name: file_hash(Path(model)/name) for name in
                                   ['tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja']}
    (output/'foreign_key_corrections.json').write_text(json.dumps(changes_by_db, indent=2)+'\n')
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir', default='outputs/cp1_recipe/data')
    p.add_argument('--output-dir', default='outputs/gemma4_native_v1/data')
    p.add_argument('--model', required=True)
    p.add_argument('--database-dir', default='/opt/dlami/nvme/nl2sql/databases')
    args = p.parse_args()
    print(json.dumps(prepare(args.source_dir, args.output_dir, args.model, args.database_dir), indent=2))


if __name__ == '__main__':
    main()
