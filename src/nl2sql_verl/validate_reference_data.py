"""Independently check the saved parquet against raw data and prompt references."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

from .prepare_reference_data import file_hash, identity, read_jsonl, sql_tokens
from .tool_profiles import load_tool_profile


def validate(output):
    import pyarrow.parquet as pq
    output = Path(output)
    manifest = json.loads((output/'manifest.json').read_text())
    for path, digest in manifest['input_sha256'].items():
        assert file_hash(path) == digest, f'Input changed: {path}'
    raw = read_jsonl(manifest['training_source'])
    references = {
        'train': read_jsonl(manifest['train_reference']),
        'dev_full': read_jsonl(manifest['eval_reference'])}
    rows = {split: pq.read_table(output/f'{split}.parquet').to_pylist()
            for split in ('train', 'dev_full', 'validation')}
    profile = load_tool_profile(manifest['tool_profile']) if manifest['tool_profile'] == 'gemma4' else None
    for split in ('train', 'dev_full', 'validation'):
        assert file_hash(output/f'{split}.parquet') == manifest['splits'][split]['sha256']
        assert len(rows[split]) == manifest['splits'][split]['count']
        assert len(json.loads((output/f'{split}_prompt_lengths.json').read_text())) == len(rows[split])
    assert len(rows['train']) == len(raw) == 6601
    for split in ('train', 'dev_full'):
        for index, row in enumerate(rows[split]):
            reference = references[split][row['extra_info']['reference_index']]
            if profile:
                assert row['prompt'] == reference['prompt'], (split, index)
                assert reference['tools'] == profile.declarations()
            else:
                assert row['prompt'][1] == reference['prompt'][1]
            assert [m['role'] for m in row['prompt']] == ['system', 'user']
            truth = json.loads(row['reward_model']['ground_truth'])
            assert truth['db_id'] == reference['db_id'] == row['extra_info']['db_id']
            assert sql_tokens(truth['gold_sql']) == sql_tokens(reference['gold_sql'])
            if split == 'train':
                assert identity(raw[index]) == identity(reference)
                assert sql_tokens(truth['gold_sql']) == sql_tokens(raw[index]['SQL'])
                user = row['prompt'][1]['content']
                assert not user.split('<question>', 1)[0].strip()
                schema = user.split('<database_schema>', 1)[1].split('</database_schema>', 1)[0]
                assert all(marker not in schema for marker in
                           ('Stats:', 'Examples:', 'Top values:', 'Nullable', 'Not Null'))
            for name in ('bm25_search_sqlite', 'sqlite_peek', 'sqlite_query'):
                assert row['extra_info']['tools_kwargs'][name]['create_kwargs'] == {'db_id': truth['db_id']}
    assert rows['validation'] == [rows['dev_full'][i] for i in manifest['dev_reference_indices']]
    train_dbs = {r['extra_info']['db_id'] for r in rows['train']}
    dev_dbs = {r['extra_info']['db_id'] for r in rows['dev_full']}
    assert not train_dbs & dev_dbs
    assert {r['extra_info']['db_id'] for r in rows['validation']} == dev_dbs

    # Check every retrieved demonstration, including multiline questions,
    # against actual raw training rows rather than trusting its database label.
    raw_by_db = defaultdict(list)
    for row in raw:
        raw_by_db[row['db_id']].append(row)
    fewshots_checked = 0
    for row in rows['dev_full']:
        prefix = row['prompt'][1]['content'].split('<question>', 1)[0]
        examples = re.findall(r'- Example \d+ \(db: ([^)]+)\)\n(.*?)(?=\n- Example |\Z)', prefix, re.S)
        assert len(examples) == 3
        for db, body in examples:
            assert db in train_dbs and db != row['extra_info']['db_id']
            matched = False
            for source in raw_by_db[db]:
                question_prefix = '  Q: '+source['question']+'\n'
                if not body.startswith(question_prefix):
                    continue
                rest = body[len(question_prefix):]
                hint = '  Hint: '+source.get('evidence', '')+'\n'
                if rest.startswith(hint):
                    rest = rest[len(hint):]
                if sql_tokens(rest.strip()) == sql_tokens(source['SQL']):
                    matched = True
                    break
            assert matched, f'Dev few-shot does not match raw training data: {db}'
            fewshots_checked += 1
    report = {
        'status': 'pass', 'rows': {k: len(v) for k, v in rows.items()},
        'exact_reference_prompt_parity': profile is not None,
        'all_6601_raw_training_rows_and_gold_verified': True,
        'validation_is_saved_subset_of_full_dev': True,
        'train_dev_database_overlap': 0,
        'eval_fewshots_matched_to_raw_training_rows': fewshots_checked,
        'train_trajectories': 0, 'eval_trajectories': 0,
        'fits_configured_prompt_limit': manifest['fits_configured_prompt_limit']}
    (output/'validation_report.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='outputs/cp1_recipe/data')
    print(json.dumps(validate(parser.parse_args().output_dir), indent=2))
