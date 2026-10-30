"""Offline diagnosis of saved rollout text; never changes the trainer or rewards.

JSONL rollout text has special tokens removed, so re-tokenized lengths are
estimates and cannot recover generation stop reasons or executed tool traces.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import sqlite3

from .consensus import rewards as rw
from .consensus.sql_utils import (
    bird_execute_sql, extract_final_answer_sql, extract_sql, get_database_path,
    is_safe_readonly_sql,
)


def format_checks(text):
    match = rw.FINAL_SQL_RE.search(text)
    code = rw.SQL_CODE_CONTENT_RE.search(match.group(0)) if match else None
    sql = code.group(1).strip() if code else ''
    return {
        'scratch_pad': bool(rw.SCRATCH_PAD_RE.search(text)),
        'relevant_tables': bool(rw.RELEVANT_TABLES_RE.search(text)),
        'relevant_columns': bool(rw.RELEVANT_COLUMNS_RE.search(text)),
        'final_sql_structure': bool(match),
        'final_sql_extracted': bool(extract_final_answer_sql(text)),
        'clean_sql_code': bool(sql) and not any(
            marker in (sql if marker in ('call:', '```') else sql.lower())
            for marker in ('<scratch_pad', '<final_answer', 'call:', '```')
        ),
    }


def explain(sql, db_id, database_dir):
    """Compile only: detects false positives in the lexical readonly guard."""
    path = get_database_path(db_id, database_dir)
    if not path:
        return False, 'missing_database'
    try:
        with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as conn:
            conn.execute('EXPLAIN ' + sql)
        return True, ''
    except Exception as exc:
        return False, str(exc)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--files', nargs='+', required=True)
    p.add_argument('--database-dir', required=True)
    p.add_argument('--tokenizer', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    execute_cache = {}
    summaries = []
    for filename in args.files:
        rows = [json.loads(line) for line in Path(filename).open()]
        texts = [r['output'] for r in rows]
        lengths = [len(ids) for ids in tok(texts, add_special_tokens=False)['input_ids']]
        queries = [(json.loads(r['gts'])['db_id'], extract_sql(r['output'])) for r in rows]
        missing = list(dict.fromkeys(q for q in queries if q not in execute_cache))

        def execute(q):
            db, sql = q
            ok, _, err = bird_execute_sql(sql, db, args.database_dir, timeout_s=60)
            return q, (ok, err)

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            execute_cache.update(pool.map(execute, missing))
        reasons, errors, matrix, failed_db = Counter(), Counter(), Counter(), Counter()
        records = []
        logged_disagreements = 0
        for i, (row, length, query) in enumerate(zip(rows, lengths, queries)):
            db, sql = query
            checks = format_checks(row['output'])
            fmt = all(checks.values())
            ok, err = execute_cache[query]
            logged_disagreements += int('execution' in row and row['execution'] != int(ok))
            matrix[f'format_{int(fmt)}_execution_{int(ok)}'] += 1
            failures = [k for k, value in checks.items() if not value]
            if not fmt:
                reasons.update(failures)
                failed_db[db] += 1
            if not ok:
                category = ('no_extracted_sql' if not sql else
                            'readonly_guard' if not is_safe_readonly_sql(sql) else
                            'missing_column' if 'no such column' in err else
                            'missing_table' if 'no such table' in err else
                            'ambiguous_column' if 'ambiguous column' in err else
                            'timeout' if 'interrupt' in err or 'timeout' in err else
                            'syntax' if 'syntax' in err or 'unrecognized token' in err else
                            'other')
                errors[category] += 1
            else:
                category = ''
            if not fmt or not ok:
                record = {
                    'row': i, 'uid': row.get('uid'), 'db_id': db,
                    'logged_score': row['score'],
                    'format': int(fmt), 'execution_replay': int(ok),
                    'logged_format': row.get('format'), 'logged_execution': row.get('execution'),
                    'missing_format_checks': failures, 'execution_category': category,
                    'execution_error': err, 'sql': sql,
                    'retokenized_length_without_special_tokens': length,
                    'tool_expression_count': len(re.findall(r'call:\w+\s*\{', row['output'])),
                    'has_complete_final_answer': bool(extract_final_answer_sql(row['output'])),
                    'output': row['output'],
                    'question': row['input'].rsplit('Question:', 1)[-1].split('\nmodel')[0],
                }
                if category == 'readonly_guard':
                    record['sqlite_explain'] = explain(sql, db, args.database_dir)
                records.append(record)
        n = len(rows)
        fmt_ok = sum(v for k, v in matrix.items() if k.startswith('format_1'))
        exec_ok = sum(v for k, v in matrix.items() if k.endswith('execution_1'))
        summary = {
            'file': filename, 'rows': n, 'format_pass': fmt_ok,
            'execution_replay_pass': exec_ok,
            'format_percent': 100*fmt_ok/n, 'execution_replay_percent': 100*exec_ok/n,
            'matrix': dict(matrix), 'format_failures_overlapping': dict(reasons),
            'execution_failures': dict(errors), 'format_failure_db': dict(failed_db),
            'format_failed_approx_length_ge_2000': sum(
                r['format'] == 0 and r['retokenized_length_without_special_tokens'] >= 2000 for r in records),
            'all_rows_approx_length_ge_2000': sum(x >= 2000 for x in lengths),
            'complete_final_but_format_failed': sum(
                r['format'] == 0 and r['has_complete_final_answer'] for r in records),
            # Empty SQL gives zero on all other reward channels, so -0.1 is
            # precisely the fully saturated length penalty (weight 0.1).
            'no_sql_with_saturated_length_penalty': sum(
                r['execution_category'] == 'no_extracted_sql' and abs(r['logged_score']+0.1)<1e-6
                for r in records),
            'replay_disagrees_with_logged_execution': logged_disagreements,
        }
        outname = Path(filename).parent.name + '_' + Path(filename).stem
        (root/(outname + '_failures.jsonl')).write_text(''.join(json.dumps(r) + '\n' for r in records))
        summaries.append(summary)
        (root/'summary.json').write_text(json.dumps(summaries, indent=2)+'\n')
        print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
