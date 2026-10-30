"""Score completed dev predictions with the saved official BIRD evaluator.

Uses an isolated func_timeout dependency and read-only SQLite connections. No
inference, training, or environment installation occurs in this script.
"""
import argparse
import hashlib
import importlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time


def load_official(directory):
    sys.path.insert(0, str(directory / 'deps'))
    sys.path.insert(0, str(directory))
    return importlib.import_module('evaluation')


def initialize_worker(directory):
    official = load_official(Path(directory))
    connect = official.sqlite3.connect

    def readonly_connect(path, *args, **kwargs):
        # The comparison and timeout functions remain untouched. The adapter
        # prevents any predicted SQL from writing to the source databases.
        kwargs['uri'] = True
        return connect(Path(path).resolve().as_uri() + '?mode=ro', *args, **kwargs)

    official.sqlite3.connect = readonly_connect


def score_one(item):
    import evaluation
    return evaluation.execute_model(*item)


def main(args):
    from .consensus.sql_utils import extract_final_answer_sql
    root = Path(args.output_dir).resolve()
    if args.wait:
        while not (root / 'summary.json').exists():
            time.sleep(10)
    summary = json.loads((root / 'summary.json').read_text())
    assert summary['status'] == 'completed'
    directory = root / 'bird_official'
    source = json.loads((directory / 'source.json').read_text())
    assert hashlib.sha256((directory / 'evaluation.py').read_bytes()).hexdigest() == source['sha256']
    official = load_official(directory)
    rows = [json.loads(line) for line in (root / 'results.jsonl').open()]
    assert [r['reference_index'] for r in rows] == list(range(1534))
    import pyarrow.parquet as pq
    table = pq.read_table('outputs/gemma4_native_v1/data/dev_full.parquet').to_pylist()
    metadata = json.loads(Path(args.dev_metadata).read_text())
    assert len(metadata) == len(rows) == len(table) == 1534
    items, predictions = [], {}
    for index, (row, original, reference) in enumerate(zip(rows, metadata, table, strict=True)):
        gold = json.loads(reference['reward_model']['ground_truth'])
        assert gold['db_id'] == row['db_id'] == original['db_id']
        # The parquet preparation adds a terminal semicolon; verify that this
        # is the only gold-side difference, then use the untouched BIRD SQL.
        assert gold['gold_sql'].strip().rstrip(';').rstrip() == original['SQL'].strip().rstrip(';').rstrip()
        assert reference['extra_info']['question'].strip() == original['question'].strip()
        prediction = extract_final_answer_sql(row['assistant_text'])
        predictions[str(index)] = prediction + '\t----- bird -----\t' + row['db_id']
        db = Path(args.database_dir) / row['db_id'] / (row['db_id'] + '.sqlite')
        assert db.exists()
        items.append((prediction, original['SQL'], str(db.resolve()), index, args.timeout))
    (directory / 'predict_dev.json').write_text(json.dumps(predictions, indent=2) + '\n')
    started = time.monotonic()
    with mp.get_context('spawn').Pool(args.cpus, initializer=initialize_worker,
                                     initargs=(str(directory),)) as pool:
        results = list(pool.imap_unordered(score_one, items))
    results = official.sort_results(results)
    assert [r['sql_idx'] for r in results] == list(range(1534))
    simple, moderate, challenging, accuracy, counts = official.compute_acc_by_diff(results, args.dev_metadata)
    mismatches = [{'reference_index': r['sql_idx'], 'official_bird_ex': r['res'],
                   'native_result_reward': rows[r['sql_idx']]['scores']['result']}
                  for r in results if r['res'] != rows[r['sql_idx']]['scores']['result']]
    correct = sum(r['res'] for r in results)
    report = {'metric': 'BIRD execution accuracy (EX)', 'correct': correct, 'total': 1534,
              'accuracy_percent': accuracy, 'timeout_seconds': args.timeout,
              'comparison': 'set(predicted_rows) == set(gold_rows), raw SQLite values',
              'missing_final_answer': sum(not extract_final_answer_sql(r['assistant_text']) for r in rows),
              'by_difficulty': {label: {'count': count, 'accuracy_percent': score}
                                for label, count, score in zip(['simple', 'moderate', 'challenging'],
                                                               counts[:3], [simple, moderate, challenging], strict=True)},
              'native_reward_mismatches': mismatches, 'official_source': source,
              'dev_metadata_sha256': hashlib.sha256(Path(args.dev_metadata).read_bytes()).hexdigest(),
              'connection_adapter': 'SQLite read-only URI; comparison/timeout functions unchanged',
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'results_sha256': summary['results_sha256'], 'elapsed_seconds': time.monotonic() - started}
    (directory / 'execution_results.json').write_text(json.dumps(results, indent=2) + '\n')
    (directory / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    summary['bird_ex'] = report
    temp = root / 'summary.json.tmp'
    temp.write_text(json.dumps(summary, indent=2) + '\n')
    temp.replace(root / 'summary.json')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--dev-metadata', default='data/bird_dev_data/raw/dev.json')
    parser.add_argument('--database-dir', default='/opt/dlami/nvme/nl2sql/databases/dev_databases')
    parser.add_argument('--timeout', type=float, default=30.)
    parser.add_argument('--cpus', type=int, default=16)
    parser.add_argument('--wait', action='store_true')
    main(parser.parse_args())
