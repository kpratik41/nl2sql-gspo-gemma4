"""BIRD execution accuracy: exact sets of rows, positional columns; no SQL rewriting."""
import argparse
import json
from pathlib import Path

from .database import Database, atomic_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='outputs/california_schools_temp0')
    p.add_argument('--gold', default='data/raw/dev.json')
    args = p.parse_args()
    gold = {r['question_id']: r for r in json.loads(Path(args.gold).read_text())}
    database = Database()
    evaluations, filtered = [], []
    for path in sorted(Path(args.output).glob('[0-9]*.json'), key=lambda p: int(p.stem)):
        result = json.loads(path.read_text())
        if result['status'] == 'filtered_context_length':
            filtered.append({'question_id': result['question_id'], 'reason': result['status']})
            continue
        target = gold[result['question_id']]
        row = {'question_id': result['question_id'], 'db_id': result['db_id'], 'difficulty': target.get('difficulty'),
               'correct': False, 'status': result['status']}
        try:
            expected = database.query(target['db_id'], target['SQL'], max_rows=None, timeout=60)
        except Exception as exc:
            row['gold_error'] = str(exc)
        else:
            try:
                predicted = database.query(result['db_id'], result['prediction'], max_rows=None, timeout=60)
                row['correct'] = set(map(tuple, expected['rows'])) == set(map(tuple, predicted['rows']))
                row['gold_rows'], row['predicted_rows'] = len(expected['rows']), len(predicted['rows'])
            except Exception as exc:
                row['prediction_error'] = str(exc)
        evaluations.append(row)
    correct = sum(r['correct'] for r in evaluations)
    summary = {'requested_samples': len(evaluations)+len(filtered), 'samples': len(evaluations), 'correct': correct,
               'filtered_count': len(filtered), 'filtered': filtered,
               'execution_accuracy': correct / len(evaluations) if evaluations else None,
               'gold_errors': sum('gold_error' in r for r in evaluations),
               'prediction_errors': sum('prediction_error' in r for r in evaluations), 'items': evaluations}
    atomic_json(Path(args.output) / 'evaluation.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'items'}, indent=2))


if __name__ == '__main__':
    main()
