import argparse
import hashlib
import json
from pathlib import Path

from rank_bm25 import BM25Okapi

from .database import Database, atomic_json, mschema
from .tools import TOOLS, tokenize

SYSTEM = '''You are a SQLite text-to-SQL agent. Answer the question with one executable read-only SQLite query.
Use the supplied M-Schema, column statistics, question, and evidence. Evidence defines authoritative mappings and metrics.
The three examples come from different training databases and illustrate SQL patterns; their table and column names do not belong to the current database.
Draft a candidate query, then use sqlite_query to verify the exact SQL before finalizing. Inspect results and repair if needed.
Use sqlite_peek to inspect types, ranges, NULLs, and samples. Use bm_25_search to find exact stored text literals.
Use native function calls through the supplied tool definitions. Make at most ONE tool call per round, then stop and wait for the real tool response. You have at most 8 tool rounds; answer sooner when ready.
Return exactly the requested columns and filters. Use SQLite syntax, quote unusual identifiers, and use floating-point division for ratios.
Do not add a LIMIT to the final SQL unless justified by the question. Tool row truncation is only a display limit.
When ready, return exactly one final SQL query as <final_answer><sql_code>YOUR SQL</sql_code></final_answer>. Put no prose, markdown fences, or tool calls inside sql_code.
'''


VERIFICATION = '''Before writing SQL, identify the requested output columns in order, the entity represented by one output row, required filters, join keys, aggregation, and ordering or tie requirements. Keep this checklist in your reasoning; do not add it to the final answer.
Project only the requested answer fields, in the order requested. Columns used only for filtering, joining, grouping, or sorting need not appear in SELECT. If asked only for a rate, return the rate without adding a name or identifier.
Distinguish a ratio from a percentage and an average of ratios from a ratio of totals. Follow evidence for the numerator, denominator, and scale. Consider NULLs, zero denominators, and join multiplicity without inventing extra filters.
Map each identifier to its owning table in M-Schema. For sqlite_peek and bm_25_search, pass plain schema names; SQL identifier quoting belongs in the SQL string. If a tool reports an invalid identifier, check its valid names and the schema before retrying; do not repeat the same failed call.
A query that executes is not necessarily correct. Before finalizing, compare the returned column count/order, filters, aggregation, and result shape with the question and evidence. A truncated tool preview is not the full result, and an empty result alone does not justify relaxing requested filters.
Use tool rounds purposefully and reserve a round to execute the final candidate. If a verified candidate satisfies the requirements, return that exact SQL without adding columns or making speculative changes. If it needs a repair, execute the repaired SQL while budget remains. Never fabricate a tool result.
'''

PROMPTS = {'baseline': SYSTEM, 'verified': SYSTEM + VERIFICATION}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default='california_schools', help='Database id, or all for all 1534 dev questions')
    parser.add_argument('--output', default=None)
    parser.add_argument('--prompt-profile', choices=sorted(PROMPTS), default='baseline')
    parser.add_argument('--stats-profile', choices=['baseline', 'extended'], default='baseline')
    args = parser.parse_args()
    dev = json.loads(Path('data/raw/dev.json').read_text())
    train = [json.loads(line) for line in Path('data/raw/train-6601.jsonl').read_text().splitlines() if line.strip()]
    assert len(dev) == 1534 and len(train) == 6601
    assert not ({x['db_id'] for x in train} & {x['db_id'] for x in dev}), 'Train/dev databases must be disjoint'
    database = Database()
    # Verify database availability, without opening or profiling unused train DBs.
    paths = {db: str(database.resolve(db)) for db in sorted({x['db_id'] for x in dev + train})}
    index = BM25Okapi([tokenize(x['question'] + ' ' + x.get('evidence', '')) for x in train])
    selected = [x for x in dev if args.db == 'all' or x['db_id'] == args.db]
    if not selected:
        raise ValueError('No questions selected')
    schemas = {}
    suffix = '' if args.prompt_profile == 'baseline' else f'.{args.prompt_profile}'
    if args.stats_profile != 'baseline':
        suffix += f'.stats-{args.stats_profile}'
    out = Path(args.output or f'data/prepared/{args.db}{suffix}.jsonl')
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w') as f:
        for item in selected:
            db_id = item['db_id']
            if db_id not in schemas:
                print(f'Profiling/loading {db_id}', flush=True)
                schemas[db_id] = mschema(database.schema(db_id, args.stats_profile))
            scores = index.get_scores(tokenize(item['question'] + ' ' + item.get('evidence', '')))
            indices = sorted(range(len(train)), key=lambda i: (-scores[i], i))[:3]
            examples = [{'train_index': i, 'bm25_score': float(scores[i]), **train[i]} for i in indices]
            example_text = '\n\n'.join(f"Example {n}:\nDatabase: {e['db_id']}\nQuestion: {e['question']}\nEvidence: {e.get('evidence', '')}\nSQL: {e['SQL']}" for n, e in enumerate(examples, 1))
            user = f"<database_schema>\n{schemas[db_id]}\n</database_schema>\n\n<training_examples>\n{example_text}\n</training_examples>\n\n<db_id>{db_id}</db_id>\n<question>{item['question']}</question>\n<hint>{item.get('evidence', '')}</hint>"
            # Evaluation gold is deliberately absent from inference records.
            record = {'question_id': item['question_id'], 'db_id': db_id, 'question': item['question'],
                      'messages': [{'role': 'system', 'content': PROMPTS[args.prompt_profile]}, {'role': 'user', 'content': user}],
                      'tools': TOOLS, 'few_shot_examples': examples,
                      'schema_fingerprint': database.fingerprint(db_id)}
            if args.stats_profile != 'baseline':
                record['stats_profile'] = args.stats_profile
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    atomic_json(out.with_suffix('.manifest.json'), {'count': len(selected), 'selection': args.db,
                'source': json.loads(Path('data/raw/provenance.json').read_text()), 'db_paths': paths,
                'input_sha256': hashlib.sha256(out.read_bytes()).hexdigest(), 'few_shots': 3,
                'prompt_profile': args.prompt_profile,
                'stats_profile': args.stats_profile,
                'retrieval': 'BM25Okapi(question + evidence), training only; ties by training row index'})
    print(f'Wrote {len(selected)} prompts to {out}', flush=True)


if __name__ == '__main__':
    main()
