import json
import sqlite3
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from bird_inference.database import Database, mschema
from bird_inference.run import extract_sql, infer
from bird_inference.tools import ToolRunner, resolve_identifier, resolve_columns


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.path = self.root / 'toy.sqlite'
        with sqlite3.connect(self.path) as c:
            c.executescript('CREATE TABLE schools (id INTEGER PRIMARY KEY, "School Name" TEXT, score REAL);'
                            'INSERT INTO schools VALUES (1, "Oak High", 20), (2, "Oak High", NULL), (3, "Bay School", 40);')
        self.db = Database([self.root], self.root / 'cache')
        self.tools = ToolRunner(self.db, 'toy')

    def tearDown(self):
        self.tmp.cleanup()

    def test_exact_stats_cache_hit_and_invalidation(self):
        schema = self.db.schema('toy')
        score = schema['tables']['schools']['columns']['score']
        self.assertEqual((score['nulls'], score['distinct'], score['numeric_mean']), (1, 2, 30))
        with patch.object(self.db, '_schema', side_effect=AssertionError('cache should prevent DB scan')):
            self.assertEqual(self.db.schema('toy'), schema)
        with sqlite3.connect(self.path) as c:
            c.execute('INSERT INTO schools VALUES (4, "New", 60)')
        updated = self.db.schema('toy')
        self.assertNotEqual(schema['fingerprint'], updated['fingerprint'])
        self.assertEqual(updated['tables']['schools']['rows'], 4)
        self.assertIn('【Schema】', mschema(updated))

    def test_read_only_and_exact_sql(self):
        with self.assertRaises(ValueError):
            self.db.query('toy', 'DELETE FROM schools')
        with self.assertRaises(sqlite3.OperationalError):
            self.db.query('toy', 'WITH x AS (SELECT 1) DELETE FROM schools')
        result = self.db.query('toy', 'SELECT id FROM schools ORDER BY id', max_rows=1)
        self.assertEqual(result['rows'], [(1,)])
        self.assertTrue(result['truncated'])
        self.assertEqual(len(self.db.query('toy', 'SELECT id FROM schools', max_rows=None)['rows']), 3)
        self.assertEqual(self.db.query('toy', 'SELECT COUNT(*) FROM schools', max_rows=1)['rows'], [(3,)])

    def test_extended_statistics_and_baseline_cache_preservation(self):
        baseline = self.db.schema('toy')
        extended = self.db.schema('toy', 'extended')
        score = extended['tables']['schools']['columns']['score']
        self.assertEqual((score['numeric_count'], score['numeric_median'], score['numeric_q25'], score['numeric_q75']), (2, 30, 25, 35))
        self.assertEqual(score['numeric_stddev_population'], 10)
        self.assertEqual(score['mode']['tied_values'], 2)
        self.assertEqual(score['mode']['frequency'], 1)
        name = extended['tables']['schools']['columns']['School Name']
        self.assertEqual(name['mode'], {'value': 'Oak High', 'frequency': 2, 'tied_values': 1, 'non_null_fraction': 2/3})
        self.assertEqual(name['numeric_count'], 0)
        self.assertIsNone(name['numeric_median'])
        self.assertIn('Numeric median: 30', mschema(extended))
        self.assertIn('Mode representative:', mschema(extended))
        self.assertEqual(self.db.schema('toy'), baseline)
        with patch('bird_inference.database.distribution_stats', side_effect=AssertionError('cached stats must not be recalculated')):
            self.assertEqual(self.db.schema('toy', 'extended'), extended)
        with sqlite3.connect(self.path) as c:
            c.execute('INSERT INTO schools VALUES (4, "New", 60)')
        updated = self.db.schema('toy', 'extended')['tables']['schools']['columns']['score']
        self.assertEqual(updated['numeric_median'], 40)
        self.assertEqual((updated['numeric_q25'], updated['numeric_q75']), (30, 50))

    def test_extended_mixed_storage_and_empty_columns(self):
        with sqlite3.connect(self.path) as c:
            c.execute('CREATE TABLE mixed (value)')
            c.executemany('INSERT INTO mixed VALUES (?)', [(1,), (2,), (2,), (9,), ('100',), (None,)])
            c.execute('CREATE TABLE empty (value REAL)')
            c.execute('CREATE TABLE nulls (value REAL)')
            c.execute('INSERT INTO nulls VALUES (NULL)')
        schema = self.db.schema('toy', 'extended')
        stats = schema['tables']['mixed']['columns']['value']
        self.assertEqual(stats['numeric_count'], 4)
        self.assertEqual(stats['storage_counts'], {'integer': 4, 'text': 1, 'null': 1})
        self.assertEqual(stats['numeric_mean'], 3.5)
        self.assertEqual((stats['numeric_median'], stats['numeric_q25'], stats['numeric_q75']), (2, 1.75, 3.75))
        self.assertAlmostEqual(stats['numeric_stddev_population'], (10.25)**0.5)
        self.assertEqual(stats['mode']['value'], 2)
        for table in ['empty', 'nulls']:
            stats = schema['tables'][table]['columns']['value']
            self.assertIsNone(stats['numeric_median'])
            self.assertIsNone(stats['numeric_stddev_population'])
            self.assertIsNone(stats['mode'])

    def test_extended_filtered_peek_uses_filtered_stats_and_cache(self):
        args = {'db_id': 'toy', 'table': 'schools', 'columns': ['score', 'School Name'], 'where': 'id > 1'}
        baseline = self.tools.call('sqlite_peek', args)
        runner = ToolRunner(self.db, 'toy', stats_profile='extended')
        result = runner.call('sqlite_peek', args)
        score = result['columns']['score']['profile']
        self.assertEqual((score['numeric_count'], score['numeric_median'], score['numeric_stddev_population']), (1, 40, 0))
        self.assertEqual(result['columns']['School Name']['profile']['mode']['tied_values'], 2)
        self.assertNotIn('numeric_median', baseline['columns']['score']['profile'])
        with patch.object(self.db, 'query', side_effect=AssertionError('cached filtered profile must not rescan')):
            self.assertEqual(runner.call('sqlite_peek', args), result)

    def test_peek_quoted_columns_and_filtered_stats(self):
        result = self.tools.call('sqlite_peek', {'db_id': 'toy', 'table': 'schools', 'columns': ['School Name', 'score'], 'where': 'id > 1'})
        self.assertNotIn('error', result)
        self.assertEqual(result['columns']['score']['profile']['rows'], 2)
        self.assertEqual(result['columns']['score']['profile']['nulls'], 1)
        self.assertEqual(result['columns']['score']['profile']['numeric_mean'], 40)
        with patch.object(self.db, 'query', side_effect=AssertionError('cache should prevent repeated filtered scan')):
            self.assertEqual(self.tools.call('sqlite_peek', {'db_id': 'toy', 'table': 'schools', 'columns': ['School Name', 'score'], 'where': 'id > 1'}), result)

    def test_value_search_and_db_scope(self):
        result = self.tools.call('bm_25_search', {'db_id': 'toy', 'table': 'schools', 'column': 'School Name', 'query': 'Oak', 'top_k': 1})
        self.assertIn('Oak High', result['values'])
        self.assertEqual(len(result['values']), len(set(result['values'])))
        self.assertIn('error', self.tools.call('sqlite_query', {'db_id': 'another', 'sql': 'SELECT 1'}))
        self.assertIn('error', self.tools.call('sqlite_peek', {'db_id': 'toy', 'table': 'schools', 'columns': ['fake']}))
        self.assertIn('error', self.tools.call('unknown', {'db_id': 'toy'}))

    def test_identifier_wrappers_case_and_cache_reuse(self):
        args = {'db_id': 'toy', 'table': '`SCHOOLS`', 'columns': ['"school name"', '[SCORE]']}
        result = self.tools.call('sqlite_peek', args)
        self.assertNotIn('error', result)
        self.assertEqual(set(result['columns']), {'School Name', 'score'})
        self.assertEqual(args['columns'], ['"school name"', '[SCORE]'])
        with patch.object(self.db, 'query', side_effect=AssertionError('canonical names should share the cache')):
            self.assertEqual(self.tools.call('sqlite_peek', {'db_id': 'toy', 'table': 'schools',
                                                          'columns': ['School Name', 'score']}), result)
        result = self.tools.call('bm_25_search', {'db_id': 'toy', 'table': '[schools]',
                                                'column': '`school name`', 'query': 'Oak', 'top_k': 1})
        self.assertIn('Oak High', result['values'])

    def test_invalid_identifier_has_feedback_without_fuzzy_rewrite(self):
        for column in ['SchoolName', '"School Name', None]:
            result = self.tools.call('sqlite_peek', {'db_id': 'toy', 'table': 'schools', 'columns': [column]})
            self.assertIn('error', result)
            self.assertIn('School Name', result['message'])
        # Exact names win, and valid SQL escaping is handled without guessing names.
        self.assertEqual(resolve_identifier('"literal"', ['"literal"', 'literal'], 'column'), '"literal"')
        self.assertEqual(resolve_identifier('"a""b"', ['a"b'], 'column'), 'a"b')

    def test_serialized_column_lists_and_wildcard(self):
        for columns in ['School Name, score', '["School Name", "score"]', '"School Name", score']:
            result = self.tools.call('sqlite_peek', {'db_id': 'toy', 'table': 'schools', 'columns': columns})
            self.assertNotIn('error', result)
            self.assertEqual(set(result['columns']), {'School Name', 'score'})
        result = self.tools.call('sqlite_peek', {'db_id': 'toy', 'table': 'schools', 'columns': '*'})
        self.assertEqual(set(result['columns']), {'id', 'School Name', 'score'})
        self.assertEqual(resolve_columns('a,b', ['a,b', 'a', 'b']), ['a,b'])
        self.assertEqual(resolve_columns('"a,b", c', ['a,b', 'c']), ['a,b', 'c'])
        with self.assertRaises(ValueError):
            resolve_columns('score, fake', ['score'])

    def test_final_answer_extraction(self):
        self.assertEqual(extract_sql('text <final_answer><sql_code>SELECT 1</sql_code></final_answer>'), 'SELECT 1')
        self.assertIsNone(extract_sql('<think>SELECT 2</think>'))
        self.assertIsNone(extract_sql('<final_answer><sql_code>SELECT 1</sql_code></final_answer>' * 2))


class PreparedDataTests(unittest.TestCase):
    @unittest.skipUnless(Path('data/prepared/california_schools.jsonl').exists(), 'Run data preparation first')
    def test_dataset_and_no_dev_gold_in_records(self):
        records = [json.loads(line) for line in Path('data/prepared/california_schools.jsonl').read_text().splitlines()]
        self.assertEqual(len(records), 89)
        self.assertEqual(len({r['question_id'] for r in records}), 89)
        for r in records:
            self.assertNotIn('SQL', r)
            self.assertEqual(len(r['few_shot_examples']), 3)
            self.assertTrue(all(e['db_id'] != r['db_id'] for e in r['few_shot_examples']))
            self.assertEqual({t['function']['name'] for t in r['tools']}, {'sqlite_query', 'sqlite_peek', 'bm_25_search'})


class RoundBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_eight_tool_rounds_then_final_answer(self):
        requests = []
        async def create(**kwargs):
            requests.append(kwargs)
            if len(requests) <= 8:
                message = {'role': 'assistant', 'tool_calls': [{'id': str(len(requests)), 'type': 'function',
                           'function': {'name': 'sqlite_query', 'arguments': '{"db_id":"toy","sql":"SELECT 1"}'}}]}
            else:
                message = {'role': 'assistant', 'content': '<final_answer><sql_code>SELECT 1</sql_code></final_answer>'}
            return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop', message=SimpleNamespace(model_dump=lambda **kw: message))], usage=None)
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        args = SimpleNamespace(max_tool_rounds=8, max_tokens=8192, max_model_len=43000, model='test')
        record = {'question_id': 1, 'db_id': 'toy', 'question': 'test', 'messages': [{'role': 'user', 'content': 'test'}], 'tools': [], 'stats_profile': 'extended'}
        with patch('bird_inference.run.token_count', return_value=(100, 43000)), patch('bird_inference.run.ToolRunner') as runner:
            runner.return_value.call.return_value = {'rows': [[1]], 'columns': ['1'], 'truncated': False}
            result = await infer(client, 'http://localhost:8000/v1', 0, record, args, None)
            runner.assert_called_once_with(None, 'toy', stats_profile='extended')
            self.assertEqual(runner.return_value.call.call_count, 8)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['prediction'], 'SELECT 1')
        self.assertEqual(len(requests), 9)
        self.assertEqual(requests[-1]['tool_choice'], 'none')
        self.assertTrue(all(r['temperature'] == 0 and not r['parallel_tool_calls'] for r in requests))
        self.assertTrue(all(not r['extra_body']['chat_template_kwargs']['truncate_history_thinking'] for r in requests))

    async def test_context_overflow_is_reported_without_truncation(self):
        args = SimpleNamespace(max_tool_rounds=8, max_tokens=8192, max_model_len=43000, model='test')
        record = {'question_id': 1, 'db_id': 'toy', 'question': 'test', 'messages': [{'role': 'user', 'content': 'keep me'}], 'tools': []}
        with patch('bird_inference.run.token_count', return_value=(40000, 43000)):
            result = await infer(None, 'http://localhost:8000/v1', 0, record, args, None)
        self.assertEqual(result['status'], 'filtered_context_length')
        self.assertEqual(result['messages'], record['messages'])


if __name__ == '__main__':
    unittest.main()
