import asyncio
import json
import sqlite3

import pytest

from src.nl2sql_verl.gemma_arguments import parse_arguments
from src.nl2sql_verl.prepare_data import normalize_record, prepare
from src.nl2sql_verl.task import assistant_tokens, compute_score, database_path


@pytest.fixture
def database(tmp_path, monkeypatch):
    directory = tmp_path/'demo'
    directory.mkdir()
    with sqlite3.connect(directory/'demo.sqlite') as conn:
        conn.execute('CREATE TABLE people(name TEXT, age INTEGER)')
        conn.executemany('INSERT INTO people VALUES (?, ?)', [('Ada', 36), ('Bob', 40)])
    monkeypatch.setenv('NL2SQL_DATABASE_DIR', str(tmp_path))
    return tmp_path


def row():
    return {'messages': [{'role': 'user', 'content': '<db_id>demo</db_id>How many people?'},
                         {'role': 'assistant', 'content': 'SELECT COUNT(*) FROM people;'}],
            'db_id': 'demo', 'gold_sql': 'SELECT COUNT(*) FROM people;'}


def test_parquet_normalization_does_not_leak_gold(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    src, dest = tmp_path/'source.parquet', tmp_path/'verl.parquet'
    pq.write_table(pa.Table.from_pylist([row()]), src)
    assert prepare(src, dest)['kept'] == 1
    result = pq.read_table(dest).to_pylist()[0]
    assert all(m['role'] != 'assistant' for m in result['prompt'])
    assert 'SELECT COUNT(*) FROM people' not in str(result['prompt'])
    assert json.loads(result['reward_model']['ground_truth'])['gold_sql'] == row()['gold_sql']
    assert result['extra_info']['tools_kwargs']['sqlite_peek']['create_kwargs']['db_id'] == 'demo'
    # Already-normalized parquet is accepted too.
    assert normalize_record(result)['reward_model'] == result['reward_model']


def test_prompt_filter_counts_tokens_with_transformers_mapping_default(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import BatchEncoding

    class Tokenizer:
        def apply_chat_template(self, messages, *, return_dict=True, **kwargs):
            # Transformers 5.18 defaults to a two-key BatchEncoding, whose
            # len() is not the number of tokens in the prompt.
            assert len(kwargs['tools']) == 3
            ids = list(range(20 if messages[-1]['content'] == 'long prompt' else 3))
            return BatchEncoding({'input_ids': ids, 'attention_mask': [1]*len(ids)}) if return_dict else ids

    short, long = row(), row()
    long['messages'][0]['content'] = 'long prompt'
    source, destination = tmp_path/'source.parquet', tmp_path/'filtered.parquet'
    pq.write_table(pa.Table.from_pylist([short, long]), source)
    result = prepare(source, destination, Tokenizer(), max_prompt_length=10)
    assert result['kept'] == 1 and result['overlength_dropped'] == 1
    assert pq.read_table(destination).num_rows == 1


def test_native_gemma_arrays_and_sql_literals():
    values = parse_arguments('db_id:<|"|>demo<|"|>,columns:[<|"|>name<|"|>,<|"|>age<|"|>],'
                             'sql:<|"|>SELECT "a:b,{c}" FROM people<|"|>,profile:false,limit:3')
    assert values['columns'] == ['name', 'age']
    assert values['sql'] == 'SELECT "a:b,{c}" FROM people'
    assert values['profile'] is False and values['limit'] == 3
    with pytest.raises(ValueError):
        parse_arguments('db_id:!!!')


def test_verl_native_tool_parser_dispatch():
    from src.nl2sql_verl.agent_loop import NL2SQLGemmaParser
    class Tokenizer:
        pad_token = None
        def convert_tokens_to_ids(self, value):
            return 99
        def decode(self, ids, **kwargs):
            return '<|tool_call>call:sqlite_peek{db_id:<|"|>demo<|"|>,table:<|"|>people<|"|>,columns:[<|"|>name<|"|>,<|"|>age<|"|>]}<tool_call|>'
    parser = NL2SQLGemmaParser(Tokenizer())
    _, calls = asyncio.run(parser.extract_tool_calls([1,2]))
    assert parser.stop_token_ids == [99]
    assert calls[0].name == 'sqlite_peek'
    assert json.loads(calls[0].arguments)['columns'] == ['name','age']


@pytest.mark.parametrize('prefix', ['<scratch_pad>Check it.</scratch_pad>\n', '<|tool_call>'])
def test_observed_gemma_call_with_optional_opening_token(prefix):
    from src.nl2sql_verl.agent_loop import NL2SQLGemmaParser
    class Tokenizer:
        pad_token = None
        def convert_tokens_to_ids(self, value):
            return 99
        def decode(self, ids, **kwargs):
            return prefix + 'call:sqlite_query{db_id:<|"|>demo<|"|>,sql:<|"|>SELECT "a:b,{c}" FROM people<|"|>}<tool_call|>'
    content, calls = asyncio.run(NL2SQLGemmaParser(Tokenizer()).extract_tool_calls([1]))
    assert len(calls) == 1
    assert json.loads(calls[0].arguments)['sql'] == 'SELECT "a:b,{c}" FROM people'
    assert '<|tool_call>' not in content


@pytest.mark.parametrize('text', [
    'An example is call:sqlite_query{db_id:<|"|>demo<|"|>}<tool_call|>',
    'call:sqlite_query{db_id:<|"|>demo<|"|>}',
])
def test_tool_parser_rejects_inline_mentions_and_incomplete_calls(text):
    from src.nl2sql_verl.agent_loop import NL2SQLGemmaParser
    class Tokenizer:
        pad_token = None
        def convert_tokens_to_ids(self, value):
            return 99
        def decode(self, ids, **kwargs):
            return text
    content, calls = asyncio.run(NL2SQLGemmaParser(Tokenizer()).extract_tool_calls([1]))
    assert content == text and calls == []


def test_reward_uses_only_assistant_text(database, monkeypatch):
    correct = ('<scratch_pad>Count rows. <relevant_tables>people</relevant_tables>'
               '<relevant_columns>name</relevant_columns></scratch_pad>'
               '<final_answer><sql_code>SELECT COUNT(*) FROM people;</sql_code></final_answer>')
    kwargs = dict(data_source='nl2sql_bird', ground_truth=json.dumps({'db_id':'demo','gold_sql':row()['gold_sql']}))
    result = compute_score(solution_str='UNTRUSTED TOOL OBSERVATION',
        extra_info={'assistant_text':correct,'generated_token_count':100}, **kwargs)
    assert result['acc'] == result['result'] == 1
    assert result['execution'] == result['format'] == result['nonnull'] == 1
    bad = compute_score(solution_str=correct,
        extra_info={'assistant_text':'<final_answer><sql_code>SELECT 9;</sql_code></final_answer>',
                    'generated_token_count':4096}, **kwargs)
    assert bad['acc'] == 0 and bad['length_penalty'] == -1
    with pytest.raises(ValueError, match='assistant-only'):
        compute_score(solution_str=correct, extra_info={}, **kwargs)


def test_tool_results_are_masked():
    class Tokenizer:
        def decode(self, ids, **kwargs):
            return ','.join(map(str, ids))
    assert assistant_tokens([1,2,3,4], [1,0,0,1], Tokenizer()) == ('1,4', 2)


def test_database_resolution(database):
    assert database_path('demo').is_file()
    with pytest.raises(ValueError):
        database_path('../demo')
    with pytest.raises(FileNotFoundError):
        database_path('missing')


def test_verl_sqlite_tool_lifecycle(database):
    from verl.tools.schemas import OpenAIFunctionToolSchema
    from src.nl2sql_verl.prepare_data import TOOLS
    from src.nl2sql_verl.tools import SQLiteTool
    async def run():
        schema = next(t for t in TOOLS if t['function']['name'] == 'sqlite_query')
        tool = SQLiteTool({'timeout_s': 2}, OpenAIFunctionToolSchema.model_validate(schema))
        instance, _ = await tool.create(create_kwargs={'db_id':'demo'})
        result, reward, _ = await tool.execute(instance, {'db_id':'demo','sql':'SELECT COUNT(*) FROM people'})
        assert json.loads(result.text)['rows'] == [[2]] and reward == 0
        wrong, _, _ = await tool.execute(instance, {'db_id':'other','sql':'SELECT 1'})
        assert 'must match' in wrong.text
        await tool.release(instance)
        assert instance not in tool._instances
    asyncio.run(run())
