import json
from pathlib import Path

import pytest

from src.nl2sql_verl.prepare_reference_data import (
    build_record, match_training, reference_messages, sql_tokens)
from src.nl2sql_verl.tool_profiles import ToolProfile, load_tool_profile


def example(question='Which item?', sql='SELECT item FROM items'):
    profile = load_tool_profile()
    user = (f'<question>{question}</question><hint>item means item</hint>'
            '<database_schema>items(item TEXT)</database_schema><db_id>example</db_id>')
    return {'db_id': 'example', 'question': question, 'evidence': 'item means item',
            'gold_sql': sql, 'tools': profile.declarations(),
            'prompt': profile.messages(user), 'messages': profile.messages(user)}


def test_raw_matching_preserves_order_duplicates_and_validates_gold():
    a, b = example('a'), example('b')
    raw = [dict(b, SQL=b['gold_sql']+';'), dict(a, SQL=a['gold_sql']),
           dict(a, SQL=a['gold_sql'])]
    matched = match_training(raw, [a, a, b])
    assert [i for _, i, _ in matched] == [2, 0, 1]
    with pytest.raises(ValueError, match='Gold SQL differs'):
        match_training([dict(a, SQL='SELECT secret FROM items')], [a])


def test_sql_whitespace_does_not_change_literal_comparison():
    assert sql_tokens('SELECT\n item FROM items;') == sql_tokens('SELECT item FROM items')
    assert sql_tokens("SELECT 'a  b'") != sql_tokens("SELECT 'a b'")


def test_reward_truth_is_metadata_and_prompt_contains_no_trajectory():
    row = example(sql='SELECT hidden_answer FROM secret')
    result = build_record(row, load_tool_profile(), 2, 9, 'train')
    assert [m['role'] for m in result['prompt']] == ['system', 'user']
    assert 'hidden_answer' not in json.dumps(result['prompt'])
    assert sql_tokens(json.loads(result['reward_model']['ground_truth'])['gold_sql']) == sql_tokens(row['gold_sql'])
    assert result['extra_info']['tools_kwargs']['sqlite_query']['create_kwargs'] == {'db_id': 'example'}


def test_custom_model_instructions_are_independent_of_semantic_tools(tmp_path):
    path = tmp_path/'system.txt'
    path.write_text('Use the native function calls provided by your chat template.')
    custom = load_tool_profile('custom', path)
    row = example()
    messages = reference_messages(row, custom)
    assert messages[0]['content'] == path.read_text()
    assert messages[1] == row['prompt'][1]
    assert custom.declarations() == load_tool_profile().declarations()
    assert 'call:' not in messages[0]['content']
    with pytest.raises(ValueError, match='requires'):
        load_tool_profile('custom')


def test_chat_template_receives_tools_instead_of_handwritten_wire_tokens():
    seen = {}

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            seen.update(messages=messages, **kwargs)
            return [1, 2, 3]

    profile = ToolProfile('future-model', 'model-specific instructions')
    messages = profile.messages('question')
    assert profile.prompt_length(Tokenizer(), messages) == 3
    assert seen['tools'] == profile.declarations()
    assert seen['add_generation_prompt'] is True


@pytest.mark.parametrize('filename', [
    'train-6601-schema-bare-tool.jsonl', 'old-dev-schema-tool-unpatched.jsonl'])
def test_supplied_reference_sample_is_reproduced(filename):
    path = Path('outputs')/filename
    if not path.exists():
        pytest.skip('Local reference JSONL is unavailable')
    with path.open() as stream:
        row = json.loads(stream.readline())
    normalized = build_record(row, load_tool_profile(), 0, 0, 'test')
    assert normalized['prompt'] == row['prompt']
    assert json.loads(normalized['reward_model']['ground_truth']) == {
        'db_id': row['db_id'], 'gold_sql': row['gold_sql']}
