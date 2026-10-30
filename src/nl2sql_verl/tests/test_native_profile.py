import asyncio
import json
import os
from pathlib import Path
import re
import sqlite3

from omegaconf import OmegaConf
import pytest

from src.nl2sql_verl.gemma_native_profile import format_tool_call
from src.nl2sql_verl.gemma_arguments import parse_arguments
from src.nl2sql_verl.prepare_native_data import corrected_user_prompt
from src.nl2sql_verl.tool_profiles import load_tool_profile


def test_every_native_example_roundtrips_through_actual_argument_parser():
    prompt = load_tool_profile('gemma4_native_v1').system_prompt
    assert not re.search(r'^call:', prompt, re.M)
    calls = re.findall(r'<\|tool_call>call:(\w+)\{(.*?)\}<tool_call\|>', prompt, re.S)
    assert len(calls) == 5
    for name, args in calls:
        parsed = parse_arguments(args)
        assert parsed['db_id'] == '<DBID>'
        if name == 'sqlite_query':
            assert isinstance(parsed['sql'], str)
        if name == 'sqlite_peek':
            assert parsed['columns'] == ['<COL1>']
    assert 'quoting the entire SQL string is optional' not in prompt


def test_native_serializer_handles_arrays_literals_and_rejects_unrepresentable_strings():
    values = {'db_id': 'example', 'sql': 'SELECT "a:b,{c}"', 'columns': ['a', 'b'], 'n': 3, 'ok': True, 'where': None}
    call = format_tool_call('sqlite_query', values)
    body = call.split('{',1)[1].rsplit('}',1)[0]
    assert parse_arguments(body) == values
    with pytest.raises(ValueError):
        format_tool_call('bad-name', {})
    with pytest.raises(ValueError):
        format_tool_call('sqlite_query', {'sql': '<|"|>'})


def test_implicit_composite_fk_uses_primary_key_order_and_preserves_non_schema_text(tmp_path):
    db = tmp_path/'example.sqlite'
    with sqlite3.connect(db) as c:
        c.execute('CREATE TABLE parent (x INT, y INT, PRIMARY KEY(y,x))')
        c.execute('CREATE TABLE child (a INT, b INT, FOREIGN KEY(a,b) REFERENCES parent)')
    user = 'Question parent.None\n<database_schema>child.a=parent.None\nchild.b=parent.None</database_schema>\nHint parent.None'
    result, fixes, _ = corrected_user_prompt(user, 'example', str(tmp_path))
    assert result == 'Question parent.None\n<database_schema>child.a=parent.y\nchild.b=parent.x</database_schema>\nHint parent.None'
    assert len(fixes) == 2


def test_real_native_agent_executes_error_repairs_and_masks_actual_observations(tmp_path, monkeypatch):
    from transformers import AutoTokenizer
    from verl.experimental.agent_loop.agent_loop import DictConfigWrap, ToolListWrap
    from verl.tools.tool_registry import initialize_tools_from_config
    from verl.utils.dataset.rl_dataset import RLHFDataset
    from verl.workers.rollout.replica import TokenOutput
    from src.nl2sql_verl.native_agent_loop import NL2SQLNativeAgentLoop
    from src.nl2sql_verl.reward_entry_native import compute_score
    model = Path(os.environ.get('NL2SQL_TEST_MODEL',
        '/home/ubuntu/nl2sql-checkpoints/dapo-fsdp-cp2-16gpu-train/global_step_5'))
    if not model.exists():
        pytest.skip('Integration test requires the local Gemma tokenizer')
    monkeypatch.setenv('NL2SQL_DATABASE_DIR', str(tmp_path))
    with sqlite3.connect(tmp_path/'example.sqlite') as c:
        c.execute('CREATE TABLE items (id INTEGER)')
        c.execute('INSERT INTO items VALUES (7)')
    tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
    profile = load_tool_profile('gemma4_native_v1')

    async def exercise():
        class Manager:
            prompts = []
            async def generate(self, request_id, prompt_ids, sampling_params, **kwargs):
                self.prompts.append(tok.decode(prompt_ids, skip_special_tokens=False))
                assert 0 < sampling_params['max_tokens'] <= 4096
                if len(self.prompts)>1:
                    assert sampling_params['max_tokens'] < 4096
                if len(self.prompts)==1:
                    text = '<scratch_pad>Verify.</scratch_pad>'+format_tool_call('sqlite_query', {'db_id':'example','sql':'SELECT missing FROM items'})
                elif len(self.prompts)==2:
                    assert 'no such column' in self.prompts[-1]
                    text = '<scratch_pad>Repair.</scratch_pad>'+format_tool_call('sqlite_query', {'db_id':'example','sql':'SELECT id FROM items'})
                else:
                    assert '[[7]]' in self.prompts[-1]
                    text = '<scratch_pad>Verified.</scratch_pad><relevant_tables>items</relevant_tables><relevant_columns>items.id</relevant_columns><final_answer><sql_code>SELECT id FROM items</sql_code></final_answer><turn|>'
                return TokenOutput(token_ids=tok.encode(text, add_special_tokens=False))
        cfg = OmegaConf.create({'actor_rollout_ref':{'rollout':{'full_determinism':False,'prompt_length':30720,'response_length':4096,'multi_turn':{'max_user_turns':None,'max_assistant_turns':9,'max_parallel_calls':3,'max_tool_response_length':512,'tool_response_truncate_side':'right','format':'nl2sql_gemma4'}}}})
        agent = NL2SQLNativeAgentLoop(trainer_config=DictConfigWrap(cfg), server_manager=Manager(), tokenizer=tok, processor=None,
            dataset_cls=RLHFDataset, data_config=DictConfigWrap(OmegaConf.create({})), hf_model_type='gemma4',
            tools=ToolListWrap(initialize_tools_from_config('src/nl2sql_verl/configs/tools_native.yaml')))
        return await agent.run({'temperature':1.2,'top_p':.95}, raw_prompt=profile.messages('Get ids'),
            extra_info={'tool_profile':profile.name}, tools_kwargs={'sqlite_query':{'create_kwargs':{'db_id':'example'}}})

    output = asyncio.run(exercise())
    extra = output.extra_fields
    assert extra['tool_execution_count']==2 and extra['tool_error_count']==1 and extra['tool_success_count']==1
    assert extra['assistant_generation_count']==3 and extra['final_sql_verified_exact']==1
    assert '[[7]]' not in extra['assistant_text']
    assert sum(output.response_mask)<len(output.response_mask)
    score = compute_score('nl2sql_bird', '', {'db_id':'example','gold_sql':'SELECT id FROM items'}, extra)
    assert score['format']==score['execution']==score['result']==1
    assert score['tool_execution_count']==2


def test_native_runtime_keeps_actual_traces_and_ignores_padding():
    from src.nl2sql_verl.native_agent_loop import METRICS
    from src.nl2sql_verl.native_runtime import records_from_extra
    fields={'reward_extra_info':{name:0 for name in METRICS},
            'assistant_text':'<|tool_call>call:sqlite_query{}<tool_call|>',
            'native_tool_events':[{'name':'sqlite_query','success':False,'observation':'no such column'}],
            'native_generation_trace':[{'last_token_id':49}]}
    fields['reward_extra_info']['tool_execution_count']=1
    rows=records_from_extra([fields,{}],['example_0_0','padding'],[{}, {'is_padding':True}])
    assert len(rows)==1 and rows[0]['metrics']['tool_execution_count']==1
    assert rows[0]['tool_events'][0]['observation']=='no such column'
    assert rows[0]['generation_trace'][0]['last_token_id']==49
    with pytest.raises(KeyError):
        records_from_extra([{}],['missing_0_0'],[{}])


def test_unfinished_native_rollout_cannot_score_fenced_draft_sql(tmp_path,monkeypatch):
    from src.nl2sql_verl.native_agent_loop import METRICS
    from src.nl2sql_verl.reward_entry_native import compute_score
    monkeypatch.setenv('NL2SQL_DATABASE_DIR',str(tmp_path))
    with sqlite3.connect(tmp_path/'example.sqlite') as c:
        c.execute('CREATE TABLE items (id INTEGER)')
        c.execute('INSERT INTO items VALUES (7)')
    extra={name:0 for name in METRICS}
    extra.update(assistant_text='<scratch_pad>```sql\nSELECT id FROM items\n```</scratch_pad>'+format_tool_call(
        'sqlite_query',{'db_id':'example','sql':'SELECT id FROM items'}),generated_token_count=40)
    result=compute_score('nl2sql_bird','',{'db_id':'example','gold_sql':'SELECT id FROM items'},extra)
    assert all(result[name]==0 for name in ['format','execution','result','nonnull','table_linking','column_linking','score'])
