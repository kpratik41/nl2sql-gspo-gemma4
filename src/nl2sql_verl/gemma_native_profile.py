"""Gemma-specific wire examples; semantic tool schemas stay model-independent."""
import math
import re

from .consensus.prompts import SYSTEM_PROMPT_TEMPLATES
from .consensus.tool_calling import tool_catalog_compact


def argument_literal(value):
    if isinstance(value, str):
        if '<|"|>' in value:
            raise ValueError('Gemma delimiter inside a string is not representable')
        return '<|"|>'+value+'<|"|>'
    if value is None:
        return 'null'
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)) and math.isfinite(value):
        return str(value)
    if isinstance(value, list):
        return '['+','.join(argument_literal(x) for x in value)+']'
    if isinstance(value, dict):
        if any(not re.fullmatch(r'[A-Za-z_]\w*', k) for k in value):
            raise ValueError('Gemma argument keys must be identifiers')
        return '{'+','.join(k+':'+argument_literal(v) for k, v in value.items())+'}'
    raise ValueError(f'Unsupported Gemma argument type: {type(value)}')


def format_tool_call(name, arguments):
    if not re.fullmatch(r'[A-Za-z_]\w*', name):
        raise ValueError('Tool name must be an identifier')
    return '<|tool_call>call:'+name+argument_literal(arguments)+'<tool_call|>'


def system_prompt():
    """Retain task/SQL instructions, replacing the old incompatible wire contract."""
    prompt = SYSTEM_PROMPT_TEMPLATES.replace('{TOOL_CATALOG_COMPACT}', tool_catalog_compact()).strip()
    before, rest = prompt.split('Native tool-call syntax is mandatory:', 1)
    _, after = rest.split('Schema linking requirement:', 1)
    protocol = '''Native Gemma tool-call syntax is mandatory:
- Every call starts with the actual special token <|tool_call> and ends with <tool_call|>.
- Between those tokens emit call:tool_name{key:value,...}.
- Delimit EVERY string value, including db_id, table, SQL and each string in an array, with <|"|> on both sides.
- Integers, booleans, null and arrays keep their native types. Do not put SQL in JSON double quotes.
- Emit <tool_call|> immediately after the argument object, then wait for the actual tool observation.
- Use at most one tool call per assistant generation. Never simulate tool results or continue reasoning after a call in the same generation.
- A bare call expression without the special delimiters does not execute a tool.
- Before finalizing, execute the exact final SQL with sqlite_query and use its actual observation. Repair errors and retry.
- Keep reasoning compact so the complete interaction, including tool observations, fits the 4,096-token response budget.

'''
    prompt = before+protocol+'Schema linking requirement:'+after
    lines = []
    for line in prompt.splitlines():
        if line.startswith('call:sqlite_query{'):
            match = re.fullmatch(r'call:sqlite_query\{db_id:<DBID>,sql:(.*),max_return_rows:10\}', line)
            if match is None:
                raise ValueError('Unrecognized legacy SQLite example')
            line = format_tool_call('sqlite_query', {'db_id': '<DBID>', 'sql': match[1], 'max_return_rows': 10})
        elif line.startswith('call:bm25_search_sqlite{'):
            line = format_tool_call('bm25_search_sqlite', {'db_id': '<DBID>', 'table': '<TABLE>',
                                    'column': '<COLUMN>', 'query': '<SEARCH_TERM>', 'top_k': 5})
        elif line.startswith('call:sqlite_peek{'):
            line = format_tool_call('sqlite_peek', {'db_id': '<DBID>', 'table': '<TABLE>',
                                    'columns': ['<COL1>'], 'limit': 20})
        elif line.startswith('call:'):
            raise ValueError('Unrecognized bare tool example')
        lines.append(line)
    return '\n'.join(lines)
