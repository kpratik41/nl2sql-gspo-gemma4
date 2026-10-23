"""Parse Gemma native argument literals, including array-valued tool parameters."""
import json
import re

TOKEN = re.compile(r'\s*(?:<\|"\|>(.*?)<\|"\|>|(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)|([A-Za-z_]\w*)|([{}\[\]:,]))', re.S)


def parse_arguments(text):
    tokens, offset = [], 0
    while offset < len(text):
        if not text[offset:].strip():
            break
        match = TOKEN.match(text, offset)
        if match is None:
            raise ValueError(f'Invalid Gemma argument literal at character {offset}')
        string, number, word, punctuation = match.groups()
        tokens.append(json.dumps(string, ensure_ascii=False) if string is not None else
                      number if number is not None else word if word in {'true', 'false', 'null'} else
                      json.dumps(word) if word is not None else punctuation)
        offset = match.end()
    result = json.loads('{'+''.join(tokens)+'}')
    if not isinstance(result, dict):
        raise ValueError('Tool arguments must be an object')
    return result
