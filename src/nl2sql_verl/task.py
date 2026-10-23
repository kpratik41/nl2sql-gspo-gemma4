"""Task boundary shared by the verl agent, tools and terminal reward."""
import json
import os
from functools import lru_cache
from pathlib import Path

from .consensus.rewards import make_nl2sql_rewards

REWARD_NAMES = ('format', 'execution', 'result', 'table_linking', 'column_linking', 'nonnull', 'length_penalty')
REWARD_WEIGHTS = (0.2, 0.5, 2.0, 0.5, 0.5, 0.1, 0.1)


def database_path(db_id, root=None):
    """Resolve only an example's database beneath configured local roots."""
    if not isinstance(db_id, str) or not db_id or Path(db_id).name != db_id or db_id in {'.', '..'}:
        raise ValueError('db_id must be a database identifier, not a path')
    base = Path(root or os.environ.get('NL2SQL_DATABASE_DIR', 'databases')).resolve()
    for directory in (base, base/'train_databases', base/'dev_databases'):
        for name in (f'{db_id}/{db_id}.sqlite', f'{db_id}.sqlite', f'{db_id}/{db_id}.db', f'{db_id}.db'):
            path = (directory/name).resolve()
            if path.is_relative_to(base) and path.is_file():
                return path
    raise FileNotFoundError(f'No SQLite database for {db_id!r} below {base}')


def assistant_tokens(response_ids, response_mask, tokenizer):
    if len(response_ids) != len(response_mask):
        raise ValueError('response token/mask lengths differ')
    ids = [int(t) for t, m in zip(response_ids, response_mask, strict=True) if m]
    return tokenizer.decode(ids, skip_special_tokens=False), len(ids)


@lru_cache(maxsize=4)
def _rewards(root, timeout):
    return make_nl2sql_rewards(root, exec_timeout_s=timeout, gold_timeout_s=timeout, reward_workers=1)


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if data_source != 'nl2sql_bird':
        raise ValueError(f'Unexpected data source: {data_source}')
    truth = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    extra = extra_info or {}
    # Decoded mixed trajectories contain tool observations; never score those.
    if 'assistant_text' not in extra or 'generated_token_count' not in extra:
        raise ValueError('Use NL2SQLToolAgentLoop: assistant-only text/token count is missing')
    text = extra['assistant_text']
    db_id, gold = truth['db_id'], truth['gold_sql']
    path = database_path(db_id)
    # Each branch helper expects root/db_id/db_id.sqlite or root/db_id.sqlite.
    root = str(path.parent.parent if path.parent.name == db_id else path.parent)
    functions = _rewards(root, float(os.environ.get('NL2SQL_EXEC_TIMEOUT_S', '60')))
    values = [f([text], db_id=[db_id], gold_sql=[gold])[0] for f in functions[:-1]]
    maximum = int(os.environ.get('NL2SQL_LENGTH_MAX', '4096'))
    buffer = int(os.environ.get('NL2SQL_LENGTH_BUFFER', '512'))
    if maximum <= 0 or not 0 < buffer <= maximum:
        raise ValueError('Require 0 < length penalty buffer <= maximum')
    penalty = min(0., max(-1., (maximum-buffer-int(extra['generated_token_count']))/buffer))
    values.append(penalty)
    weights = tuple(float(x) for x in os.environ.get('NL2SQL_REWARD_WEIGHTS', '').split(',')) if os.environ.get('NL2SQL_REWARD_WEIGHTS') else REWARD_WEIGHTS
    if len(weights) != len(values):
        raise ValueError('Expected seven reward weights')
    components = dict(zip(REWARD_NAMES, map(float, values), strict=True))
    return {'score': sum(w*v for w, v in zip(weights, values, strict=True)),
            'acc': float(components['result']), **components}
