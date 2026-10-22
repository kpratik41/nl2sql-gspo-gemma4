import csv
import json
import re

from rank_bm25 import BM25Okapi

from .database import EXTENDED_STATS_VERSION, distribution_stats, quote


def tool(name, description, properties, required):
    return {'type': 'function', 'function': {'name': name, 'description': description,
            'parameters': {'type': 'object', 'properties': properties, 'required': required,
                           'additionalProperties': False}}}


STRING = {'type': 'string'}
TOOLS = [
    tool('sqlite_query', 'Execute an actual read-only SQLite SELECT or WITH query to verify syntax and results. Return columns, rows, and truncation status. No LIMIT is added to the SQL.',
         {'db_id': STRING, 'sql': STRING, 'max_return_rows': {'type': 'integer', 'minimum': 1, 'maximum': 100, 'default': 10}}, ['db_id', 'sql']),
    tool('sqlite_peek', 'Inspect distinct column samples, declared types, NULL counts, distinct counts, ranges, and numeric means. Unfiltered profiles reuse cached full-table statistics. Optional WHERE restricts the preview and profile.',
         {'db_id': STRING, 'table': STRING, 'columns': {'type': 'array', 'items': STRING, 'minItems': 1},
          'limit': {'type': 'integer', 'minimum': 1, 'maximum': 20, 'default': 10}, 'where': STRING}, ['db_id', 'table', 'columns']),
    tool('bm_25_search', 'Find exact stored text values in a SQLite column using BM25 plus substring/word matches. Returns up to top_k matches per method, deduplicated. Use to verify name, location, status, or category literals.',
         {'db_id': STRING, 'table': STRING, 'column': STRING, 'query': STRING,
          'top_k': {'type': 'integer', 'minimum': 1, 'maximum': 20, 'default': 10}, 'where': STRING}, ['db_id', 'table', 'column', 'query']),
]


def tokenize(text):
    return re.findall(r'\w+', text.lower())


def resolve_identifier(value, names, kind):
    """Resolve exact/case-insensitive names and balanced SQLite quoting only."""
    if isinstance(value, str):
        # Preserve literal names containing quotes or surrounding whitespace.
        if value in names:
            return value
        clean = value.strip()
        if len(clean) >= 2 and (clean[0], clean[-1]) in [('"', '"'), ('`', '`'), ('[', ']')]:
            closing = clean[-1]
            clean = clean[1:-1].replace(closing * 2, closing)
        matches = [name for name in names if name.lower() == clean.lower()]
        if len(matches) == 1:
            return matches[0]
    raise ValueError(f'Unknown {kind}: {value!r}. Valid {kind} names: {list(names)!r}')


def resolve_columns(value, names):
    """Accept unambiguous serialized column lists, then validate every name."""
    if isinstance(value, str):
        # A real column name containing a comma must remain one identifier.
        try:
            return [resolve_identifier(value, names, 'column')]
        except ValueError:
            pass
        try:
            decoded = json.loads(value)
        except (ValueError, TypeError):
            decoded = None
        value = decoded if isinstance(decoded, list) else next(csv.reader([value], skipinitialspace=True))
    if not isinstance(value, list) or not value:
        raise ValueError('columns must be a non-empty list of schema column names')
    if value == ['*'] and '*' not in names:
        return list(names)
    return list(dict.fromkeys(resolve_identifier(c, names, 'column') for c in value))


class ToolRunner:
    def __init__(self, database, db_id, stats_profile='baseline'):
        self.database, self.db_id, self.stats_profile = database, db_id, stats_profile

    def call(self, name, args):
        try:
            definitions = {t['function']['name']: t['function']['parameters'] for t in TOOLS}
            if name not in definitions:
                raise ValueError(f'Unknown tool: {name}')
            if not isinstance(args, dict):
                raise ValueError('Arguments must be an object')
            definition = definitions[name]
            if set(args) - set(definition['properties']) or set(definition['required']) - set(args):
                raise ValueError('Unexpected or missing arguments; follow the tool schema')
            if args.get('db_id') != self.db_id:
                raise ValueError('Tools can only access the current question database')
            if name == 'sqlite_query':
                return self.database.query(self.db_id, args['sql'], max_rows=max(1, min(100, int(args.get('max_return_rows', 10)))))
            schema = self.database.schema(self.db_id, self.stats_profile)
            table = resolve_identifier(args['table'], schema['tables'], 'table')
            names = schema['tables'][table]['columns']
            cols = resolve_columns(args['columns'], names) if name == 'sqlite_peek' else [resolve_identifier(args['column'], names, f'column in {table}')]
            normalized = {**args, 'table': table}
            normalized.update({'columns': cols} if name == 'sqlite_peek' else {'column': cols[0]})
            return self.peek(normalized, schema) if name == 'sqlite_peek' else self.search(normalized)
        except Exception as exc:
            return {'error': type(exc).__name__, 'message': str(exc)}

    def peek(self, args, schema):
        table = args['table']
        where = f" WHERE ({args['where']})" if args.get('where') else ''
        limit = max(1, min(20, int(args.get('limit', 10))))
        def build():
            out = {}
            for col in args['columns']:
                cq, tq = quote(col), quote(table)
                rows = self.database.query(self.db_id, f'SELECT DISTINCT {cq} FROM {tq}{where} ORDER BY {cq} LIMIT {limit}')['rows']
                stats = schema['tables'][table]['columns'][col].copy()
                if where:
                    values = self.database.query(self.db_id, f'SELECT COUNT(*), COUNT({cq}), COUNT(DISTINCT {cq}), MIN({cq}), MAX({cq}), AVG(CASE WHEN typeof({cq}) IN (\'integer\',\'real\') THEN {cq} END) FROM {tq}{where}')['rows'][0]
                    stats = {'rows': values[0], 'nulls': values[0]-values[1], 'non_nulls': values[1], 'distinct': values[2], 'min': values[3], 'max': values[4], 'numeric_mean': values[5], 'type': stats['type']}
                    if self.stats_profile == 'extended':
                        conn = self.database.connect(self.db_id)
                        try:
                            stats.update(distribution_stats(conn, table, col, stats, args['where']))
                        finally:
                            conn.close()
                out[col] = {'samples': [r[0] for r in rows], 'profile': stats}
            return {'table': table, 'profile_scope': 'full filtered table' if where else 'full table', 'columns': out}
        cache_args = args if self.stats_profile == 'baseline' else {'args': args, 'stats_profile': self.stats_profile, 'version': EXTENDED_STATS_VERSION}
        return self.database.cached(self.db_id, 'peek', cache_args, build)

    def search(self, args):
        query = args['query']
        if not isinstance(query, str) or not query.strip():
            raise ValueError('Query must be non-empty')
        table, col = args['table'], args['column']
        where = f" AND ({args['where']})" if args.get('where') else ''
        def build():
            sql = f'SELECT DISTINCT {quote(col)} FROM {quote(table)} WHERE {quote(col)} IS NOT NULL{where} ORDER BY {quote(col)}'
            result = self.database.query(self.db_id, sql, max_rows=100000)
            return {'values': [str(r[0]) for r in result['rows']], 'corpus_truncated': result['truncated']}
        corpus = self.database.cached(self.db_id, 'values', [table, col, where], build)
        values = corpus['values']
        k = max(1, min(20, int(args.get('top_k', 10))))
        tokens = tokenize(query)
        documents = [tokenize(v) or [''] for v in values]
        if not values:
            return {'values': [], 'corpus_truncated': False}
        scores = BM25Okapi(documents).get_scores(tokens)
        ranked = sorted(range(len(values)), key=lambda i: (-scores[i], i))[:k]
        matches = [(sum(2 * (t in v.lower()) + 3 * bool(re.search(r'\b' + re.escape(t) + r'\b', v, re.I)) + 1.5 * v.lower().startswith(t) for t in tokens), i) for i, v in enumerate(values)]
        ranked += [i for score, i in sorted(matches, key=lambda x: (-x[0], len(values[x[1]]), x[1]))[:k] if score > 0]
        return {'values': [values[i] for i in dict.fromkeys(ranked)], 'corpus_truncated': corpus['corpus_truncated']}
