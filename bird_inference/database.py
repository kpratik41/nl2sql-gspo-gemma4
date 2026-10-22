import csv
import fcntl
import hashlib
import json
import math
import os
import re
import sqlite3
import time
from pathlib import Path

DEFAULT_ROOTS = [Path('/home/ec2-user/nl2sql-gspo-gemma4/databases') / f'{s}_databases'
                 for s in ('dev', 'train')]
CACHE_VERSION = 2
EXTENDED_STATS_VERSION = 1


def quote(name):
    return '"' + name.replace('"', '""') + '"'


def json_default(value):
    if isinstance(value, bytes):
        return {'blob_hex': value.hex()}
    raise TypeError(type(value).__name__)


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=json_default) + '\n')
    tmp.replace(path)


class Database:
    def __init__(self, roots=None, cache='cache'):
        self.roots = [Path(x) for x in (roots or os.environ.get('BIRD_DB_ROOTS', '').split(os.pathsep)) if x] or DEFAULT_ROOTS
        self.cache = Path(cache)

    def resolve(self, db_id):
        if not re.fullmatch(r'[A-Za-z0-9_-]+', db_id):
            raise ValueError('Invalid database identifier')
        for root in self.roots:
            for path in (root / db_id / f'{db_id}.sqlite', root / f'{db_id}.sqlite'):
                if path.is_file():
                    return path.resolve()
        raise FileNotFoundError(f'Database not found: {db_id}')

    def fingerprint(self, db_id):
        path = self.resolve(db_id)
        files = [path, Path(str(path) + '-wal')] + sorted((path.parent / 'database_description').glob('*.csv'))
        facts = []
        for p in files:
            if not p.exists():
                continue
            stat = p.stat()
            # SQLite's header change counter catches same-size updates even on
            # filesystems with coarse timestamps. Include WAL tail checksums too.
            with p.open('rb') as f:
                header = f.read(100)
                f.seek(max(0, stat.st_size - 32))
                tail = f.read(32)
            facts.append((str(p), stat.st_ino, stat.st_size, stat.st_mtime_ns,
                          stat.st_ctime_ns, header.hex(), tail.hex()))
        return hashlib.sha256(json.dumps([CACHE_VERSION, facts]).encode()).hexdigest()

    def cached(self, db_id, namespace, params, build):
        identity = [self.fingerprint(db_id), params]
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        path = self.cache / namespace / f'{db_id}-{key}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if path.exists():
                return json.loads(path.read_text())
            value = build()
            atomic_json(path, value)
            # Return the same representation on cache misses and hits (e.g. tuples/BLOBs).
            return json.loads(path.read_text())

    def connect(self, db_id, timeout=45):
        conn = sqlite3.connect(self.resolve(db_id).as_uri() + '?mode=ro', uri=True, timeout=5)
        conn.execute('PRAGMA query_only=ON')
        start = time.monotonic()
        conn.set_progress_handler(lambda: int(time.monotonic() - start > timeout), 10000)
        return conn

    def query(self, db_id, sql, max_rows=100, timeout=45):
        if not isinstance(sql, str) or not re.match(r'^\s*(SELECT|WITH)\b', sql, re.I):
            raise ValueError('Only SELECT or WITH queries are allowed')
        conn = self.connect(db_id, timeout)
        try:
            cur = conn.execute(sql)
            rows = cur.fetchall() if max_rows is None else cur.fetchmany(max_rows + 1)
            truncated = max_rows is not None and len(rows) > max_rows
            return {'columns': [x[0] for x in cur.description],
                    'rows': rows[:max_rows] if truncated else rows, 'truncated': truncated}
        finally:
            conn.close()

    def schema(self, db_id, stats_profile='baseline'):
        if stats_profile == 'extended':
            return self.cached(db_id, 'schema_extended', {'version': EXTENDED_STATS_VERSION},
                               lambda: self._extended_schema(db_id))
        if stats_profile != 'baseline':
            raise ValueError(f'Unknown statistics profile: {stats_profile}')
        return self.cached(db_id, 'schema', {}, lambda: self._schema(db_id))

    def _extended_schema(self, db_id):
        # schema() reads JSON afresh, so enriching this object cannot mutate the
        # baseline cache. Separate namespace/version preserves old experiments.
        schema = self.schema(db_id)
        conn = self.connect(db_id, timeout=1800)
        try:
            for table, info in schema['tables'].items():
                for col, stats in info['columns'].items():
                    stats.update(distribution_stats(conn, table, col, stats))
        finally:
            conn.close()
        schema['stats_profile'] = 'extended'
        schema['extended_stats_version'] = EXTENDED_STATS_VERSION
        return schema

    def _schema(self, db_id):
        conn = self.connect(db_id, timeout=600)
        try:
            tables = {}
            names = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
            for table in names:
                tq = quote(table)
                descriptions = {}
                desc = self.resolve(db_id).parent / 'database_description' / f'{table}.csv'
                if desc.exists():
                    with desc.open(encoding='utf-8-sig', errors='replace', newline='') as f:
                        for row in csv.DictReader(f):
                            row = {k.strip(): (v or '').strip() for k, v in row.items() if k}
                            name = row.get('original_column_name', '')
                            descriptions[name] = ' '.join(row.get(k, '') for k in ('column_description', 'value_description')).strip()
                total = conn.execute(f'SELECT COUNT(*) FROM {tq}').fetchone()[0]
                columns = {}
                for _, col, kind, notnull, default, pk in conn.execute(f'PRAGMA table_info({tq})').fetchall():
                    cq = quote(col)
                    nonnull, distinct, low, high, mean = conn.execute(
                        f'SELECT COUNT({cq}), COUNT(DISTINCT {cq}), MIN({cq}), MAX({cq}), '
                        f"AVG(CASE WHEN typeof({cq}) IN ('integer','real') THEN {cq} END) FROM {tq}").fetchone()
                    top = conn.execute(f'SELECT {cq}, COUNT(*) AS n FROM {tq} WHERE {cq} IS NOT NULL GROUP BY {cq} ORDER BY n DESC, {cq} LIMIT 3').fetchall()
                    columns[col] = {'type': kind, 'primary_key': bool(pk), 'not_null_declared': bool(notnull),
                                    'default': default, 'description': descriptions.get(col, ''), 'rows': total,
                                    'nulls': total-nonnull, 'non_nulls': nonnull, 'distinct': distinct,
                                    'min': low, 'max': high, 'numeric_mean': mean, 'top_values': top}
                fks = [{'column': r[3], 'ref_table': r[2], 'ref_column': r[4]} for r in conn.execute(f'PRAGMA foreign_key_list({tq})')]
                tables[table] = {'rows': total, 'columns': columns, 'foreign_keys': fks}
            return {'db_id': db_id, 'fingerprint': self.fingerprint(db_id), 'stats_scope': 'full database, exact', 'tables': tables}
        finally:
            conn.close()


def distribution_stats(conn, table, column, stats, where=None):
    """Full-population distributions; numeric TEXT is deliberately not cast.

    Quartiles use linear interpolation at (n-1)*p, including duplicates and
    excluding NULL/non-numeric storage values. Mode covers all non-NULL values
    using SQLite GROUP BY semantics and reports a representative plus tie count.
    """
    tq, cq = quote(table), quote(column)
    scope = f'({where})' if where else '1'
    storage = dict(conn.execute(f'SELECT typeof({cq}), COUNT(*) FROM {tq} WHERE {scope} GROUP BY typeof({cq})'))
    n = storage.get('integer', 0) + storage.get('real', 0)
    extra = {'storage_counts': storage, 'numeric_count': n,
             'null_fraction': stats['nulls'] / stats['rows'] if stats['rows'] else None,
             'numeric_median': None, 'numeric_q25': None, 'numeric_q75': None,
             'numeric_stddev_population': None, 'mode': None}
    if stats['non_nulls']:
        # Reuse baseline full-table frequencies when available; filtered tool
        # profiles calculate their own mode rather than reusing global values.
        top = stats.get('top_values')
        if top:
            value, frequency = top[0]
        else:
            value, frequency = conn.execute(
                f'SELECT {cq}, COUNT(*) AS n FROM {tq} WHERE {scope} AND {cq} IS NOT NULL '
                f'GROUP BY {cq} ORDER BY n DESC, {cq} LIMIT 1').fetchone()
        if frequency == 1:
            ties = stats['distinct']
        elif stats['distinct'] == 1:
            ties = 1
        else:
            ties = conn.execute(f'SELECT COUNT(*) FROM (SELECT COUNT(*) AS n FROM {tq} '
                                f'WHERE {scope} AND {cq} IS NOT NULL GROUP BY {cq}) WHERE n = ?',
                                (frequency,)).fetchone()[0]
        extra['mode'] = {'value': value, 'frequency': frequency, 'tied_values': ties,
                         'non_null_fraction': frequency / stats['non_nulls']}
    if n:
        positions = {p: (n - 1) * p for p in (.25, .5, .75)}
        ranks = sorted({r for pos in positions.values() for r in (math.floor(pos), math.ceil(pos))})
        numeric = f"{scope} AND typeof({cq}) IN ('integer','real')"
        ordered = dict(conn.execute(
            f'WITH ordered AS (SELECT {cq} AS value, ROW_NUMBER() OVER (ORDER BY {cq}) - 1 AS position '
            f'FROM {tq} WHERE {numeric}) SELECT position, value FROM ordered '
            f'WHERE position IN ({",".join("?" for _ in ranks)})', ranks))
        for p, field in [(.25, 'numeric_q25'), (.5, 'numeric_median'), (.75, 'numeric_q75')]:
            pos = positions[p]; low, high = math.floor(pos), math.ceil(pos)
            extra[field] = ordered[low] if low == high else ((1 - (pos - low)) * ordered[low] + (pos - low) * ordered[high])
        mean = stats['numeric_mean']
        if mean is not None and math.isfinite(mean):
            variance = conn.execute(f'SELECT AVG(({cq} - ?) * ({cq} - ?)) FROM {tq} WHERE {numeric}',
                                    (mean, mean)).fetchone()[0]
            if variance is not None and math.isfinite(variance):
                extra['numeric_stddev_population'] = math.sqrt(max(0, variance))
    return extra


def display(value, max_chars=160):
    text = json.dumps(value, ensure_ascii=False, default=json_default)
    return text if len(text) <= max_chars else text[:max_chars] + '… [display truncated]'


def mschema(schema):
    lines = [f"【DB_ID】 {schema['db_id']}", '【Schema】', 'Column statistics cover the full table; distinct excludes NULL. Numeric means include numeric SQLite storage values only.']
    if schema.get('stats_profile') == 'extended':
        lines.append('Numeric median/quartiles/stddev include integer/real storage values only, not numeric-looking TEXT or NULL. Quartiles use linear interpolation at (n-1)*p; stddev is population stddev. Mode excludes NULL; when tied, one representative and the number of tied values are shown. These are whole-table statistics, not answers to question-specific filtered aggregates.')
    fks = []
    for table, info in schema['tables'].items():
        lines += [f"# Table: {table}, Rows: {info['rows']}", '[']
        for col, c in info['columns'].items():
            details = [c['type'] or 'UNTYPED']
            if c['primary_key']:
                details.append('Primary Key')
            if c['not_null_declared']:
                details.append('NOT NULL')
            if c['description']:
                details.append(c['description'])
            details += [f"Nulls: {c['nulls']}", f"Non-null: {c['non_nulls']}", f"Distinct: {c['distinct']}",
                        f"Min: {display(c['min'])}", f"Max: {display(c['max'])}"]
            if c['numeric_mean'] is not None:
                details.append(f"Numeric mean: {c['numeric_mean']:.6g}")
            if 'numeric_count' in c:
                details.append(f"Storage counts: {display(c['storage_counts'])}")
                details.append(f"Numeric count: {c['numeric_count']}")
                if c['null_fraction'] is not None:
                    details.append(f"NULL fraction: {c['null_fraction']:.6g}")
                if c['numeric_count']:
                    for key, label in [('numeric_median', 'Numeric median'), ('numeric_q25', 'Numeric Q25'),
                                       ('numeric_q75', 'Numeric Q75'), ('numeric_stddev_population', 'Numeric stddev (population)')]:
                        details.append(f"{label}: {display(c[key])}")
                mode = c['mode']
                if mode is None:
                    details.append('Mode: none (no non-NULL values)')
                else:
                    details.append(f"Mode representative: {display(mode['value'])}; frequency: {mode['frequency']}; tied values: {mode['tied_values']}; non-null fraction: {mode['non_null_fraction']:.6g}")
            details.append('Top values (value, count): ' + display(c['top_values'], 240))
            lines.append(f"({col}, " + ', '.join(details) + '),')
        lines.append(']')
        for fk in info['foreign_keys']:
            fks.append(f"{table}.{fk['column']}={fk['ref_table']}.{fk['ref_column']}")
    return '\n'.join(lines + ['【Foreign keys】'] + fks)
