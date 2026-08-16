import os
import re
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Tuple


SQL_BLOCK_RE = re.compile(r"```sql\s*(.*?)```", re.IGNORECASE | re.DOTALL)
SQL_FENCE_RE = re.compile(r"```\s*(.*?)```", re.IGNORECASE | re.DOTALL)
SQL_CODE_TAG_RE = re.compile(r"<sql_code>\s*(.*?)\s*</sql_code>", re.IGNORECASE | re.DOTALL)
FINAL_ANSWER_TAG_RE = re.compile(r"<final_answer>\s*(.*?)\s*</final_answer>", re.IGNORECASE | re.DOTALL)
SQL_START_RE = re.compile(r"\b(SELECT|WITH)\b", re.IGNORECASE)

from nl2sql_gspo.tool_dialects import (  # noqa: E402
    GEMMA_TOOL_CALL_MARKER_RE as TOOL_CALL_MARKER_RE,  # back-compat alias
    contains_any_tool_call_marker,
)

BAD_SQL_RE = re.compile(
    r"\b(DROP|ALTER|TRUNCATE|ATTACH|DETACH|PRAGMA|VACUUM|REINDEX|CREATE|INSERT|UPDATE|DELETE)\b",
    re.IGNORECASE,
)

_DB_LOCK = threading.Lock()
_DB_CONNECTIONS: Dict[str, Optional[sqlite3.Connection]] = {}


def extract_completion_text(completion: Any) -> str:
    if completion is None:
        return ""

    if isinstance(completion, list):
        return "\n".join(
            str(m.get("content", ""))
            for m in completion
            if isinstance(m, dict)
        )

    if isinstance(completion, dict):
        return str(completion.get("content", ""))

    return str(completion)


def clean_sql(sql: str) -> str:
    sql = sql.strip()
    sql = re.sub(r"^\s*SQL\s*:\s*", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"</?(scratch_pad|final_answer|sql_code)>", "", sql, flags=re.IGNORECASE)

    stop_markers = [
        "\n\nExplanation:",
        "\nExplanation:",
        "\nThe query",
        "\nThis query",
    ]

    for marker in stop_markers:
        idx = sql.lower().find(marker.lower())
        if idx != -1:
            sql = sql[:idx].strip()

    sql = sql.rstrip(";").strip()
    if sql:
        sql += ";"

    return sql


def _last_match(pattern: re.Pattern[str], text: str) -> Optional[re.Match[str]]:
    match = None
    for match in pattern.finditer(text):
        pass
    return match


def extract_final_answer_sql(completion: Any) -> str:
    """Extract SQL only from the final-answer contract used by tool prompts.

    This intentionally ignores draft SQL in ``<scratch_pad>`` and SQL embedded
    inside tool calls/responses. During tool-call RL a rollout that stops after
    a tool call should not receive execution/result reward for an unfinished
    candidate query.
    """

    text = extract_completion_text(completion).strip()
    if not text:
        return ""

    final_answer = _last_match(FINAL_ANSWER_TAG_RE, text)
    if not final_answer:
        return ""

    final_answer_text = final_answer.group(1).strip()
    tagged_sql = _last_match(SQL_CODE_TAG_RE, final_answer_text)
    if tagged_sql:
        return clean_sql(tagged_sql.group(1))

    sql_start = SQL_START_RE.search(final_answer_text)
    if sql_start:
        return clean_sql(final_answer_text[sql_start.start():])

    return ""


def extract_sql(completion: Any, *, prefer_final_answer: bool = True) -> str:
    text = extract_completion_text(completion).strip()

    if not text:
        return ""

    if prefer_final_answer:
        final_sql = extract_final_answer_sql(text)
        if final_sql:
            return final_sql

    m = _last_match(SQL_CODE_TAG_RE, text)
    if m:
        return clean_sql(m.group(1))

    m = _last_match(FINAL_ANSWER_TAG_RE, text)
    if m:
        final_answer_text = m.group(1).strip()

        tagged_sql = _last_match(SQL_CODE_TAG_RE, final_answer_text)
        if tagged_sql:
            return clean_sql(tagged_sql.group(1))

        m2 = SQL_START_RE.search(final_answer_text)
        if m2:
            return clean_sql(final_answer_text[m2.start():])

    m = SQL_BLOCK_RE.search(text)
    if m:
        return clean_sql(m.group(1))

    m = SQL_FENCE_RE.search(text)
    if m:
        candidate = m.group(1)
        if SQL_START_RE.search(candidate):
            return clean_sql(candidate)

    # If the completion contains tool-call syntax but no final-answer SQL, it
    # is an unfinished agentic rollout. Do not reward a draft CandidateSQL from
    # the scratchpad or a SQL argument inside a tool call as the final answer.
    if contains_any_tool_call_marker(text):
        return ""

    m = SQL_START_RE.search(text)
    if m:
        return clean_sql(text[m.start():])

    return clean_sql(text)


def is_safe_readonly_sql(sql: str) -> bool:
    if not sql:
        return False

    if BAD_SQL_RE.search(sql):
        return False

    return bool(SQL_START_RE.search(sql))


def get_database_path(db_id: str, database_dir: str) -> str:
    if not db_id:
        return ""

    candidates = [
        os.path.join(database_dir, db_id, f"{db_id}.sqlite"),
        os.path.join(database_dir, db_id, f"{db_id}.db"),
        os.path.join(database_dir, f"{db_id}.sqlite"),
        os.path.join(database_dir, f"{db_id}.db"),
        os.path.join(database_dir, "train_databases", db_id, f"{db_id}.sqlite"),
        os.path.join(database_dir, "train_databases", db_id, f"{db_id}.db"),
        os.path.join(database_dir, "dev_databases", db_id, f"{db_id}.sqlite"),
        os.path.join(database_dir, "dev_databases", db_id, f"{db_id}.db"),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    return ""


def get_db_connection(db_path: str) -> Optional[sqlite3.Connection]:
    if not db_path or not os.path.exists(db_path):
        return None

    with _DB_LOCK:
        if db_path in _DB_CONNECTIONS:
            return _DB_CONNECTIONS[db_path]

        try:
            uri = f"file:{db_path}?mode=ro"
            conn = sqlite3.connect(
                uri,
                uri=True,
                check_same_thread=False,
                timeout=30,
            )
            conn.execute("PRAGMA query_only = ON")
            conn.execute("PRAGMA cache_size = -2000")
            _DB_CONNECTIONS[db_path] = conn
            return conn
        except Exception:
            _DB_CONNECTIONS[db_path] = None
            return None


def normalize_cell(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, float):
        return round(value, 6)

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    if isinstance(value, str):
        return value.strip().lower()

    return value


def execute_sql(
    sql: str,
    db_id: str,
    database_dir: str,
    max_rows: int = 5000,
) -> Tuple[bool, Optional[List[Tuple[Any, ...]]], str]:
    if not is_safe_readonly_sql(sql):
        return False, None, "Unsafe or non-readonly SQL"

    db_path = get_database_path(db_id=db_id, database_dir=database_dir)
    conn = get_db_connection(db_path)

    if conn is None:
        return False, None, f"Could not connect to DB for db_id={db_id}"

    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchmany(max_rows + 1)

        if len(rows) > max_rows:
            rows = rows[:max_rows]

        normalized_rows = [
            tuple(normalize_cell(cell) for cell in row)
            for row in rows
        ]

        return True, normalized_rows, ""

    except Exception as exc:
        return False, None, str(exc)


def normalize_rows(rows: Optional[List[Tuple[Any, ...]]]) -> List[Tuple[Any, ...]]:
    if rows is None:
        return []

    try:
        return sorted(rows, key=lambda r: repr(r))
    except Exception:
        return rows


def result_match(
    pred_rows: Optional[List[Tuple[Any, ...]]],
    gold_rows: Optional[List[Tuple[Any, ...]]],
) -> bool:
    return normalize_rows(pred_rows) == normalize_rows(gold_rows)


# ---- BIRD-style execution + matching ----
#
# The official BIRD evaluator (AlibabaResearch/DAMO-ConvAI/bird/llm/src/evaluation.py)
# compares results with raw `set(predicted_res) == set(ground_truth_res)` — no
# normalization of strings, floats, or whitespace, and a per-query timeout.
#
# These helpers reproduce that behavior for use inside training rewards.

_BIRD_GOLD_CACHE: Dict[Tuple[str, str], Tuple[bool, Optional[frozenset], str]] = {}
_BIRD_GOLD_CACHE_LOCK = threading.Lock()


def _rows_to_hashable_set(rows: Optional[List[Tuple[Any, ...]]]) -> frozenset:
    if not rows:
        return frozenset()
    hashable: List[Tuple[Any, ...]] = []
    for row in rows:
        try:
            hashable.append(tuple(row))
        except Exception:
            hashable.append((repr(row),))
    try:
        return frozenset(hashable)
    except TypeError:
        # Some cells may be unhashable (rare; e.g. lists). Fall back to repr.
        return frozenset(tuple(repr(c) for c in row) for row in hashable)


def bird_execute_sql(
    sql: str,
    db_id: str,
    database_dir: str,
    timeout_s: float = 5.0,
) -> Tuple[bool, Optional[List[Tuple[Any, ...]]], str]:
    """BIRD-style execution: raw rows (no normalization), per-query timeout.

    Uses a watchdog thread that calls `connection.interrupt()` on timeout, which
    works from non-main threads (unlike SIGALRM).
    """
    if not is_safe_readonly_sql(sql):
        return False, None, "Unsafe or non-readonly SQL"

    db_path = get_database_path(db_id=db_id, database_dir=database_dir)
    if not db_path or not os.path.exists(db_path):
        return False, None, f"Could not connect to DB for db_id={db_id}"

    # BIRD opens a fresh connection per query; we mirror that to keep parity.
    try:
        uri = f"file:{db_path}?mode=ro"
        conn = sqlite3.connect(
            uri,
            uri=True,
            check_same_thread=False,
            timeout=30,
        )
    except Exception as exc:
        return False, None, f"Could not open DB: {exc}"

    timer = threading.Timer(timeout_s, conn.interrupt)
    timer.daemon = True
    try:
        timer.start()
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        return True, rows, ""
    except Exception as exc:
        return False, None, str(exc)
    finally:
        timer.cancel()
        try:
            conn.close()
        except Exception:
            pass


def bird_get_gold_rows(
    gold_sql: str,
    db_id: str,
    database_dir: str,
    timeout_s: float = 30.0,
) -> Tuple[bool, Optional[frozenset], str]:
    """Cache gold-side execution by (db_id, gold_sql) since gold is fixed per
    training row but we re-evaluate it once per rollout in a group of G.
    """
    key = (db_id, gold_sql)
    with _BIRD_GOLD_CACHE_LOCK:
        cached = _BIRD_GOLD_CACHE.get(key)
    if cached is not None:
        return cached

    ok, rows, err = bird_execute_sql(
        sql=gold_sql,
        db_id=db_id,
        database_dir=database_dir,
        timeout_s=timeout_s,
    )
    result = (ok, _rows_to_hashable_set(rows) if ok else None, err)

    with _BIRD_GOLD_CACHE_LOCK:
        _BIRD_GOLD_CACHE[key] = result
    return result


def bird_result_match(
    pred_rows: Optional[List[Tuple[Any, ...]]],
    gold_row_set: Optional[frozenset],
) -> bool:
    """Set equality on RAW rows, matching BIRD's `set(pred) == set(gold)`."""
    if gold_row_set is None:
        return False
    return _rows_to_hashable_set(pred_rows) == gold_row_set
