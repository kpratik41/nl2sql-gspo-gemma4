import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List

from nl2sql_gspo.sql_utils import (
    extract_completion_text,
    extract_sql,
    bird_execute_sql,
    bird_get_gold_rows,
    bird_result_match,
)

from nl2sql_gspo.schema_utils import (
    extract_tables_from_sql,
    extract_columns_from_sql,
    jaccard,
)


# Strict response format expected from the model:
#   <scratch_pad> ... </scratch_pad>
#   <final_answer>
#   <sql_code> ... </sql_code>
#   </final_answer>
FORMAT_RE = re.compile(
    r"<scratch_pad>\s*.+?\s*</scratch_pad>\s*"
    r"<final_answer>\s*<sql_code>\s*.+?\s*</sql_code>\s*</final_answer>",
    re.IGNORECASE | re.DOTALL,
)


def _has_any_nonnull(rows) -> bool:
    if not rows:
        return False

    for row in rows:
        for cell in row:
            if cell is None:
                continue
            if isinstance(cell, str) and not cell.strip():
                continue
            return True

    return False


def make_nl2sql_rewards(
    database_dir: str,
    exec_timeout_s: float = 60.0,
    gold_timeout_s: float = 60.0,
    reward_workers: int = 1,
    tokenizer=None,
    length_penalty_max: int = 4096,
    length_penalty_buffer: int = 512,
):
    """Build the 7 reward functions consumed by ``GRPOTrainer``.

    Order: format, execution, result, table_linking, column_linking, nonnull,
    length_penalty (Soft Overlong Punishment from DAPO §3.4).

    Parameters
    ----------
    exec_timeout_s : per-query SQL execution timeout for predicted SQL.
    gold_timeout_s : per-query SQL execution timeout for gold SQL.
    tokenizer      : if provided, used to count completion length in tokens
                     for the length penalty; otherwise ``len(text.split())``.
    length_penalty_max    : ``L_max`` in the DAPO Soft Overlong Punishment
                            formula. Beyond this length the penalty saturates
                            at -1.
    length_penalty_buffer : ``L_cache`` — the linear ramp begins at
                            ``L_max - L_cache`` and reaches -1 at ``L_max``.
    """

    # Shared per-process cache so the three execution-based rewards
    # (`execution_reward`, `result_reward`, `nonnull_reward`) only run each
    # predicted SQL once per (db_id, sql_text). Key on the extracted SQL string
    # which is what gets passed to bird_execute_sql, so identical
    # rollouts and even identical re-extracted SQL across the same generation
    # batch coalesce. We cap the cache to bound memory; eviction is
    # FIFO via dict insertion order.
    _exec_cache: dict = {}
    _EXEC_CACHE_MAX = 8192
    _exec_cache_lock = threading.Lock()

    reward_workers = max(int(reward_workers or 1), 1)

    def _parallel_map(fn, items):
        if reward_workers <= 1 or len(items) <= 1:
            return [fn(item) for item in items]
        with ThreadPoolExecutor(max_workers=reward_workers) as executor:
            return list(executor.map(fn, items))

    def _cached_bird_execute(sql: str, db_id: str):
        key = (db_id, sql)
        with _exec_cache_lock:
            cached = _exec_cache.get(key)
        if cached is not None:
            return cached
        ok, rows, err = bird_execute_sql(
            sql=sql,
            db_id=db_id,
            database_dir=database_dir,
            timeout_s=exec_timeout_s,
        )
        with _exec_cache_lock:
            if len(_exec_cache) >= _EXEC_CACHE_MAX:
                # Drop oldest entry (FIFO).
                try:
                    _exec_cache.pop(next(iter(_exec_cache)))
                except StopIteration:
                    pass
            _exec_cache[key] = (ok, rows, err)
        return ok, rows, err

    def _execution_reward_one(item) -> float:
        completion, current_db_id = item
        sql = extract_sql(completion)
        ok, _, _ = _cached_bird_execute(sql, current_db_id)
        return 1.0 if ok else 0.0

    def _result_reward_one(item) -> float:
        completion, current_db_id, current_gold_sql = item
        pred_sql = extract_sql(completion)
        gold_sql_text = extract_sql(current_gold_sql)

        pred_ok, pred_rows, _ = _cached_bird_execute(pred_sql, current_db_id)
        gold_ok, gold_row_set, _ = bird_get_gold_rows(
            gold_sql=gold_sql_text,
            db_id=current_db_id,
            database_dir=database_dir,
            timeout_s=gold_timeout_s,
        )
        return 1.0 if pred_ok and gold_ok and bird_result_match(pred_rows, gold_row_set) else 0.0

    def _nonnull_reward_one(item) -> float:
        completion, current_db_id = item
        sql = extract_sql(completion)
        ok, rows, _ = _cached_bird_execute(sql, current_db_id)
        return 1.0 if ok and _has_any_nonnull(rows) else 0.0

    def format_reward(completions, **kwargs) -> List[float]:
        rewards = []
        for completion in completions:
            text = extract_completion_text(completion)
            rewards.append(1.0 if FORMAT_RE.search(text) else 0.0)
        return rewards

    def execution_reward(completions, db_id=None, **kwargs) -> List[float]:
        db_ids = db_id or kwargs.get("db_ids") or [""] * len(completions)
        return _parallel_map(_execution_reward_one, list(zip(completions, db_ids)))

    def result_reward(completions, db_id=None, gold_sql=None, **kwargs) -> List[float]:
        db_ids = db_id or kwargs.get("db_ids") or [""] * len(completions)
        gold_sqls = gold_sql or kwargs.get("query") or kwargs.get("sql") or [""] * len(completions)
        items = list(zip(completions, db_ids, gold_sqls))
        return _parallel_map(_result_reward_one, items)

    def table_linking_reward(completions, gold_sql=None, **kwargs) -> List[float]:
        gold_sqls = gold_sql or kwargs.get("query") or kwargs.get("sql") or [""] * len(completions)
        rewards = []
        for completion, current_gold_sql in zip(completions, gold_sqls):
            pred_tables = extract_tables_from_sql(extract_sql(completion))
            gold_tables = extract_tables_from_sql(extract_sql(current_gold_sql))
            if not gold_tables:
                rewards.append(0.0)
                continue
            rewards.append(1.0 if pred_tables == gold_tables else 0.0)
        return rewards

    def column_linking_reward(completions, gold_sql=None, **kwargs) -> List[float]:
        gold_sqls = gold_sql or kwargs.get("query") or kwargs.get("sql") or [""] * len(completions)
        rewards = []
        for completion, current_gold_sql in zip(completions, gold_sqls):
            pred_cols = extract_columns_from_sql(extract_sql(completion))
            gold_cols = extract_columns_from_sql(extract_sql(current_gold_sql))
            if not gold_cols:
                rewards.append(0.0)
                continue
            rewards.append(jaccard(pred_cols, gold_cols))
        return rewards

    def nonnull_reward(completions, db_id=None, **kwargs) -> List[float]:
        db_ids = db_id or kwargs.get("db_ids") or [""] * len(completions)
        return _parallel_map(_nonnull_reward_one, list(zip(completions, db_ids)))

    def length_penalty_reward(completions, **kwargs) -> List[float]:
        """Soft Overlong Punishment (DAPO, Eq. 13).

        ``len(y) <= L_max - L_cache``      -> 0
        ``L_max - L_cache < len(y) <= L_max`` -> linear ramp in [-1, 0]
        ``len(y) > L_max``                 -> -1
        """
        l_max = max(int(length_penalty_max), 1)
        l_cache = max(int(length_penalty_buffer), 1)
        soft_threshold = l_max - l_cache

        rewards: List[float] = []
        for completion in completions:
            text = extract_completion_text(completion)
            if tokenizer is not None:
                try:
                    length = len(tokenizer.encode(text, add_special_tokens=False))
                except Exception:
                    length = len(text.split())
            else:
                length = len(text.split())

            if length <= soft_threshold:
                rewards.append(0.0)
            elif length <= l_max:
                # Linear from 0 (at soft_threshold) down to -1 (at l_max).
                penalty = (soft_threshold - length) / float(l_cache)
                rewards.append(float(penalty))
            else:
                rewards.append(-1.0)
        return rewards

    return [
        format_reward,
        execution_reward,
        result_reward,
        table_linking_reward,
        column_linking_reward,
        nonnull_reward,
        length_penalty_reward,
    ]
