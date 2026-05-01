import re
from typing import List

from nl2sql_gspo.sql_utils import (
    extract_completion_text,
    extract_sql,
    execute_sql,
    is_safe_readonly_sql,
    result_match,
)

from nl2sql_gspo.schema_utils import (
    extract_schema_from_prompt,
    extract_schema_items_from_sql,
    tokenize_sql,
    get_ngrams,
    jaccard,
)


def prompt_to_text(prompt) -> str:
    if isinstance(prompt, list):
        return "\n".join(
            str(m.get("content", ""))
            for m in prompt
            if isinstance(m, dict)
        )

    return str(prompt or "")


def make_nl2sql_rewards(database_dir: str):
    def format_reward(completions, **kwargs) -> List[float]:
        rewards = []

        for completion in completions:
            completion_text = extract_completion_text(completion)
            sql = extract_sql(completion)

            if not sql:
                rewards.append(0.0)
                continue

            score = 0.0

            if is_safe_readonly_sql(sql):
                score += 0.2

            if re.search(r"\bSELECT\b|\bWITH\b", sql, re.IGNORECASE):
                score += 0.2

            if not re.search(r"```|Explanation:|Here is|The query|This query", completion_text, re.IGNORECASE):
                score += 0.1

            rewards.append(min(score, 1.0))

        return rewards

    def execution_reward(completions, db_id=None, **kwargs) -> List[float]:
        db_ids = db_id or kwargs.get("db_ids") or [""] * len(completions)
        rewards = []

        for completion, current_db_id in zip(completions, db_ids):
            sql = extract_sql(completion)
            ok, _, _ = execute_sql(
                sql=sql,
                db_id=current_db_id,
                database_dir=database_dir,
            )
            rewards.append(1.0 if ok else 0.0)

        return rewards

    def result_reward(completions, db_id=None, gold_sql=None, **kwargs) -> List[float]:
        db_ids = db_id or kwargs.get("db_ids") or [""] * len(completions)
        gold_sqls = gold_sql or kwargs.get("query") or kwargs.get("sql") or [""] * len(completions)

        rewards = []

        for completion, current_db_id, current_gold_sql in zip(completions, db_ids, gold_sqls):
            pred_sql = extract_sql(completion)
            gold_sql_text = extract_sql(current_gold_sql)

            pred_ok, pred_rows, _ = execute_sql(
                sql=pred_sql,
                db_id=current_db_id,
                database_dir=database_dir,
            )

            gold_ok, gold_rows, _ = execute_sql(
                sql=gold_sql_text,
                db_id=current_db_id,
                database_dir=database_dir,
            )

            if pred_ok and gold_ok and result_match(pred_rows, gold_rows):
                rewards.append(1.0)
            else:
                rewards.append(0.0)

        return rewards

    def schema_linking_reward(
        completions,
        prompts=None,
        messages=None,
        gold_sql=None,
        **kwargs,
    ) -> List[float]:
        prompt_like = prompts or kwargs.get("prompt") or messages or [""] * len(completions)
        gold_sqls = gold_sql or kwargs.get("query") or kwargs.get("sql") or [""] * len(completions)

        rewards = []

        for completion, prompt, current_gold_sql in zip(completions, prompt_like, gold_sqls):
            prompt_text = prompt_to_text(prompt)
            gold_sql_text = extract_sql(current_gold_sql)

            schema_tables, schema_cols = extract_schema_from_prompt(prompt_text)
            schema_items = schema_tables | schema_cols

            pred_items = extract_schema_items_from_sql(extract_sql(completion))
            gold_items = extract_schema_items_from_sql(gold_sql_text)

            target_items = gold_items if gold_items else schema_items
            rewards.append(jaccard(pred_items, target_items))

        return rewards

    def ngram_reward(completions, gold_sql=None, **kwargs) -> List[float]:
        gold_sqls = gold_sql or kwargs.get("query") or kwargs.get("sql") or [""] * len(completions)
        rewards = []

        for completion, current_gold_sql in zip(completions, gold_sqls):
            pred_tokens = tokenize_sql(extract_sql(completion))
            gold_tokens = tokenize_sql(extract_sql(current_gold_sql))

            if not pred_tokens or not gold_tokens:
                rewards.append(0.0)
                continue

            unigram_score = jaccard(set(pred_tokens), set(gold_tokens))
            bigram_score = jaccard(get_ngrams(pred_tokens, 2), get_ngrams(gold_tokens, 2))
            trigram_score = jaccard(get_ngrams(pred_tokens, 3), get_ngrams(gold_tokens, 3))

            score = 0.2 * unigram_score + 0.4 * bigram_score + 0.4 * trigram_score
            rewards.append(score)

        return rewards

    def evidence_utilization_reward(completions, evidence=None, **kwargs) -> List[float]:
        evidences = evidence or kwargs.get("external_knowledge") or kwargs.get("hint") or [""] * len(completions)
        rewards = []

        for completion, current_evidence in zip(completions, evidences):
            sql = extract_sql(completion).lower()
            evidence_text = str(current_evidence or "").lower()

            if not evidence_text.strip():
                rewards.append(0.0)
                continue

            tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{2,}", evidence_text)
            stopwords = {
                "the", "and", "for", "with", "that", "this",
                "should", "must", "from", "into", "only",
            }

            tokens = {
                token for token in tokens
                if token not in stopwords
            }

            if not tokens:
                rewards.append(0.0)
                continue

            matched = sum(1 for token in tokens if token in sql)
            rewards.append(min(1.0, matched / max(3, len(tokens))))

        return rewards

    return [
        format_reward,
        execution_reward,
        result_reward,
        schema_linking_reward,
        ngram_reward,
        evidence_utilization_reward,
    ]