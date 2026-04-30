import argparse
import json

from nl2sql_gspo.data import normalize_record
from nl2sql_gspo.rewards import make_nl2sql_rewards
from nl2sql_gspo.sql_utils import extract_sql, execute_sql


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--database_dir", required=True)
    args = parser.parse_args()

    with open(args.file, "r", encoding="utf-8") as f:
        raw = json.loads(next(f))

    example = normalize_record(raw)

    completion = raw.get("gold_sql") or raw.get("query") or raw.get("sql")

    if not completion and raw.get("messages"):
        for m in raw["messages"]:
            if m.get("role") == "assistant":
                completion = m.get("content", "")
                break

    sql = extract_sql(completion)
    print("Extracted SQL:")
    print(sql)

    ok, rows, err = execute_sql(
        sql=sql,
        db_id=example["db_id"],
        database_dir=args.database_dir,
    )

    print("\nExecution OK:", ok)
    print("Error:", err)
    print("Rows sample:", rows[:3] if rows else rows)

    reward_functions = make_nl2sql_rewards(args.database_dir)

    completions = [completion]
    kwargs = {
        "db_id": [example["db_id"]],
        "gold_sql": [example["gold_sql"]],
        "evidence": [example["evidence"]],
        "prompt": [example["prompt"]],
        "messages": [example["messages"]],
    }

    print("\nRewards:")
    for reward_fn in reward_functions:
        print(reward_fn.__name__, reward_fn(completions, **kwargs))


if __name__ == "__main__":
    main()