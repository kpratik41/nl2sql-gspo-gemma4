"""What-if scenarios: how do the new rewards score across realistic completions?

Picks a real dev example, then scores 5 hand-crafted completions:
  A) gold-equivalent SQL inside the strict XML format -> max reward
  B) gold-equivalent SQL but plain text (no XML format)
  C) format OK + executable SQL with WRONG result
  D) format OK + SQL that does NOT execute (typo'd column)
  E) garbage / no SQL at all
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nl2sql_gspo.rewards import make_nl2sql_rewards

# pick the first dev example
dev_path = ROOT / "outputs" / "dev-20251106-schema.jsonl"
with dev_path.open() as f:
    row = None
    for line in f:
        r = json.loads(line)
        # use normalize equivalents
        from nl2sql_gspo.data import normalize_record
        n = normalize_record(r)
        if n["db_id"] and n["gold_sql"]:
            row = n
            break

assert row is not None
print("db_id:", row["db_id"])

# The dev file stores gold_sql wrapped in <scratch_pad>/<final_answer>/<sql_code>.
# Extract the raw SQL once so our hand-crafted scenarios don't double-wrap.
from nl2sql_gspo.sql_utils import extract_sql
raw_gold = extract_sql(row["gold_sql"]).rstrip(";")
print("raw_gold:", raw_gold)
print()

gold_for_reward = row["gold_sql"]  # what the trainer actually passes

def wrap(sql: str) -> str:
    return (
        "<scratch_pad>\nthinking\n</scratch_pad>\n"
        "<final_answer>\n<sql_code>" + sql + "</sql_code>\n</final_answer>"
    )

# A) max-quality: format + same-as-gold SQL
A = wrap(raw_gold)
# B) right SQL but no XML format -> format reward = 0
B = raw_gold
# C) format OK + executable but wrong result.
C_sql = "SELECT name FROM sqlite_master WHERE type = 'table' LIMIT 1"
C = wrap(C_sql)
# D) format OK but SQL fails to execute (refers to nonexistent table)
D_sql = "SELECT col FROM table_does_not_exist_zzz"
D = wrap(D_sql)
# E) garbage
E = "I cannot answer that question."

completions = [A, B, C, D, E]
labels = [
    "A) gold SQL + correct format",
    "B) gold SQL + NO format wrapper",
    "C) format OK + executes + WRONG result",
    "D) format OK + does NOT execute",
    "E) garbage / no SQL",
]

reward_fns = make_nl2sql_rewards(database_dir=str(ROOT / "databases"))
names = ["format", "execution", "result", "table_link", "column_link", "nonnull"]
weights = [0.2, 0.5, 2.0, 0.5, 0.5, 0.1]

per = []
for fn in reward_fns:
    scores = fn(
        completions=completions,
        db_id=[row["db_id"]] * len(completions),
        gold_sql=[gold_for_reward] * len(completions),
        evidence=[row.get("evidence", "")] * len(completions),
        prompts=[row["prompt"]] * len(completions),
        messages=[row["messages"]] * len(completions),
    )
    per.append(scores)

print(f"{'scenario':<42}" + "".join(f"{n:>12}" for n in names) + f"{'WEIGHTED':>12}")
print("-" * (42 + 12 * 7))
for i, lbl in enumerate(labels):
    row_scores = [per[j][i] for j in range(len(names))]
    weighted = sum(w * s for w, s in zip(weights, row_scores))
    cells = "".join(f"{s:>12.3f}" for s in row_scores)
    print(f"{lbl:<42}{cells}{weighted:>12.3f}")

print()
print("Reward weights:", dict(zip(names, weights)))
print("Max possible weighted reward:", sum(weights))
