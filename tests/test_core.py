import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nl2sql_gspo.data import normalize_record
from nl2sql_gspo.rewards import make_nl2sql_rewards
from nl2sql_gspo.train_gspo_nl2sql import parse_args
from nl2sql_gspo.sql_utils import _DB_CONNECTIONS, extract_sql, get_database_path, is_safe_readonly_sql


class NormalizeRecordTests(unittest.TestCase):
    def test_preserves_uppercase_sql_field_from_bird_records(self):
        raw_path = ROOT / "data" / "bird_train_data" / "raw" / "train-6601.jsonl"
        with raw_path.open("r", encoding="utf-8") as handle:
            first_record = json.loads(next(handle))

        normalized = normalize_record(first_record)

        self.assertEqual(normalized["db_id"], first_record["db_id"])
        self.assertEqual(normalized["gold_sql"], first_record["SQL"])
        self.assertEqual(len(normalized["prompt"]), 2)
        self.assertEqual(normalized["prompt"][0]["role"], "system")
        self.assertEqual(normalized["prompt"][1]["role"], "user")


class SqlUtilsTests(unittest.TestCase):
    def test_extract_sql_from_fenced_response(self):
        completion = """Here is the query:\n```sql\nSELECT * FROM movies\n```"""
        self.assertEqual(extract_sql(completion), "SELECT * FROM movies;")

    def test_safe_readonly_sql_blocks_writes(self):
        self.assertTrue(is_safe_readonly_sql("SELECT * FROM movies;"))
        self.assertFalse(is_safe_readonly_sql("DROP TABLE movies;"))

    def test_get_database_path_supports_top_level_split_layout(self):
        database_root = ROOT / "databases"
        train_db = get_database_path("movie_platform", str(database_root))
        dev_db = get_database_path("california_schools", str(database_root))

        self.assertTrue(train_db.endswith("movie_platform.sqlite"))
        self.assertTrue(dev_db.endswith("california_schools.sqlite"))


class TrainScriptTests(unittest.TestCase):
    def test_parse_args_accepts_resume_checkpoint(self):
        args = parse_args(
            [
                "--model_name_or_path",
                "model",
                "--train_file",
                "train.jsonl",
                "--database_dir",
                "databases",
                "--output_dir",
                "outputs/run",
                "--resume_from_checkpoint",
                "outputs/run/checkpoint-100",
            ]
        )

        self.assertEqual(args.resume_from_checkpoint, "outputs/run/checkpoint-100")


class RewardTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_root = Path(self.temp_dir.name)
        self.db_id = "reward_db"
        db_dir = self.database_root / self.db_id
        db_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = db_dir / f"{self.db_id}.sqlite"

        conn = sqlite3.connect(self.db_path)
        conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
        conn.executemany(
            "INSERT INTO users (name, age) VALUES (?, ?)",
            [("Alice", 30), ("Bob", 40), ("Carol", 25)],
        )
        conn.commit()
        conn.close()

        self.reward_functions = make_nl2sql_rewards(str(self.database_root))
        self.format_reward = self.reward_functions[0]
        self.execution_reward = self.reward_functions[1]
        self.result_reward = self.reward_functions[2]
        self.schema_linking_reward = self.reward_functions[3]
        self.ngram_reward = self.reward_functions[4]
        self.evidence_utilization_reward = self.reward_functions[5]

        self.common_kwargs = {
            "db_id": [self.db_id],
            "gold_sql": ["SELECT name FROM users WHERE age > 30;"],
            "messages": [[
                {
                    "role": "user",
                    "content": "CREATE TABLE users (id INTEGER, name TEXT, age INTEGER)",
                }
            ]],
            "evidence": ["Use the age column and filter for values greater than 30."],
        }

    def tearDown(self):
        for connection in _DB_CONNECTIONS.values():
            if connection is not None:
                connection.close()
        _DB_CONNECTIONS.clear()
        self.temp_dir.cleanup()

    def test_execution_and_result_rewards_distinguish_correctness(self):
        completions = [
            "SELECT name FROM users WHERE age > 30;",
            "SELECT name FROM users WHERE age > 20;",
            "SELECT missing_column FROM users WHERE age > 30;",
        ]

        execution_scores = self.execution_reward(completions=completions, db_id=[self.db_id] * 3)
        result_scores = self.result_reward(
            completions=completions,
            db_id=[self.db_id] * 3,
            gold_sql=[self.common_kwargs["gold_sql"][0]] * 3,
        )

        self.assertEqual(execution_scores, [1.0, 1.0, 0.0])
        self.assertEqual(result_scores, [1.0, 0.0, 0.0])

    def test_schema_and_ngram_rewards_prefer_closer_sql(self):
        exact_completion = ["SELECT name FROM users WHERE age > 30;"]
        partial_completion = ["SELECT name FROM users;"]
        unrelated_completion = ["SELECT id FROM users WHERE age < 10;"]

        exact_schema = self.schema_linking_reward(completions=exact_completion, **self.common_kwargs)[0]
        partial_schema = self.schema_linking_reward(completions=partial_completion, **self.common_kwargs)[0]
        unrelated_schema = self.schema_linking_reward(completions=unrelated_completion, **self.common_kwargs)[0]

        exact_ngram = self.ngram_reward(completions=exact_completion, gold_sql=self.common_kwargs["gold_sql"])[0]
        partial_ngram = self.ngram_reward(completions=partial_completion, gold_sql=self.common_kwargs["gold_sql"])[0]

        self.assertGreaterEqual(exact_schema, partial_schema)
        self.assertGreater(partial_schema, unrelated_schema)
        self.assertGreater(exact_ngram, partial_ngram)

    def test_format_and_evidence_rewards_reward_clean_sql_and_hint_usage(self):
        clean_completion = "SELECT name FROM users WHERE age > 30;"
        explained_completion = "Here is the query:\nSELECT name FROM users WHERE age > 30;\nExplanation: filter by age"
        unrelated_completion = "SELECT name FROM users;"

        clean_format = self.format_reward(completions=[clean_completion])[0]
        explained_format = self.format_reward(completions=[explained_completion])[0]
        evidence_hit = self.evidence_utilization_reward(
            completions=[clean_completion],
            evidence=self.common_kwargs["evidence"],
        )[0]
        evidence_miss = self.evidence_utilization_reward(
            completions=[unrelated_completion],
            evidence=self.common_kwargs["evidence"],
        )[0]

        self.assertGreater(clean_format, explained_format)
        self.assertGreater(evidence_hit, evidence_miss)


if __name__ == "__main__":
    unittest.main()