import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nl2sql_gspo.data import normalize_record
from nl2sql_gspo.rewards import make_nl2sql_rewards
from nl2sql_gspo.train_gspo_nl2sql import parse_args
from nl2sql_gspo.sql_utils import _DB_CONNECTIONS, extract_sql, get_database_path, is_safe_readonly_sql
from scripts.run_inference_bird import (
    get_generation_messages,
    plan_transformers_worker_devices,
    plan_vllm_device_groups,
    shard_rows_for_data_parallel,
)


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

    def test_extracts_db_id_and_evidence_from_schema_built_messages(self):
        normalized = normalize_record(
            {
                "messages": [
                    {"role": "system", "content": "You are a Text-to-SQL agent."},
                    {
                        "role": "user",
                        "content": (
                            "<question>Example question</question>\n"
                            "<hint>use the releaseDate column</hint>\n"
                            "<database_schema>\n`card_games`\n</database_schema>\n"
                            "<db_id>card_games</db_id>"
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": "<final_answer><sql_code>SELECT code FROM sets</sql_code></final_answer>",
                    },
                ]
            }
        )

        self.assertEqual(normalized["db_id"], "card_games")
        self.assertEqual(normalized["evidence"], "use the releaseDate column")
        self.assertEqual(normalized["gold_sql"], "<final_answer><sql_code>SELECT code FROM sets</sql_code></final_answer>")


class SqlUtilsTests(unittest.TestCase):
    def test_extract_sql_from_fenced_response(self):
        completion = """Here is the query:\n```sql\nSELECT * FROM movies\n```"""
        self.assertEqual(extract_sql(completion), "SELECT * FROM movies;")

    def test_extract_sql_from_tagged_response(self):
        completion = (
            "<scratch_pad>reasoning</scratch_pad>\n"
            "<final_answer>\n"
            "<sql_code>SELECT Body FROM posts WHERE id = 1</sql_code>\n"
            "</final_answer>"
        )
        self.assertEqual(extract_sql(completion), "SELECT Body FROM posts WHERE id = 1;")

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
    def test_parse_args_accepts_resume_checkpoint_and_limits(self):
        args = parse_args(
            [
                "--model_name_or_path",
                "model",
                "--train_file",
                "train.jsonl",
                "--eval_file",
                "dev.jsonl",
                "--train_limit",
                "123",
                "--eval_limit",
                "45",
                "--database_dir",
                "databases",
                "--output_dir",
                "outputs/run",
                "--eval_on_start",
                "--resume_from_checkpoint",
                "outputs/run/checkpoint-100",
            ]
        )

        self.assertEqual(args.resume_from_checkpoint, "outputs/run/checkpoint-100")
        self.assertEqual(args.train_limit, 123)
        self.assertEqual(args.eval_limit, 45)
        self.assertTrue(args.eval_on_start)


class InferenceScriptTests(unittest.TestCase):
    def test_get_generation_messages_strips_assistant_turns_from_legacy_rows(self):
        row = {
            "messages": [
                {"role": "system", "content": "You are a Text-to-SQL agent."},
                {"role": "user", "content": "List all users."},
                {"role": "assistant", "content": "SELECT * FROM users;"},
            ]
        }

        messages = get_generation_messages(row)

        self.assertEqual([message["role"] for message in messages], ["system", "user"])

    def test_get_generation_messages_prefers_normalized_prompt(self):
        row = {
            "prompt": [
                {"role": "system", "content": "You are a Text-to-SQL agent."},
                {"role": "user", "content": "Count users."},
            ],
            "messages": [
                {"role": "system", "content": "stale system"},
                {"role": "user", "content": "stale user"},
                {"role": "assistant", "content": "stale sql"},
            ],
        }

        messages = get_generation_messages(row)

        self.assertEqual(messages, row["prompt"])

    def test_plan_vllm_device_groups_splits_visible_devices(self):
        with mock.patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"}, clear=False):
            groups = plan_vllm_device_groups(tensor_parallel_size=4, data_parallel_size=2)

        self.assertEqual(groups, [["0", "1", "2", "3"], ["4", "5", "6", "7"]])

    def test_plan_transformers_worker_devices_defaults_to_all_visible_devices(self):
        with mock.patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}, clear=False):
            devices = plan_transformers_worker_devices(data_parallel_size=0)

        self.assertEqual(devices, ["0", "1", "2", "3"])

    def test_shard_rows_for_data_parallel_round_robins_rows(self):
        rows = [{"source_idx": idx} for idx in range(5)]

        shards = shard_rows_for_data_parallel(rows, num_shards=2)

        self.assertEqual([row["source_idx"] for row in shards[0]], [0, 2, 4])
        self.assertEqual([row["source_idx"] for row in shards[1]], [1, 3])


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

    def test_result_reward_extracts_tagged_gold_sql(self):
        completions = [
            "<final_answer><sql_code>SELECT name FROM users WHERE age > 30</sql_code></final_answer>",
        ]
        tagged_gold_sql = [
            "<scratch_pad>Gold reference SQL.</scratch_pad><final_answer><sql_code>SELECT name FROM users WHERE age > 30</sql_code></final_answer>",
        ]

        result_scores = self.result_reward(
            completions=completions,
            db_id=[self.db_id],
            gold_sql=tagged_gold_sql,
        )

        self.assertEqual(result_scores, [1.0])

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