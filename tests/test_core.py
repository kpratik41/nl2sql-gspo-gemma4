import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nl2sql_gspo.data import normalize_record
from nl2sql_gspo.train_gspo_nl2sql import parse_args
from nl2sql_gspo.sql_utils import extract_sql, get_database_path, is_safe_readonly_sql


class NormalizeRecordTests(unittest.TestCase):
    def test_preserves_uppercase_sql_field_from_bird_records(self):
        raw_path = ROOT / "data" / "bird_train_data" / "raw" / "train-6601.jsonl"
        with raw_path.open("r", encoding="utf-8") as handle:
            first_record = json.loads(next(handle))

        normalized = normalize_record(first_record)

        self.assertEqual(normalized["db_id"], "movie_platform")
        self.assertTrue(normalized["gold_sql"].startswith("SELECT director_name"))
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


if __name__ == "__main__":
    unittest.main()