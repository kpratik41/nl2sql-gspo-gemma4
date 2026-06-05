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
from nl2sql_gspo.prompt_builder import PromptBuilder, PromptConfig
from nl2sql_gspo.resume import (
    ResumeManifestError,
    atomic_write_jsonl,
    build_manifest,
    checkpoint_map,
    prepare_manifest,
    safe_read_jsonl,
    validate_resume_args,
)
from nl2sql_gspo.sample_plan import expand_sample_plan
from nl2sql_gspo.sql_utils import (
    bird_execute_sql,
    bird_get_gold_rows,
    bird_result_match,
    extract_sql,
    get_database_path,
    is_safe_readonly_sql,
)
from scripts.data_generation.schema_build import MSchema, build_mschema_from_db, build_output_entry, format_user_prompt
from scripts.run_inference_bird import (
    get_generation_messages,
    plan_vllm_device_groups,
    shard_rows_for_data_parallel,
)
from scripts.run_self_consistency_bird import choose_majority_vote_candidate, rows_to_vote_signature


class NormalizeRecordTests(unittest.TestCase):
    def test_preserves_uppercase_sql_field_from_bird_records(self):
        first_record = {
            "db_id": "movie_platform",
            "question": "List movie names.",
            "evidence": "movie names are stored in title",
            "SQL": "SELECT title FROM movies;",
        }

        normalized = normalize_record(first_record)

        self.assertEqual(normalized["db_id"], first_record["db_id"])
        self.assertEqual(normalized["gold_sql"], first_record["SQL"])
        self.assertEqual(len(normalized["prompt"]), 2)
        self.assertEqual(normalized["prompt"][0]["role"], "system")
        self.assertEqual(normalized["prompt"][1]["role"], "user")


class ResumeHelperTests(unittest.TestCase):
    def test_manifest_creation_and_exact_match_resume(self):
        class Args:
            model_name_or_path = "model-a"
            input_file = "input.json"
            temperature = 0.0

        manifest = build_manifest(Args, mode="temp0", fields=["model_name_or_path", "input_file", "temperature"])
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            prepare_manifest(output_dir, manifest, resume=False)
            prepare_manifest(output_dir, manifest, resume=True)

    def test_manifest_mismatch_refuses_resume(self):
        class ArgsA:
            model_name_or_path = "model-a"

        class ArgsB:
            model_name_or_path = "model-b"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            prepare_manifest(output_dir, build_manifest(ArgsA, mode="temp0", fields=["model_name_or_path"]), resume=False)
            with self.assertRaises(ResumeManifestError):
                prepare_manifest(output_dir, build_manifest(ArgsB, mode="temp0", fields=["model_name_or_path"]), resume=True)

    def test_checkpoint_map_dedupes_by_idx_and_sample_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.jsonl"
            atomic_write_jsonl(
                path,
                [
                    {"idx": 1, "sample_id": 0, "value": "old"},
                    {"idx": 1, "sample_id": 0, "value": "new"},
                    {"idx": 1, "sample_id": 1, "value": "other"},
                ],
            )

            rows = checkpoint_map(path, lambda row: (int(row["idx"]), int(row["sample_id"])))

        self.assertEqual(rows[(1, 0)]["value"], "new")
        self.assertEqual(rows[(1, 1)]["value"], "other")

    def test_safe_read_jsonl_ignores_corrupt_final_line(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.jsonl"
            path.write_text('{"idx": 1}\n{"idx": ', encoding="utf-8")

            rows = safe_read_jsonl(path)

        self.assertEqual(rows, [{"idx": 1}])

    def test_resume_and_overwrite_are_mutually_exclusive(self):
        class Args:
            resume = True
            overwrite = True

        with self.assertRaises(ValueError):
            validate_resume_args(Args)

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
        with tempfile.TemporaryDirectory() as tmpdir:
            database_root = Path(tmpdir)
            top_level = database_root / "movie_platform.sqlite"
            split_db_dir = database_root / "dev_databases" / "california_schools"
            test_db_dir = database_root / "test_databases" / "nba_data"
            split_db_dir.mkdir(parents=True)
            test_db_dir.mkdir(parents=True)
            split_level = split_db_dir / "california_schools.sqlite"
            test_level = test_db_dir / "nba_data.sqlite"
            top_level.touch()
            split_level.touch()
            test_level.touch()

            train_db = get_database_path("movie_platform", str(database_root))
            dev_db = get_database_path("california_schools", str(database_root))
            test_db = get_database_path("nba_data", str(database_root))

        self.assertTrue(train_db.endswith("movie_platform.sqlite"))
        self.assertTrue(dev_db.endswith("california_schools.sqlite"))
        self.assertTrue(test_db.endswith("nba_data.sqlite"))

    def test_bird_result_match_uses_set_semantics_on_raw_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            database_root = Path(tmpdir)
            db_id = "toy"
            db_dir = database_root / db_id
            db_dir.mkdir(parents=True)
            db_path = db_dir / f"{db_id}.sqlite"
            conn = sqlite3.connect(db_path)
            conn.execute("CREATE TABLE users (name TEXT, age INTEGER)")
            conn.executemany("INSERT INTO users VALUES (?, ?)", [("Alice", 30), ("Bob", 40), ("Carol", 25)])
            conn.commit()
            conn.close()

            gold_ok, gold_set, _ = bird_get_gold_rows(
                gold_sql="SELECT name FROM users WHERE age > 20;",
                db_id=db_id,
                database_dir=str(database_root),
            )
            pred_ok, pred_rows, _ = bird_execute_sql(
                sql="SELECT name FROM users WHERE age > 20 ORDER BY age DESC;",
                db_id=db_id,
                database_dir=str(database_root),
            )

        self.assertTrue(gold_ok)
        self.assertTrue(pred_ok)
        self.assertTrue(bird_result_match(pred_rows, gold_set))


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


class PromptBuilderTests(unittest.TestCase):
    def _make_builder(self, tmpdir, **overrides):
        database_root = Path(tmpdir)
        db_id = "toy"
        db_dir = database_root / db_id
        db_dir.mkdir(parents=True, exist_ok=True)
        db_path = db_dir / f"{db_id}.sqlite"
        conn = sqlite3.connect(db_path)
        conn.execute("DROP TABLE IF EXISTS demo")
        conn.execute("CREATE TABLE demo (id INTEGER PRIMARY KEY, name TEXT, score REAL)")
        conn.executemany("INSERT INTO demo (name, score) VALUES (?, ?)", [("Alice", 1.5), ("Bob", 2.0)])
        conn.commit()
        conn.close()
        meanings_path = database_root / "column_meaning.json"
        meanings_path.write_text(
            json.dumps({"toy|demo|score": "# score is the metric value"}),
            encoding="utf-8",
        )
        values = {
            "database_dir": str(database_root),
            "meanings_file": str(meanings_path),
            "tool_mode": "default",
            **overrides,
        }
        config = PromptConfig(**values)
        return PromptBuilder(config)

    def test_runtime_dev_row_builds_prompt_and_gold_sql(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = self._make_builder(tmpdir)
            row = builder.build_row(
                {
                    "db_id": "toy",
                    "question": "What is Alice's score?",
                    "evidence": "Alice refers to name = 'Alice'",
                    "SQL": "SELECT score FROM demo WHERE name = 'Alice'",
                },
                force_rebuild=True,
            )

        self.assertEqual(row["gold_sql"], "SELECT score FROM demo WHERE name = 'Alice';")
        self.assertEqual([message["role"] for message in row["prompt"]], ["system", "user"])
        self.assertIn("<database_schema>", row["prompt"][1]["content"])
        self.assertEqual([tool["function"]["name"] for tool in row["tools"]], ["bm25_search_sqlite", "sqlite_peek", "sqlite_query"])

    def test_runtime_test_row_allows_empty_gold_sql(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = self._make_builder(tmpdir, bird_mode="test", tool_mode="none")
            row = builder.build_row(
                {
                    "db_id": "toy",
                    "question": "List names.",
                    "evidence": "",
                    "SQL": "",
                },
                force_rebuild=True,
            )

        self.assertEqual(row["gold_sql"], "")
        self.assertEqual(row["tools"], [])

    def test_schema_flags_change_rendered_prompt_and_cache_by_db(self):
        raw = {
            "db_id": "toy",
            "question": "List scores.",
            "evidence": "",
            "SQL": "SELECT score FROM demo",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            full_builder = self._make_builder(tmpdir, include_column_comments=True, include_stats=True)
            full_row = full_builder.build_row(raw, force_rebuild=True)
            full_again = full_builder.build_row(raw, force_rebuild=True)
            bare_builder = self._make_builder(tmpdir, include_column_comments=False, include_stats=False)
            bare_row = bare_builder.build_row(raw, force_rebuild=True)

        self.assertIn("score is the metric value", full_row["prompt"][1]["content"])
        self.assertIn("Stats:", full_row["prompt"][1]["content"])
        self.assertEqual(full_builder.schema_cache.build_count, 1)
        self.assertEqual(full_again["prompt"][1]["content"], full_row["prompt"][1]["content"])
        self.assertNotIn("score is the metric value", bare_row["prompt"][1]["content"])
        self.assertNotIn("Stats:", bare_row["prompt"][1]["content"])

    def test_tool_modes_and_named_skill_headers(self):
        raw = {
            "db_id": "toy",
            "question": "List names.",
            "evidence": "",
            "SQL": "SELECT name FROM demo",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            consensus = self._make_builder(tmpdir, tool_mode="consensus")
            consensus_row = consensus.build_row(raw, force_rebuild=True)
            skilled = self._make_builder(tmpdir)
            row0 = skilled.build_row(raw, force_rebuild=True, skill_name="default")
            row1 = skilled.build_row(raw, force_rebuild=True, skill_name="decompose-first")

        self.assertIn("consensus_at_1", [tool["function"]["name"] for tool in consensus_row["tools"]])
        self.assertEqual(row0["skill_name"], "default")
        self.assertEqual(row1["skill_name"], "decompose-first")
        self.assertTrue(row1["prompt"][0]["content"].startswith("SKILL: DECOMPOSE-FIRST"))

    def test_runtime_bm25_fewshots_are_added_for_raw_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            train_file = tmp_path / "train.jsonl"
            train_file.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "db_id": "other",
                                "question": "Find Alice score",
                                "evidence": "Alice score uses score",
                                "SQL": "SELECT score FROM demo WHERE name = 'Alice'",
                            }
                        ),
                        json.dumps(
                            {
                                "db_id": "other",
                                "question": "Count rows",
                                "evidence": "",
                                "SQL": "SELECT COUNT(*) FROM demo",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            builder = self._make_builder(tmpdir, fewshot_train_file=str(train_file), fewshot_top_n=1)
            row = builder.build_row(
                {
                    "db_id": "toy",
                    "question": "What is Alice's score?",
                    "evidence": "Alice refers to name Alice",
                    "SQL": "SELECT score FROM demo WHERE name = 'Alice'",
                },
                force_rebuild=True,
            )

        self.assertIn("- Example 1", row["prompt"][1]["content"])
        self.assertIn("Find Alice score", row["prompt"][1]["content"])

    def test_no_fewshots_does_not_require_train_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = self._make_builder(
                tmpdir,
                include_fewshots=False,
                fewshot_train_file=str(Path(tmpdir) / "missing.jsonl"),
            )
            row = builder.build_row(
                {
                    "db_id": "toy",
                    "question": "List names.",
                    "evidence": "",
                    "SQL": "SELECT name FROM demo",
                },
                force_rebuild=True,
            )

        self.assertNotIn("- Example 1", row["prompt"][1]["content"])

    def test_missing_fewshot_train_file_fails_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = self._make_builder(tmpdir, fewshot_train_file=str(Path(tmpdir) / "missing.jsonl"))
            with self.assertRaises(FileNotFoundError):
                builder.build_row(
                    {
                        "db_id": "toy",
                        "question": "List names.",
                        "evidence": "",
                        "SQL": "SELECT name FROM demo",
                    },
                    force_rebuild=True,
                )

    def test_prebuilt_prompt_is_not_modified_by_runtime_fewshots(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = self._make_builder(tmpdir, fewshot_train_file=str(Path(tmpdir) / "missing.jsonl"))
            row = builder.build_row(
                {
                    "db_id": "toy",
                    "gold_sql": "SELECT 1",
                    "prompt": [
                        {"role": "system", "content": "system"},
                        {"role": "user", "content": "user without fewshots"},
                    ],
                },
                force_rebuild=False,
            )

        self.assertEqual(row["prompt"][1]["content"], "user without fewshots")


class SamplePlanTests(unittest.TestCase):
    def test_parse_sample_plan_expands_deterministically(self):
        specs = expand_sample_plan(
            "default:2@0.8,decompose-first:1@0.7,default:1@0.0",
            num_generations=16,
            temperature=0.8,
            top_p=1.0,
        )

        self.assertEqual([spec.sample_id for spec in specs], [0, 1, 2, 3])
        self.assertEqual([spec.skill_name for spec in specs], ["default", "default", "decompose-first", "default"])
        self.assertEqual([spec.temperature for spec in specs], [0.8, 0.8, 0.7, 0.0])

    def test_default_sample_plan_uses_num_generations(self):
        specs = expand_sample_plan("", num_generations=3, temperature=0.5, top_p=0.9)

        self.assertEqual(len(specs), 3)
        self.assertTrue(all(spec.skill_name == "default" for spec in specs))
        self.assertTrue(all(spec.temperature == 0.5 for spec in specs))
        self.assertTrue(all(spec.top_p == 0.9 for spec in specs))

    def test_sample_plan_rejects_invalid_entries(self):
        with self.assertRaises(ValueError):
            expand_sample_plan("unknown:1@0.8", num_generations=1, temperature=0.8, top_p=1.0)
        with self.assertRaises(ValueError):
            expand_sample_plan("default:0@0.8", num_generations=1, temperature=0.8, top_p=1.0)
        with self.assertRaises(ValueError):
            expand_sample_plan("default:1@0.8/1.5", num_generations=1, temperature=0.8, top_p=1.0)


class SchemaBuildTests(unittest.TestCase):
    def test_format_user_prompt_omits_fewshot_preamble_when_disabled(self):
        prompt = format_user_prompt(
            question="Count rows.",
            hint="use COUNT(*)",
            db_schema="`db`\n【Schema】\n# Table: t\n[(id:NUMERIC), Primary Key]",
            db_id="db",
            few_shot_examples=[
                {"db_id": "x", "question": "q", "evidence": "h", "SQL": "SELECT 1"}
            ],
            include_fewshots=False,
        )

        self.assertTrue(prompt.startswith("\n<question>\nCount rows."))
        self.assertNotIn("Use the examples only", prompt)
        self.assertNotIn("- Example 1", prompt)
        self.assertIn("<hint>\nuse COUNT(*)\n</hint>", prompt)
        self.assertIn("<database_schema>", prompt)
        self.assertIn("<db_id>db</db_id>", prompt)

    def test_build_mschema_examples_use_frequent_values_and_truncate_long_strings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "toy.sqlite"
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute("CREATE TABLE demo (num INTEGER, day TEXT, note TEXT)")
            long_note = "L" * 60
            rows = [
                (1, "2024-01-02", long_note),
                (1, "2024-01-02", long_note),
                (1, "2024-01-02", long_note),
                (2, "2024-01-01", "short"),
                (2, "2024-01-01", "short"),
                (3, "2024-01-03", "medium"),
            ]
            cur.executemany("INSERT INTO demo VALUES (?, ?, ?)", rows)
            conn.commit()
            conn.close()

            ms = build_mschema_from_db(str(db_path), "toy", meanings={}, example_num=3, include_stats=True)
            rendered = ms.to_mschema(example_num=3)

        self.assertIn("(num:NUMERIC), Nullable, Stats:", rendered)
        self.assertIn("Examples: [1, 2, 3]", rendered)
        self.assertNotIn("Examples: [1, 1.5, 3]", rendered)
        self.assertIn("Examples: [2024-01-02, 2024-01-01, 2024-01-03]", rendered)
        self.assertIn("Examples: [LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL..., short, medium]", rendered)

    def test_build_output_entry_supports_messages_only_rows(self):
        entry = build_output_entry(
            db_id="toy",
            gold_sql="SELECT 1",
            hint="hint",
            question="question",
            user_msg="<question>question</question>",
            messages_only=True,
        )

        self.assertEqual(sorted(entry.keys()), ["messages"])
        self.assertEqual([message["role"] for message in entry["messages"]], ["system", "user", "assistant"])

    def test_to_mschema_can_omit_nullability_labels(self):
        ms = MSchema(db_id="toy")
        ms.add_table("demo")
        ms.add_field("demo", "id", field_type="NUMERIC", primary_key=True, nullable=False)
        ms.add_field("demo", "name", field_type="TEXT", nullable=True)

        rendered = ms.to_mschema(example_num=0, include_nullability=False)

        self.assertNotIn("Nullable", rendered)
        self.assertNotIn("Not Null", rendered)
        self.assertIn("(id:NUMERIC), Primary Key)", rendered)

    def test_shard_rows_for_data_parallel_round_robins_rows(self):
        rows = [{"source_idx": idx} for idx in range(5)]

        shards = shard_rows_for_data_parallel(rows, num_shards=2)

        self.assertEqual([row["source_idx"] for row in shards[0]], [0, 2, 4])
        self.assertEqual([row["source_idx"] for row in shards[1]], [1, 3])


class SelfConsistencyScriptTests(unittest.TestCase):
    def test_rows_to_vote_signature_hashes_rows_as_unordered_set(self):
        rows_a = [(1, "A"), (2, "B")]
        rows_b = [(2, "B"), (1, "A")]

        self.assertEqual(rows_to_vote_signature(rows_a), rows_to_vote_signature(rows_b))

    def test_choose_majority_vote_candidate_ignores_empty_results(self):
        candidates = [
            {
                "sample_idx": 0,
                "pred_sql": "SELECT 1;",
                "pred_executed": True,
                "pred_rows": [],
            },
            {
                "sample_idx": 1,
                "pred_sql": "SELECT name FROM users;",
                "pred_executed": True,
                "pred_rows": [("alice",)],
            },
            {
                "sample_idx": 2,
                "pred_sql": "SELECT username FROM users;",
                "pred_executed": True,
                "pred_rows": [("alice",)],
            },
        ]

        winner, meta = choose_majority_vote_candidate(candidates)

        self.assertIsNotNone(winner)
        self.assertEqual(winner["sample_idx"], 1)
        self.assertEqual(meta["ignored_empty_results"], 1)
        self.assertEqual(meta["winning_vote_count"], 2)

    def test_choose_majority_vote_candidate_returns_none_without_valid_votes(self):
        candidates = [
            {
                "sample_idx": 0,
                "pred_sql": "SELECT 1;",
                "pred_executed": False,
                "pred_rows": None,
            },
            {
                "sample_idx": 1,
                "pred_sql": "SELECT 2;",
                "pred_executed": True,
                "pred_rows": [],
            },
        ]

        winner, meta = choose_majority_vote_candidate(candidates)

        self.assertIsNone(winner)
        self.assertEqual(meta["num_valid_votes"], 0)
        self.assertEqual(meta["ignored_empty_results"], 1)


if __name__ == "__main__":
    unittest.main()
