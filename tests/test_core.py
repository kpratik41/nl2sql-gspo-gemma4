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
from nl2sql_gspo.dynamic_sampling_trainer import _is_truncated_completion
from nl2sql_gspo.rewards import make_nl2sql_rewards
from nl2sql_gspo.train_gspo_nl2sql import filter_by_prompt_length, parse_args
from nl2sql_gspo.sql_utils import (
    _DB_CONNECTIONS,
    _BIRD_GOLD_CACHE,
    bird_execute_sql,
    bird_get_gold_rows,
    bird_result_match,
    extract_sql,
    get_database_path,
    is_safe_readonly_sql,
)
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
    def test_prompt_filter_counts_tokenizer_encoding_objects(self):
        from datasets import Dataset

        class FakeEncoding:
            def __init__(self, token_count):
                self.ids = list(range(token_count))

        class FakeBatchEncoding:
            def __init__(self, token_count):
                self.encodings = [FakeEncoding(token_count)]

            def __len__(self):
                return 2

        class FakeTokenizer:
            def apply_chat_template(self, prompt, tokenize=True, add_generation_prompt=True):
                return FakeBatchEncoding(prompt[0]["tokens"])

        dataset = Dataset.from_list(
            [
                {"prompt": [{"role": "user", "content": "short", "tokens": 10}]},
                {"prompt": [{"role": "user", "content": "long", "tokens": 30}]},
            ]
        )

        filtered = filter_by_prompt_length(dataset, FakeTokenizer(), 20, "test")

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["prompt"][0]["content"], "short")

    def test_truncation_mask_only_flags_completions_at_max_length(self):
        eos_and_pad = [1, 0]

        self.assertFalse(_is_truncated_completion([10, 11, 12], eos_and_pad, 4096))
        self.assertFalse(_is_truncated_completion([10, 11, 1], eos_and_pad, 3))
        self.assertTrue(_is_truncated_completion([10, 11, 12], eos_and_pad, 3))

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
                "--save_only_model",
                "--resume_from_checkpoint",
                "outputs/run/checkpoint-100",
                "--num_iterations",
                "2",
                "--enable_dynamic_sampling",
                "--mask_truncated_completions",
                "--dapo_max_rounds",
                "3",
                "--exec_timeout_s",
                "45",
                "--length_penalty_max",
                "4096",
                "--length_penalty_buffer",
                "512",
            ]
        )

        self.assertEqual(args.resume_from_checkpoint, "outputs/run/checkpoint-100")
        self.assertEqual(args.train_limit, 123)
        self.assertEqual(args.eval_limit, 45)
        self.assertTrue(args.eval_on_start)
        self.assertTrue(args.save_only_model)
        self.assertEqual(args.num_iterations, 2)
        self.assertTrue(args.enable_dynamic_sampling)
        self.assertTrue(args.mask_truncated_completions)
        self.assertEqual(args.dapo_max_rounds, 3)
        self.assertEqual(args.exec_timeout_s, 45.0)
        self.assertEqual(args.length_penalty_max, 4096)
        self.assertEqual(args.length_penalty_buffer, 512)


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

        self.reward_functions = make_nl2sql_rewards(
            str(self.database_root),
            length_penalty_max=100,
            length_penalty_buffer=20,
        )
        self.format_reward = self.reward_functions[0]
        self.execution_reward = self.reward_functions[1]
        self.result_reward = self.reward_functions[2]
        self.table_linking_reward = self.reward_functions[3]
        self.column_linking_reward = self.reward_functions[4]
        self.nonnull_reward = self.reward_functions[5]
        self.length_penalty_reward = self.reward_functions[6]

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
        _BIRD_GOLD_CACHE.clear()
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
        unrelated_completion = ["SELECT id FROM orders WHERE age < 10;"]

        gold_kwargs = {"gold_sql": self.common_kwargs["gold_sql"]}

        exact_table = self.table_linking_reward(completions=exact_completion, **gold_kwargs)[0]
        partial_table = self.table_linking_reward(completions=partial_completion, **gold_kwargs)[0]
        unrelated_table = self.table_linking_reward(completions=unrelated_completion, **gold_kwargs)[0]

        exact_col = self.column_linking_reward(completions=exact_completion, **gold_kwargs)[0]
        partial_col = self.column_linking_reward(completions=partial_completion, **gold_kwargs)[0]

        self.assertEqual(exact_table, 1.0)
        self.assertEqual(partial_table, 1.0)
        self.assertEqual(unrelated_table, 0.0)
        self.assertGreater(exact_col, partial_col)

    def test_format_reward_requires_strict_xml_shape(self):
        good = (
            "<scratch_pad>reasoning</scratch_pad>\n"
            "<final_answer>\n<sql_code>SELECT 1</sql_code>\n</final_answer>"
        )
        missing_scratch = "<final_answer><sql_code>SELECT 1</sql_code></final_answer>"
        plain_sql = "SELECT name FROM users WHERE age > 30;"

        self.assertEqual(self.format_reward(completions=[good])[0], 1.0)
        self.assertEqual(self.format_reward(completions=[missing_scratch])[0], 0.0)
        self.assertEqual(self.format_reward(completions=[plain_sql])[0], 0.0)

    def test_nonnull_reward_only_fires_when_results_have_data(self):
        non_empty = ["SELECT name FROM users WHERE age > 30;"]
        empty = ["SELECT name FROM users WHERE age > 999;"]
        bad = ["SELECT missing FROM users;"]

        self.assertEqual(self.nonnull_reward(completions=non_empty, db_id=[self.db_id])[0], 1.0)
        self.assertEqual(self.nonnull_reward(completions=empty, db_id=[self.db_id])[0], 0.0)
        self.assertEqual(self.nonnull_reward(completions=bad, db_id=[self.db_id])[0], 0.0)

    def test_bird_result_match_uses_set_semantics_on_raw_rows(self):
        # Same set of rows but different order and duplicates -> equal under BIRD set match.
        gold_ok, gold_set, _ = bird_get_gold_rows(
            gold_sql="SELECT name FROM users WHERE age > 20;",
            db_id=self.db_id,
            database_dir=str(self.database_root),
        )
        self.assertTrue(gold_ok)

        pred_ok, pred_rows, _ = bird_execute_sql(
            sql="SELECT name FROM users WHERE age > 20 ORDER BY age DESC;",
            db_id=self.db_id,
            database_dir=str(self.database_root),
        )
        self.assertTrue(pred_ok)
        self.assertTrue(bird_result_match(pred_rows, gold_set))

    def test_result_reward_uses_bird_semantics(self):
        # Reordered duplicate-free results should match.
        completions = [
            "SELECT name FROM users WHERE age > 20 ORDER BY age DESC;",
            "SELECT name FROM users WHERE age > 999;",  # empty -> mismatch
        ]
        scores = self.result_reward(
            completions=completions,
            db_id=[self.db_id, self.db_id],
            gold_sql=["SELECT name FROM users WHERE age > 20;"] * 2,
        )
        self.assertEqual(scores, [1.0, 0.0])

    def test_length_penalty_implements_dapo_soft_overlong(self):
        # Configured: length_penalty_max=100, buffer=20 (word-count proxy).
        short = "word " * 50  # below soft threshold (80) -> 0
        edge = "word " * 80  # exactly at soft threshold -> 0
        ramp = "word " * 90  # midway in ramp: (80-90)/20 = -0.5
        cap = "word " * 100  # at L_max: (80-100)/20 = -1.0
        over = "word " * 200  # beyond L_max -> -1.0

        scores = self.length_penalty_reward(completions=[short, edge, ramp, cap, over])
        self.assertEqual(scores[0], 0.0)
        self.assertEqual(scores[1], 0.0)
        self.assertAlmostEqual(scores[2], -0.5, places=6)
        self.assertAlmostEqual(scores[3], -1.0, places=6)
        self.assertEqual(scores[4], -1.0)

    def test_length_penalty_uses_tokenizer_when_available(self):
        class FakeTokenizer:
            def encode(self, text, add_special_tokens=False):
                # Pretend each char is a token so we can hit ramps deterministically.
                return list(text)

        rewards = make_nl2sql_rewards(
            str(self.database_root),
            tokenizer=FakeTokenizer(),
            length_penalty_max=10,
            length_penalty_buffer=5,
        )
        length_reward = rewards[6]

        self.assertEqual(length_reward(completions=["abc"])[0], 0.0)  # 3 ≤ 5
        self.assertAlmostEqual(length_reward(completions=["abcdefgh"])[0], -0.6, places=6)  # 8 -> (5-8)/5
        self.assertEqual(length_reward(completions=["a" * 20])[0], -1.0)

    def test_execution_cache_dedupes_repeated_predicted_sql(self):
        from nl2sql_gspo import rewards as rewards_module

        completions = [
            "SELECT name FROM users WHERE age > 30;",
            "SELECT name FROM users WHERE age > 30;",  # duplicate -> cache hit
            "SELECT name FROM users WHERE age > 30;",  # duplicate -> cache hit
        ]
        gold_sqls = ["SELECT name FROM users WHERE age > 30;"] * 3
        db_ids = [self.db_id] * 3

        with mock.patch.object(
            rewards_module, "bird_execute_sql", wraps=rewards_module.bird_execute_sql
        ) as spy:
            # Build a fresh closure so the cache is empty for this assertion.
            rewards = make_nl2sql_rewards(str(self.database_root))
            execution_reward = rewards[1]
            result_reward = rewards[2]
            nonnull_reward = rewards[5]

            execution_reward(completions=completions, db_id=db_ids)
            result_reward(completions=completions, db_id=db_ids, gold_sql=gold_sqls)
            nonnull_reward(completions=completions, db_id=db_ids)

        # Without caching: 3 completions × 3 rewards = 9 predicted-SQL executions.
        # With caching keyed on (db_id, sql), only the first execution actually
        # hits the database; the other 8 are cache hits.
        self.assertEqual(spy.call_count, 1)

    def test_execution_cache_separates_distinct_sql(self):
        rewards = make_nl2sql_rewards(str(self.database_root))
        execution_reward = rewards[1]

        scores = execution_reward(
            completions=[
                "SELECT name FROM users WHERE age > 30;",
                "SELECT missing_column FROM users;",
                "SELECT name FROM users WHERE age > 30;",  # cache hit
            ],
            db_id=[self.db_id] * 3,
        )
        self.assertEqual(scores, [1.0, 0.0, 1.0])


class DynamicSamplingTrainerHelperTests(unittest.TestCase):
    """Unit-tests for the pure helpers on `DynamicSamplingGRPOTrainer`.

    We avoid TRL's heavyweight `__init__` by constructing instances via
    `object.__new__` and setting only the attributes the helpers touch.
    Skipped automatically when TRL or torch isn't installed in the env.
    """

    @classmethod
    def setUpClass(cls):
        try:
            import torch  # noqa: F401
            from nl2sql_gspo.dynamic_sampling_trainer import (  # noqa: F401
                DynamicSamplingGRPOTrainer,
            )
        except Exception as exc:  # ModuleNotFoundError or torch import error
            raise unittest.SkipTest(f"TRL/torch not available: {exc}")

    def _make_bare_trainer(
        self,
        num_generations=4,
        min_std=1e-6,
        pad_token_id=0,
        process_index=0,
        num_processes=1,
    ):
        import torch
        from nl2sql_gspo.dynamic_sampling_trainer import DynamicSamplingGRPOTrainer

        trainer = object.__new__(DynamicSamplingGRPOTrainer)
        trainer.num_generations = num_generations
        trainer.dynamic_sampling_min_std = min_std
        trainer.dynamic_sampling_reward_name = None
        trainer._dyn_reward_idx = None
        trainer.pad_token_id = pad_token_id
        trainer.train_dataset = None
        trainer._dyn_pool_indices = []
        trainer._dyn_pool_cursor = 0
        trainer._dyn_pool_pass = 0

        class _FakeArgs:
            seed = 0

        trainer.args = _FakeArgs()

        # Minimal stand-in for self.accelerator with a fixed process_index.
        trainer.accelerator = type(
            "_FakeAccel",
            (),
            {
                "process_index": process_index,
                "num_processes": num_processes,
                "is_main_process": True,
            },
        )()
        return trainer, torch

    def test_pad_to_width_right_pads_with_value(self):
        from nl2sql_gspo.dynamic_sampling_trainer import _pad_to_width
        import torch

        x = torch.tensor([[1, 2, 3], [4, 5, 6]])
        padded = _pad_to_width(x, target_width=5, pad_value=-1, side="right")
        self.assertEqual(padded.tolist(), [[1, 2, 3, -1, -1], [4, 5, 6, -1, -1]])

    def test_pad_to_width_left_pads_with_value(self):
        from nl2sql_gspo.dynamic_sampling_trainer import _pad_to_width
        import torch

        x = torch.tensor([[1, 2], [3, 4]])
        padded = _pad_to_width(x, target_width=4, pad_value=9, side="left")
        self.assertEqual(padded.tolist(), [[9, 9, 1, 2], [9, 9, 3, 4]])

    def test_pad_to_width_noop_when_already_wide(self):
        from nl2sql_gspo.dynamic_sampling_trainer import _pad_to_width
        import torch

        x = torch.tensor([[1, 2, 3]])
        padded = _pad_to_width(x, target_width=2, pad_value=0, side="right")
        self.assertIs(padded, x)

    def test_build_round_inputs_replicates_each_prompt_num_generations_times(self):
        trainer, _ = self._make_bare_trainer(num_generations=3)
        out = trainer._build_round_inputs([{"id": "a"}, {"id": "b"}])
        self.assertEqual([r["id"] for r in out], ["a", "a", "a", "b", "b", "b"])
        # Each replica must be an independent dict so per-row mutation
        # by reward fns doesn't leak across rollouts.
        self.assertIsNot(out[0], out[1])

    def test_extract_groups_picks_correct_rows_and_carries_through_scalars(self):
        trainer, torch = self._make_bare_trainer(num_generations=2)
        round_out = {
            "prompt_ids": torch.tensor([[1, 1], [1, 1], [2, 2], [2, 2], [3, 3], [3, 3]]),
            "advantages": torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            "completion_mask": torch.ones(6, 4, dtype=torch.long),
            "num_items_in_batch": 99,
        }
        chunk = trainer._extract_groups(round_out, [0, 2])
        self.assertEqual(chunk["_n_groups"], 2)
        self.assertEqual(chunk["prompt_ids"].tolist(), [[1, 1], [1, 1], [3, 3], [3, 3]])
        for got, want in zip(chunk["advantages"].tolist(), [0.1, 0.2, 0.5, 0.6]):
            self.assertAlmostEqual(got, want, places=5)
        self.assertEqual(chunk["num_items_in_batch"], 99)

    def test_extract_groups_with_zero_mask_zeros_completion_mask(self):
        trainer, torch = self._make_bare_trainer(num_generations=2)
        round_out = {
            "prompt_ids": torch.tensor([[1, 1], [1, 1], [2, 2], [2, 2]]),
            "completion_mask": torch.ones(4, 3, dtype=torch.long),
        }
        chunk = trainer._extract_groups(round_out, [1], zero_mask=True)
        self.assertEqual(chunk["completion_mask"].sum().item(), 0)
        self.assertEqual(chunk["prompt_ids"].tolist(), [[2, 2], [2, 2]])

    def test_extract_groups_returns_none_for_empty_index(self):
        trainer, _ = self._make_bare_trainer()
        self.assertIsNone(trainer._extract_groups({"prompt_ids": None}, []))

    def test_concat_chunks_pads_2d_tensors_and_concatenates(self):
        trainer, torch = self._make_bare_trainer(num_generations=2, pad_token_id=0)
        c1 = {
            "_n_groups": 1,
            "prompt_ids": torch.tensor([[5, 5], [5, 5]]),
            "completion_ids": torch.tensor([[9, 9], [9, 9]]),
            "completion_mask": torch.ones(2, 2, dtype=torch.long),
            "advantages": torch.tensor([0.1, -0.1]),
        }
        c2 = {
            "_n_groups": 1,
            "prompt_ids": torch.tensor([[6, 6, 6], [6, 6, 6]]),  # wider
            "completion_ids": torch.tensor([[7, 7, 7], [7, 7, 7]]),
            "completion_mask": torch.ones(2, 3, dtype=torch.long),
            "advantages": torch.tensor([0.5, -0.5]),
        }
        final = trainer._concat_chunks([c1, c2])
        # prompt_ids is left-padded with pad_token_id=0
        self.assertEqual(
            final["prompt_ids"].tolist(),
            [[0, 5, 5], [0, 5, 5], [6, 6, 6], [6, 6, 6]],
        )
        # completion_ids is right-padded
        self.assertEqual(final["completion_ids"].tolist(), [[9, 9, 0], [9, 9, 0], [7, 7, 7], [7, 7, 7]])
        for got, want in zip(final["advantages"].tolist(), [0.1, -0.1, 0.5, -0.5]):
            self.assertAlmostEqual(got, want, places=5)

    def test_draw_replacement_inputs_uses_shared_tail_queue(self):
        trainer, _ = self._make_bare_trainer()
        trainer.train_dataset = [{"id": i} for i in range(8)]
        trainer._dyn_pool_indices = list(range(8))
        trainer._dyn_pool_cursor = 8

        first = trainer._draw_replacement_inputs(3)
        second = trainer._draw_replacement_inputs(3)

        self.assertEqual([r["id"] for r in first], [7, 6, 5])
        self.assertEqual([r["id"] for r in second], [4, 3, 2])
        self.assertEqual(trainer._dyn_pool_cursor, 2)

    def test_draw_replacement_inputs_skips_other_ranks_consumed_tail_slices(self):
        trainer, _ = self._make_bare_trainer(process_index=0, num_processes=6)
        trainer.train_dataset = [{"id": i} for i in range(20)]
        trainer._dyn_pool_indices = list(range(20))
        trainer._dyn_pool_cursor = 20

        first = trainer._draw_replacement_inputs(2)
        second = trainer._draw_replacement_inputs(2)

        self.assertEqual([r["id"] for r in first], [19, 18])
        self.assertEqual([r["id"] for r in second], [7, 6])
        self.assertEqual(trainer._dyn_pool_cursor, 16)

        rank1, _ = self._make_bare_trainer(process_index=1, num_processes=6)
        rank1.train_dataset = [{"id": i} for i in range(20)]
        rank1._dyn_pool_indices = list(range(20))
        rank1._dyn_pool_cursor = 20
        self.assertEqual(
            [r["id"] for r in rank1._draw_replacement_inputs(2)],
            [17, 16],
        )

    def test_draw_replacement_inputs_wraps_only_after_tail_exhaustion(self):
        trainer, _ = self._make_bare_trainer()
        trainer.train_dataset = [{"id": i} for i in range(4)]
        trainer._dyn_pool_indices = list(range(4))
        trainer._dyn_pool_cursor = 2
        trainer._dyn_pool_pass = 99

        out = trainer._draw_replacement_inputs(3)

        self.assertEqual([r["id"] for r in out[:2]], [1, 0])
        self.assertEqual(len(out), 3)
        self.assertEqual(trainer._dyn_pool_pass, 100)

    def test_draw_replacement_inputs_returns_empty_when_no_dataset(self):
        trainer, _ = self._make_bare_trainer()
        self.assertEqual(trainer._draw_replacement_inputs(5), [])

    def test_add_policy_logps_skips_all_zero_policy_batches(self):
        trainer, torch = self._make_bare_trainer()
        trainer.use_vllm = True
        trainer.vllm_importance_sampling_correction = True
        trainer._get_per_token_logps_and_entropies = mock.Mock(
            side_effect=AssertionError("policy logps should be skipped")
        )
        output = {
            "completion_mask": torch.zeros(4, 3, dtype=torch.long),
            "_skip_policy_loss": torch.ones(4, dtype=torch.bool),
        }

        returned = trainer._add_policy_logps_for_kept(output)

        self.assertIs(returned, output)
        self.assertEqual(returned["importance_sampling_ratio"].tolist(), [[1.0] * 3] * 4)
        trainer._get_per_token_logps_and_entropies.assert_not_called()

    def test_iterative_dapo_uses_candidate_generation_before_policy_logps(self):
        from collections import defaultdict

        trainer, torch = self._make_bare_trainer(num_generations=2)
        trainer.enable_dynamic_sampling = True
        trainer.dapo_oversample_factor = 1
        trainer.dapo_max_rounds = 1
        trainer.model = type("_FakeModel", (), {"training": True})()
        trainer.use_vllm = True
        trainer.vllm_importance_sampling_correction = True
        trainer._metrics = {"train": defaultdict(list)}
        trainer.state = type("_FakeState", (), {"global_step": 0})()

        class _FakeAccel:
            process_index = 0
            num_processes = 1
            is_main_process = False
            device = torch.device("cpu")

            @staticmethod
            def gather(value):
                return value

        trainer.accelerator = _FakeAccel()
        trainer._get_per_token_logps_and_entropies = mock.Mock(
            side_effect=AssertionError("policy logps should be skipped for all-padded iterative batches")
        )
        round_out = {
            "prompt_ids": torch.tensor([[1, 1], [1, 1]]),
            "prompt_mask": torch.ones(2, 2, dtype=torch.long),
            "completion_ids": torch.tensor([[2, 3, 0], [2, 4, 0]]),
            "completion_mask": torch.ones(2, 3, dtype=torch.long),
            "advantages": torch.zeros(2),
            "sampling_per_token_logps": torch.zeros(2, 3),
            "num_items_in_batch": torch.tensor(6),
        }
        trainer._generate_and_score_candidates_no_policy_logps = mock.Mock(return_value=round_out)

        out = trainer._generate_and_score_completions([{"prompt": []}, {"prompt": []}])

        trainer._generate_and_score_candidates_no_policy_logps.assert_called_once()
        trainer._get_per_token_logps_and_entropies.assert_not_called()
        self.assertEqual(out["completion_mask"].sum().item(), 0)
        self.assertTrue(out["_skip_policy_loss"].all().item())
        self.assertEqual(out["importance_sampling_ratio"].tolist(), [[1.0] * 3] * 2)


class TrainScriptDefaultsTests(unittest.TestCase):
    def test_default_reward_weights_have_seven_entries(self):
        from nl2sql_gspo.train_gspo_nl2sql import DEFAULT_REWARD_WEIGHTS

        self.assertEqual(len(DEFAULT_REWARD_WEIGHTS), 7)
        # length-penalty weight should be small and non-negative (the reward
        # itself is in [-1, 0], so positive weight gives a non-positive signal).
        self.assertGreaterEqual(DEFAULT_REWARD_WEIGHTS[6], 0.0)

    def test_parse_reward_weights_rejects_wrong_arity(self):
        import argparse
        from nl2sql_gspo.train_gspo_nl2sql import parse_reward_weights

        with self.assertRaises(argparse.ArgumentTypeError):
            parse_reward_weights("0.1,0.2,0.3")  # too few

    def test_parse_args_defaults_match_dapo_recipe(self):
        from nl2sql_gspo.train_gspo_nl2sql import parse_args

        args = parse_args(
            [
                "--model_name_or_path", "model",
                "--train_file", "train.jsonl",
                "--database_dir", "databases",
                "--output_dir", "out",
            ]
        )
        # DAPO recipe: epsilon_high > epsilon, dapo loss, batch reward scaling.
        self.assertEqual(args.epsilon, 0.2)
        self.assertEqual(args.epsilon_high, 0.28)
        self.assertEqual(args.loss_type, "dapo")
        self.assertEqual(args.scale_rewards, "batch")
        self.assertEqual(args.beta, 0.0)
        self.assertEqual(args.exec_timeout_s, 60.0)
        self.assertEqual(args.length_penalty_max, 4096)
        self.assertEqual(args.length_penalty_buffer, 512)
        self.assertEqual(args.dapo_max_rounds, 6)


if __name__ == "__main__":
    unittest.main()