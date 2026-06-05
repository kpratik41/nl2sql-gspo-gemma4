from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from nl2sql_gspo.data import normalize_record
from nl2sql_gspo.sample_plan import skill_header_map, skill_name_from_header
from nl2sql_gspo.sql_utils import extract_sql, get_database_path
from nl2sql_gspo.tool_calling import get_tool_definitions, tool_catalog_compact

from prompts import SYSTEM_PROMPT_TEMPLATES, SYSTEM_PROMPT_TEMPLATES_CONSENSUS
from scripts.data_generation.schema_build import (
    SYSTEM_PROMPT as SCHEMA_SYSTEM_PROMPT,
    build_mschema_from_db,
    format_user_prompt,
    load_column_meanings,
)


@dataclass(frozen=True)
class PromptConfig:
    bird_mode: str = "dev"
    database_dir: str = "databases/dev_databases"
    meanings_file: str = "data/bird_dev_data/raw/column_meaning.json"
    include_column_comments: bool = True
    include_fewshots: bool = True
    include_stats: bool = True
    include_nullability: bool = True
    example_num: int = 3
    fewshot_train_file: str = "data/bird_train_data/raw/train-6601.jsonl"
    fewshot_top_n: int = 5
    tool_mode: str = "default"
    prompt_template: str = "default"
    build_prompts_at_runtime: bool = False
    raw_input_file: str = ""


def read_json_or_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    input_path = Path(path)
    if input_path.suffix.lower() == ".jsonl":
        with input_path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with input_path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    return loaded if isinstance(loaded, list) else [loaded]


def add_prompt_args(parser) -> None:
    parser.add_argument("--bird_mode", choices=["dev", "test"], default="dev")
    parser.add_argument(
        "--build_prompts_at_runtime",
        action="store_true",
        help="Build prompts from raw BIRD rows during inference instead of requiring prebuilt prompt JSONL.",
    )
    parser.add_argument(
        "--raw_input_file",
        default="",
        help="Raw BIRD json/jsonl file to use for runtime prompt generation. Overrides --input_file when set.",
    )
    parser.add_argument(
        "--meanings_file",
        default="data/bird_dev_data/raw/column_meaning.json",
        help="Column meanings JSON used when runtime-building schema prompts.",
    )
    parser.add_argument("--include_column_comments", dest="include_column_comments", action="store_true", default=True)
    parser.add_argument("--no_column_comments", dest="include_column_comments", action="store_false")
    parser.add_argument("--include_fewshots", dest="include_fewshots", action="store_true", default=True)
    parser.add_argument("--no_fewshots", dest="include_fewshots", action="store_false")
    parser.add_argument("--fewshot_train_file", default="data/bird_train_data/raw/train-6601.jsonl")
    parser.add_argument("--fewshot_top_n", type=int, default=5)
    parser.add_argument("--include_stats", dest="include_stats", action="store_true", default=True)
    parser.add_argument("--no_stats", dest="include_stats", action="store_false")
    parser.add_argument("--include_nullability", dest="include_nullability", action="store_true", default=True)
    parser.add_argument("--no_nullability", dest="include_nullability", action="store_false")
    parser.add_argument("--example_num", type=int, default=3)
    parser.add_argument("--tool_mode", choices=["none", "default", "consensus"], default="default")
    parser.add_argument("--prompt_template", choices=["default", "consensus"], default="default")


def prompt_config_from_args(args) -> PromptConfig:
    return PromptConfig(
        bird_mode=args.bird_mode,
        database_dir=args.database_dir,
        meanings_file=args.meanings_file,
        include_column_comments=args.include_column_comments,
        include_fewshots=args.include_fewshots,
        include_stats=args.include_stats,
        include_nullability=args.include_nullability,
        example_num=args.example_num,
        fewshot_train_file=args.fewshot_train_file,
        fewshot_top_n=args.fewshot_top_n,
        tool_mode=args.tool_mode,
        prompt_template=args.prompt_template,
        build_prompts_at_runtime=args.build_prompts_at_runtime,
        raw_input_file=args.raw_input_file,
    )


def _tokenize_for_bm25(text: str) -> List[str]:
    import re

    return re.findall(r"\w+", text.lower())


class FewShotRetriever:
    def __init__(self, train_file: str | Path, top_n: int):
        if top_n <= 0:
            raise ValueError("--fewshot_top_n must be >= 1")
        self.train_file = Path(train_file)
        if not self.train_file.exists():
            raise FileNotFoundError(
                f"Few-shot train file not found: {self.train_file}. "
                "Pass --no_fewshots or set --fewshot_train_file."
            )
        self.top_n = int(top_n)
        self.train_data = read_json_or_jsonl(self.train_file)
        try:
            from rank_bm25 import BM25Okapi
        except Exception as exc:
            print(f"[fewshot] rank-bm25 unavailable; using built-in BM25 retrieval: {exc}")
            BM25Okapi = _SimpleBM25
        corpus = [
            _tokenize_for_bm25(f"{item.get('question', '')} {item.get('evidence', '')}")
            for item in self.train_data
        ]
        self.bm25 = BM25Okapi(corpus)

    def retrieve(self, row: Dict[str, Any]) -> List[Dict[str, Any]]:
        query_text = f"{row.get('question', '')} {row.get('evidence', '') or row.get('external_knowledge', '') or row.get('hint', '')}".strip()
        scores = self.bm25.get_scores(_tokenize_for_bm25(query_text))
        ranked = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[: self.top_n]
        fewshots: List[Dict[str, Any]] = []
        for rank, index in enumerate(ranked, start=1):
            train_item = self.train_data[index]
            fewshots.append(
                {
                    "rank": rank,
                    "bm25_score": round(float(scores[index]), 4),
                    "db_id": train_item.get("db_id", ""),
                    "question": train_item.get("question", ""),
                    "evidence": train_item.get("evidence", ""),
                    "SQL": train_item.get("SQL", train_item.get("gold_sql", "")),
                }
            )
        return fewshots


class _SimpleBM25:
    def __init__(self, corpus: List[List[str]]):
        self.corpus = corpus
        self.doc_count = len(corpus)
        self.avgdl = sum(len(doc) for doc in corpus) / max(1, self.doc_count)
        self.doc_freq: Dict[str, int] = {}
        for doc in corpus:
            for token in set(doc):
                self.doc_freq[token] = self.doc_freq.get(token, 0) + 1

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        k1 = 1.5
        b = 0.75
        query = set(query_tokens)
        scores: List[float] = []
        for doc in self.corpus:
            doc_len = len(doc)
            term_counts: Dict[str, int] = {}
            for token in doc:
                term_counts[token] = term_counts.get(token, 0) + 1
            score = 0.0
            for token in query:
                freq = term_counts.get(token, 0)
                if not freq:
                    continue
                df = self.doc_freq.get(token, 0)
                idf = math.log(1 + (self.doc_count - df + 0.5) / (df + 0.5))
                denom = freq + k1 * (1 - b + b * doc_len / max(self.avgdl, 1e-9))
                score += idf * freq * (k1 + 1) / denom
            scores.append(score)
        return scores


class SchemaCache:
    def __init__(self, config: PromptConfig):
        self.config = config
        self._schemas: Dict[Tuple[Any, ...], str] = {}
        self._meanings: Optional[Dict[str, str]] = None

    @property
    def build_count(self) -> int:
        return len(self._schemas)

    def _load_meanings(self) -> Dict[str, str]:
        if self._meanings is not None:
            return self._meanings
        if not self.config.include_column_comments or not self.config.meanings_file:
            self._meanings = {}
            return self._meanings
        meanings_path = Path(self.config.meanings_file)
        self._meanings = load_column_meanings(str(meanings_path)) if meanings_path.exists() else {}
        return self._meanings

    def get_schema(self, db_id: str) -> str:
        key = (
            db_id,
            self.config.database_dir,
            self.config.meanings_file if self.config.include_column_comments else "",
            self.config.include_column_comments,
            self.config.include_stats,
            self.config.include_nullability,
            self.config.example_num,
        )
        if key in self._schemas:
            return self._schemas[key]

        db_path = get_database_path(db_id, self.config.database_dir)
        if not db_path:
            rendered = "Schema not available."
        else:
            mschema = build_mschema_from_db(
                db_path=db_path,
                db_id=db_id,
                meanings=self._load_meanings(),
                example_num=self.config.example_num,
                include_stats=self.config.include_stats,
            )
            rendered = mschema.to_mschema(
                example_num=self.config.example_num,
                include_nullability=self.config.include_nullability,
            )
        self._schemas[key] = rendered
        return rendered


class PromptBuilder:
    def __init__(self, config: PromptConfig):
        self.config = config
        self.schema_cache = SchemaCache(config)
        self._skill_headers = skill_header_map()
        self._fewshot_retriever: Optional[FewShotRetriever] = None

    def build_rows(
        self,
        rows: List[Dict[str, Any]],
        *,
        force_rebuild: bool = False,
        sample_id: Optional[int] = None,
        skill_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return [
            self.build_row(row, force_rebuild=force_rebuild, sample_id=sample_id, skill_name=skill_name)
            for row in rows
        ]

    def build_row(
        self,
        row: Dict[str, Any],
        *,
        force_rebuild: bool = False,
        sample_id: Optional[int] = None,
        skill_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not force_rebuild and self._has_prompt(row):
            prepared = normalize_record(row)
            if row.get("prompt"):
                prepared["prompt"] = row["prompt"]
                prepared["messages"] = row["prompt"]
            elif row.get("messages"):
                prepared["messages"] = row["messages"]
                prepared["prompt"] = row["messages"]
            prepared["question_id"] = row.get("question_id", prepared.get("question_id"))
            prepared["tools"] = row.get("tools") or prepared.get("tools") or []
            return self._apply_skill(prepared, sample_id, skill_name)

        db_id = str(row.get("db_id") or row.get("database") or row.get("database_name") or "").strip()
        question = str(row.get("question") or "").strip()
        evidence = str(row.get("evidence") or row.get("external_knowledge") or row.get("hint") or "").strip()
        gold_sql = extract_sql(row.get("gold_sql") or row.get("SQL") or row.get("query") or row.get("sql") or "")
        schema = self.schema_cache.get_schema(db_id)
        few_shot_examples = self._fewshots_for_row(row) if self.config.include_fewshots else []
        user_prompt = format_user_prompt(
            question=question,
            hint=evidence,
            db_schema=schema,
            db_id=db_id,
            few_shot_examples=few_shot_examples,
            include_fewshots=self.config.include_fewshots,
        )
        system_prompt = self._system_prompt()
        prompt_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        tools = self._tools()
        prepared = {
            "db_id": db_id,
            "gold_sql": gold_sql,
            "evidence": evidence,
            "question": question,
            "question_id": row.get("question_id"),
            "difficulty": row.get("difficulty", ""),
            "prompt": prompt_messages,
            "messages": prompt_messages,
            "tools": tools,
            "prompt_metadata": {
                **asdict(self.config),
                "runtime_built": True,
                "runtime_fewshots": bool(few_shot_examples),
                "schema_cache_build_count": self.schema_cache.build_count,
            },
        }
        return self._apply_skill(prepared, sample_id, skill_name)

    def _has_prompt(self, row: Dict[str, Any]) -> bool:
        return bool(row.get("prompt") or row.get("messages"))

    def _system_prompt(self) -> str:
        if self.config.tool_mode == "none":
            return SCHEMA_SYSTEM_PROMPT
        templates = {
            "default": SYSTEM_PROMPT_TEMPLATES,
            "consensus": SYSTEM_PROMPT_TEMPLATES_CONSENSUS,
        }
        template = templates[self.config.prompt_template]
        return template.replace("{TOOL_CATALOG_COMPACT}", tool_catalog_compact()).strip()

    def _tools(self) -> List[Dict[str, Any]]:
        if self.config.tool_mode == "none":
            return []
        return get_tool_definitions(include_consensus=self.config.tool_mode == "consensus")

    def _fewshots_for_row(self, row: Dict[str, Any]) -> List[Dict[str, Any]]:
        if self._fewshot_retriever is None:
            self._fewshot_retriever = FewShotRetriever(
                self.config.fewshot_train_file,
                self.config.fewshot_top_n,
            )
        return self._fewshot_retriever.retrieve(row)

    def _skill_for_name(self, skill_name: Optional[str]) -> Tuple[Optional[int], str, str]:
        name = (skill_name or "default").strip().lower().replace("_", "-")
        if name in {"", "default"}:
            return None, "default", ""
        if name not in self._skill_headers:
            allowed = ", ".join(sorted(self._skill_headers))
            raise ValueError(f"Unknown skill name {skill_name!r}. Allowed skills: {allowed}")
        skill_id, header = self._skill_headers[name]
        return skill_id, skill_name_from_header(header), header

    def _apply_skill(self, row: Dict[str, Any], sample_id: Optional[int], skill_name: Optional[str]) -> Dict[str, Any]:
        skill_id, resolved_skill_name, header = self._skill_for_name(skill_name)
        prepared = dict(row)
        messages = [dict(message) for message in (prepared.get("prompt") or prepared.get("messages") or [])]
        if header and messages and messages[0].get("role") == "system":
            messages[0]["content"] = header.strip() + "\n\n" + str(messages[0].get("content", ""))
        prepared["prompt"] = messages
        prepared["messages"] = messages
        prepared["skill_id"] = skill_id
        prepared["skill_name"] = resolved_skill_name
        metadata = dict(prepared.get("prompt_metadata") or {})
        metadata["skill_id"] = skill_id
        metadata["skill_name"] = resolved_skill_name
        prepared["prompt_metadata"] = metadata
        return prepared
