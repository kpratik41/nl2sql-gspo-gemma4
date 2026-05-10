#!/usr/bin/env python3
"""
build_jsonl.py - Build a JSONL fine-tuning dataset from BIRD-SQL dev split.

Usage:
    python build_jsonl.py [OPTIONS]

Options:
    --base-dir      PATH  Root of bird_sql folder              [default: auto-detected]
    --n-examples    INT   Number of examples to process        [default: 5, -1 = all]
    --example-num   INT   Max column examples in MSchema       [default: 3]
    --output        PATH  Output .jsonl path                   [default: <base-dir>/output.jsonl]
    --no-comments         Disable inline column comments        [flag]
    --no-fewshots         Disable few-shot examples             [flag]
    --no-stats            Disable column stats/examples         [flag]
"""

import argparse
import json
import os
from pathlib import Path
import re
import sqlite3
from typing import Any, Dict, List, Optional


# ================================================================
# 1. UTILITIES
# ================================================================

def examples_to_str(examples: list) -> list:
    """Deduplicate while preserving order, convert to str."""
    seen, result = set(), []
    for e in examples:
        s = str(e)
        if s not in seen:
            seen.add(s)
            result.append(s)
    return result


def read_json(file_path: str):
    path = Path(file_path)
    with path.open("r", encoding="utf-8") as f:
        # Train artifacts are JSONL while some dev-side sources are JSON arrays.
        # Accept both so the caller can switch datasets without changing loader code.
        if path.suffix.lower() == ".jsonl":
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


def format_stat_value(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, float):
        return str(round(value, 4))
    return str(value)


def truncate_values(values: List[Any], max_items: int = 3, max_length: int = 50) -> List[str]:
    rendered = [format_stat_value(value) for value in values if value is not None]
    if max_items > 0:
        rendered = rendered[:max_items]
    if max_length <= 0:
        return rendered
    if max_length <= 3:
        return [value[:max_length] for value in rendered]
    return [value if len(value) <= max_length else value[: max_length - 3] + "..." for value in rendered]


def get_top_occurring_values(cur, table: str, col_name: str, limit: int = 5) -> List[Any]:
    cur.execute(
        f"SELECT `{col_name}`, COUNT(*) AS cnt FROM `{table}` "
        f"WHERE `{col_name}` IS NOT NULL "
        f"GROUP BY `{col_name}` ORDER BY cnt DESC, `{col_name}` ASC LIMIT {int(limit)};"
    )
    return [row[0] for row in cur.fetchall()]


def resolve_default_paths(base_dir: Optional[str], split: str) -> Dict[str, str]:
    base_path = Path(base_dir) if base_dir else Path(__file__).resolve().parents[2]
    split_data_dir = base_path / "data" / f"bird_{split}_data" / "raw"
    split_db_dir = base_path / "databases" / f"{split}_databases"

    if split == "train":
        input_name = "train-6601-few-shot.jsonl"
        meanings_name = "train_column_meaning.json"
    else:
        input_name = "dev_20251106-few-shot.json"
        meanings_name = "column_meaning.json"

    # Prefer the repository layout used by this workspace, but retain compatibility
    # with the older standalone bird_sql directory layout this script started from.
    if split_data_dir.exists() and split_db_dir.exists():
        input_path = split_data_dir / input_name
        meanings_path = split_data_dir / meanings_name
        db_dir = split_db_dir
    else:
        input_path = base_path / ("train-6601-few-shot.jsonl" if split == "train" else "few_shot_bird_dev.json")
        meanings_path = base_path / meanings_name
        db_dir = base_path / f"{split}_databases"

    output_path = input_path.with_name(f"{input_path.stem}-schema.jsonl")
    return {
        "input_file": str(input_path),
        "meanings_file": str(meanings_path),
        "database_dir": str(db_dir),
        "output_file": str(output_path),
    }


# ================================================================
# 2. MSCHEMA CLASS (verbatim from your codebase, self-contained)
# ================================================================

class MSchema:
    def __init__(self, db_id: str = "Anonymous", schema: Optional[str] = None):
        self.db_id = db_id
        self.schema = schema
        self.tables: Dict[str, Any] = {}
        self.foreign_keys: List = []

    def add_table(self, name: str, fields: Optional[dict] = None, comment: str = None, **kwargs):
        field_map = {} if fields is None else fields.copy()
        self.tables[name] = {"fields": field_map, "examples": [], "comment": comment, **kwargs}

    def add_field(
            self, table_name: str, field_name: str, field_type: str = "",
            primary_key: bool = False, nullable: bool = True, default: Any = None,
            autoincrement: bool = False, comment: str = "", examples: Optional[list] = None, **kwargs):
        self.tables[table_name]["fields"][field_name] = {
            "type": field_type,
            "primary_key": primary_key,
            "nullable": nullable,
            "default": default if default is None else str(default),
            "autoincrement": autoincrement,
            "comment": comment,
            "examples": [] if examples is None else examples.copy(),
            **kwargs,
        }

    def add_foreign_key(self, table_name, field_name, ref_schema, ref_table_name, ref_field_name):
        self.foreign_keys.append([table_name, field_name, ref_schema, ref_table_name, ref_field_name])

    def get_field_type(self, field_type: str, simple_mode: bool = True) -> str:
        return field_type.split("(")[0] if simple_mode else field_type

    def has_table(self, table_name: str) -> bool:
        return table_name in self.tables

    def has_column(self, table_name: str, field_name: str) -> bool:
        return self.has_table(table_name) and field_name in self.tables[table_name]["fields"]

    def get_field_info(self, table_name: str, field_name: str) -> Dict:
        try:
            return self.tables[table_name]["fields"][field_name]
        except Exception:
            return {}

    def single_table_mschema(self, table_name: str,
                             selected_columns: List = None,
                             example_num: int = 3,
                             show_type_detail: bool = False,
                             include_nullability: bool = True) -> str:
        table_info = self.tables.get(table_name, {})
        output = []

        table_comment = table_info.get("comment", "")
        prefix = f"{self.schema}.{table_name}" if (self.schema and len(self.schema) > 0) else table_name
        table_line = f"# Table: {prefix}"
        if table_comment and table_comment != "None" and len(table_comment) > 0:
            table_line += f", {table_comment}"

        # Surface table-level context up front so the model can reason about scale
        # before it sees per-column details.
        summary_parts = []
        if table_info.get("row_count") is not None:
            summary_parts.append(f"Rows: {table_info['row_count']}")
        if table_info.get("column_count") is not None:
            summary_parts.append(f"Columns: {table_info['column_count']}")
        if summary_parts:
            table_line += " | " + ", ".join(summary_parts)
        output.append(table_line)

        field_lines = []
        for field_name, field_info in table_info["fields"].items():
            if selected_columns is not None and field_name.lower() not in selected_columns:
                continue

            raw_type = self.get_field_type(field_info["type"], not show_type_detail)
            field_line = f"({field_name}:{raw_type.upper()})"
            details = []

            # -- inline column comment ----------------------------------------
            comment = field_info.get("comment", "")
            if comment and comment.strip():
                details.append(comment.strip())

            # -- primary key label -------------------------------------------
            if field_info.get("primary_key", False):
                details.append("Primary Key")

            if include_nullability:
                details.append("Nullable" if field_info.get("nullable", True) else "Not Null")

            default = field_info.get("default")
            if default is not None:
                details.append(f"Default: {format_stat_value(default)}")

            # Keep derived stats compact and consistently labeled so they read as
            # schema metadata rather than free-form prose.
            stats_parts = []
            for label, key in (("Nulls", "null_count"), ("Non-null", "non_null_count"), ("Distinct", "distinct_count")):
                value = field_info.get(key)
                if value is not None:
                    stats_parts.append(f"{label}: {value}")

            if field_info.get("min_value") is not None:
                stats_parts.append(f"Min: {format_stat_value(field_info['min_value'])}")
            if field_info.get("avg_value") is not None:
                stats_parts.append(f"Avg: {format_stat_value(field_info['avg_value'])}")
            if field_info.get("max_value") is not None:
                stats_parts.append(f"Max: {format_stat_value(field_info['max_value'])}")
            if stats_parts:
                details.append("Stats: " + "; ".join(stats_parts))

            # Frequent categorical values are often more useful than raw examples
            # when the model needs to infer the role of a text column.
            top_values = truncate_values(field_info.get("top_values", []), max_items=example_num)
            if top_values:
                details.append(f"Top values: [{', '.join(top_values)}]")

            # -- examples -----------------------------------------------------
            if example_num > 0 and len(field_info.get("examples", [])) > 0:
                examples = [s for s in field_info["examples"] if s is not None]
                examples = truncate_values(examples_to_str(examples), max_items=example_num, max_length=50)

                if examples:
                    details.append(f"Examples: [{', '.join(examples)}]")

            if details:
                field_line += ", " + ", ".join(details)
                field_line += ")"
            field_lines.append(field_line)

        output.append("[")
        output.append(",\n".join(field_lines))
        output.append("]")
        return "\n".join(output)

    def to_mschema(self, selected_tables: List = None, selected_columns: List = None,
                   example_num: int = 3, show_type_detail: bool = False,
                   include_nullability: bool = True) -> str:
        output = [f"`{self.db_id}`", "【Schema】"]

        if selected_tables is not None:
            selected_tables = [s.lower() for s in selected_tables]
        if selected_columns is not None:
            selected_columns = [s.lower() for s in selected_columns]
            selected_tables = [s.split(".")[0].lower() for s in selected_columns]

        for table_name, table_info in self.tables.items():
            if selected_tables is None or table_name.lower() in selected_tables:
                column_names = list(table_info["fields"].keys())
                cur_selected = (
                    [c.lower() for c in column_names
                     if f"{table_name}.{c}".lower() in selected_columns]
                    if selected_columns is not None else None
                )
                output.append(self.single_table_mschema(table_name, cur_selected,
                                                       example_num, show_type_detail,
                                                       include_nullability=include_nullability))

        if self.foreign_keys:
            output.append("【Foreign keys】")
            for fk in self.foreign_keys:
                table1, col1, ref_schema, table2, col2 = fk
                if selected_tables is None or (
                    table1.lower() in selected_tables and table2.lower() in selected_tables):
                    if ref_schema == self.schema:
                        output.append(f"{table1}.{col1}={table2}.{col2}")

        return "\n".join(output)


# ================================================================
# 3. COLUMN-MEANING LOADER
# ================================================================

def load_column_meanings(path: str) -> Dict[str, str]:
    """
    Load column_meaning.json.
    Key format: "<db_id>|<table>|<column>"
    Value:       raw string like "# CustomerID is ..."
    Returns a dict keyed by (db_id, table, column) -> cleaned comment string.
    """
    raw = read_json(path)
    cleaned: Dict[str, str] = {}
    for key, val in raw.items():
        # Strip leading #, quotes, whitespace
        comment = val.strip()
        comment = re.sub(r'^#\s*', '', comment)  # remove leading #
        comment = comment.strip("\"'").strip()
        cleaned[key] = comment                    # keep original key format
    return cleaned


def get_comment(meanings: Dict[str, str], db_id: str, table: str, column: str) -> str:
    """Look up comment for a column; return empty string if not found."""
    key = f"{db_id}|{table}|{column}"
    return meanings.get(key, "")


# ================================================================
# 4. DB -> MSCHEMA BUILDER
# ================================================================

DATE_PATTERNS = [
    r"^\d{4}-\d{2}-\d{2}$",
    r"^\d{4}/\d{2}/\d{2}$",
    r"^\d{2}/\d{2}/\d{4}$",
    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$",
]


def _looks_like_date(value) -> bool:
    return any(re.match(p, str(value).strip()) for p in DATE_PATTERNS)


def _is_numeric(v) -> bool:
    try:
        float(str(v))
        return True
    except ValueError:
        return False


def classify_column(col_type: str, samples: list) -> str:
    ct = col_type.upper()
    if any(t in ct for t in ("INT", "REAL", "FLOAT", "DOUBLE", "NUMERIC", "DECIMAL", "NUMBER")):
        return "NUMERIC"
    if any(t in ct for t in ("DATE", "TIME", "DATETIME", "TIMESTAMP")):
        return "DATE"
    non_null = [v for v in samples if v is not None]
    if not non_null:
        return "TEXT"
    if all(_looks_like_date(v) for v in non_null[:10]):
        return "DATE"
    if all(_is_numeric(v) for v in non_null[:20]):
        return "NUMERIC"
    return "TEXT"


def build_mschema_from_db(db_path: str, db_id: str,
                          meanings: Dict[str, str],
                          example_num: int = 3,
                          include_stats: bool = True) -> MSchema:
    """
    Introspect a SQLite DB -> MSchema.
    Column comments are injected inline from column_meaning.json.
    """
    ms = MSchema(db_id=db_id)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # -- table list -----------------------------------------------------
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    tables = [row[0] for row in cur.fetchall()]

    # Capture foreign keys once up front so the final schema can express join paths
    # without re-querying SQLite during prompt construction.
    # -- foreign keys ---------------------------------------------------
    for table in tables:
        try:
            cur.execute(f"PRAGMA foreign_key_list('{table}');")
            for fk in cur.fetchall():
                ms.add_foreign_key(table, fk[3], None, fk[2], fk[4])
        except Exception:
            pass

    # -- columns --------------------------------------------------------
    for table in tables:
        try:
            cur.execute(f"SELECT COUNT(*) FROM `{table}`")
            row_count = cur.fetchone()[0]
        except Exception:
            row_count = None

        cur.execute(f"PRAGMA table_info('{table}');")
        cols = cur.fetchall()   # (cid, name, type, notnull, dflt, pk)
        ms.add_table(table, row_count=row_count, column_count=len(cols))

        for cid, col_name, col_type, notnull, dflt, pk in cols:
            col_type = col_type or "TEXT"

            # sample values for type classification
            try:
                cur.execute(
                    f"SELECT `{col_name}` FROM `{table}` "
                    f"WHERE `{col_name}` IS NOT NULL LIMIT 20;"
                )
                samples = [r[0] for r in cur.fetchall()]
            except Exception:
                samples = []

            kind = classify_column(col_type, samples)

            # Compute lightweight descriptive stats directly from SQLite so prompt
            # quality improves without requiring a separate profiling pipeline.
            # Summary stats stay type-specific, while examples always show the
            # most frequent concrete values seen in the column.
            # The output intentionally stays small enough to fit prompt budgets.
            # -- compute examples / stats ------------------------------------
            examples: list = []
            non_null_count = None
            null_count = None
            distinct_count = None
            min_value = None
            avg_value = None
            max_value = None
            top_values: List[Any] = []
            if include_stats:
                try:
                    if kind == "NUMERIC":
                        cur.execute(
                            f"SELECT COUNT(`{col_name}`), COUNT(DISTINCT `{col_name}`), "
                            f"MIN(`{col_name}`), AVG(`{col_name}`), MAX(`{col_name}`) "
                            f"FROM `{table}` WHERE `{col_name}` IS NOT NULL;"
                        )
                        non_null_count, distinct_count, min_value, avg_value, max_value = cur.fetchone()
                        if row_count is not None and non_null_count is not None:
                            null_count = row_count - non_null_count
                        if non_null_count:
                            examples = get_top_occurring_values(cur, table, col_name)

                    elif kind == "DATE":
                        cur.execute(
                            f"SELECT COUNT(`{col_name}`), COUNT(DISTINCT `{col_name}`), "
                            f"MIN(`{col_name}`), MAX(`{col_name}`) "
                            f"FROM `{table}` WHERE `{col_name}` IS NOT NULL;"
                        )
                        non_null_count, distinct_count, min_value, max_value = cur.fetchone()
                        if row_count is not None and non_null_count is not None:
                            null_count = row_count - non_null_count
                        if non_null_count:
                            examples = get_top_occurring_values(cur, table, col_name)

                    else:  # TEXT
                        cur.execute(
                            f"SELECT COUNT(`{col_name}`), COUNT(DISTINCT `{col_name}`) "
                            f"FROM `{table}` WHERE `{col_name}` IS NOT NULL;"
                        )
                        non_null_count, distinct_count = cur.fetchone()
                        if row_count is not None and non_null_count is not None:
                            null_count = row_count - non_null_count
                        top_values = get_top_occurring_values(cur, table, col_name)
                        examples = top_values[:]
                except Exception:
                    examples = samples[:3]
                    if row_count is not None:
                        non_null_count = len([sample for sample in samples if sample is not None])
                        null_count = row_count - non_null_count
                        distinct_count = len({str(sample) for sample in samples if sample is not None})

            # -- inline comment from column_meaning.json ----------------------
            comment = get_comment(meanings, db_id, table, col_name) if meanings else ""

            # SQLite reports primary keys separately from NOT NULL. Treat PKs as
            # non-null in the rendered schema to avoid contradictory metadata.
            ms.add_field(
                table_name     = table,
                field_name     = col_name,
                field_type     = kind,
                primary_key    = bool(pk),
                nullable       = not (bool(notnull) or bool(pk)),
                default        = dflt,
                autoincrement  = False,
                comment        = comment,
                examples       = [str(e) for e in examples if e is not None],
                null_count     = null_count,
                non_null_count = non_null_count,
                distinct_count = distinct_count,
                min_value      = min_value,
                avg_value      = round(avg_value, 4) if avg_value is not None else None,
                max_value      = max_value,
                top_values     = top_values,
            )

    conn.close()
    return ms


# ================================================================
# 5. PROMPT TEMPLATES
# ================================================================

SYSTEM_PROMPT = """You are a Text-to-SQL agent specialized in SQLite.

MISSION

You will be given:
- <database_schema>: schema of the target SQLite database
- <question>: natural language query to answer
- <hint>: optional evidence or guidance
- <db_id>: database identifier

Your task is to produce one valid, executable, read-only SQLite query that exactly answers the question.
Do not ask follow-up questions. Use only the provided database schema, question, hint, few-shot examples, and optional SQL generation memory.

SQL RULES
- SQLite only.
- Return a single SELECT query only; do not use DDL, DML, PRAGMA, ATTACH, file IO, or network access.
- Use exact table and column names from <database_schema>.
- Quote unusual identifiers with backticks.
- Avoid SELECT *; return only the requested columns in the requested order.
- Treat <hint> mappings as authoritative.
- Prefer canonical IDs unless the question explicitly asks for names, titles, text, or descriptions.
- Use robust string matching when appropriate: TRIM(...), COLLATE NOCASE, or LOWER(...).
- For dates, use SQLite-compatible expressions such as strftime('%Y', col), SUBSTR(col, 1, 4), or date(col).
- For percentages and ratios, cast the numerator as REAL.
- For highest/lowest/top/earliest/latest questions, use the correct ORDER BY direction and LIMIT unless all ties are requested.
- Ensure every non-aggregated selected column is included in GROUP BY.

OUTPUT FORMAT
Respond in exactly one turn using this XML shape and no extra text outside the tags:

<scratch_pad>
Briefly identify the relevant tables, columns, joins, filters, aggregation, ordering, and output shape. Keep this concise.
</scratch_pad>
<final_answer>
<sql_code>SELECT ...</sql_code>
</final_answer>"""


USER_TEMPLATE = """
Use the examples only for SQL patterns; they may come from different databases.
{FEWSHOTS}

<question>
{QUESTION}
</question>

<hint>
{HINT}
</hint>

<database_schema>
{DBSCHEMA}
</database_schema>

<db_id>{DBID}</db_id>

---
Question: {QUESTION}
Evidence: {HINT}
"""


USER_TEMPLATE_NO_FEWSHOTS = """
<question>
{QUESTION}
</question>

<hint>
{HINT}
</hint>

<database_schema>
{DBSCHEMA}
</database_schema>

<db_id>{DBID}</db_id>

---
Question: {QUESTION}
Evidence: {HINT}
"""


# ================================================================
# 6. FORMATTERS
# ================================================================

def format_fewshots(few_shot_examples: list, enabled: bool = True) -> str:
    if not enabled or not few_shot_examples:
        return "(no examples)"
    lines = []
    for i, ex in enumerate(few_shot_examples, 1):
        lines.append(f"- Example {i} (db: {ex.get('db_id', '?')})")
        lines.append(f"  Q: {ex.get('question', '')}")
        if ex.get("evidence"):
            lines.append(f"  Hint: {ex['evidence']}")
        lines.append(ex.get("SQL", ""))
    return "\n".join(lines).strip()


def format_assistant(sql: str) -> str:
    return (
        "<scratch_pad>\nGold reference SQL.\n</scratch_pad>\n"
        f"<final_answer>\n<sql_code>{sql}</sql_code>\n</final_answer>"
    )


def format_user_prompt(
    question: str,
    hint: str,
    db_schema: str,
    db_id: str,
    few_shot_examples: list,
    include_fewshots: bool = True,
) -> str:
    template = USER_TEMPLATE if include_fewshots else USER_TEMPLATE_NO_FEWSHOTS
    values = {
        "QUESTION": question,
        "HINT": hint,
        "DBSCHEMA": db_schema,
        "DBID": db_id,
    }
    if include_fewshots:
        values["FEWSHOTS"] = format_fewshots(few_shot_examples, enabled=True)
    return template.format(**values)


def build_output_entry(
    db_id: str,
    gold_sql: str,
    hint: str,
    question: str,
    user_msg: str,
    messages_only: bool = False,
) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": format_assistant(gold_sql)},
    ]
    if messages_only:
        return {"messages": messages}
    return {
        "db_id": db_id,
        "gold_sql": gold_sql,
        "evidence": hint,
        "question": question,
        "messages": messages,
    }


# ================================================================
# 7. MAIN
# ================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build JSONL fine-tuning data from BIRD train/dev files with enriched schema statistics."
    )
    parser.add_argument(
        "--base-dir",
        default=None,
        help="Repository root or legacy bird_sql directory (defaults to auto-detected repo root).",
    )
    parser.add_argument(
        "--split", choices=["train", "dev"], default="train",
        help="Dataset split to use when resolving default input, database, and meanings paths.",
    )
    parser.add_argument(
        "--input-file", default=None,
        help="Input dataset path (.json or .jsonl). Defaults to the split-specific few-shot file.",
    )
    parser.add_argument(
        "--database-dir", default=None,
        help="Directory containing split databases (for example databases/train_databases).",
    )
    parser.add_argument(
        "--meanings-file", default=None,
        help="Column meanings JSON path. Defaults to the split-specific column meanings file.",
    )
    parser.add_argument(
        "--n-examples", type=int, default=5,
        help="Number of examples to process (-1 = all)",
    )
    parser.add_argument(
        "--example-num", type=int, default=3,
        help="Max column examples shown per field in MSchema",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output .jsonl path (default: <base-dir>/output.jsonl)",
    )
    parser.add_argument(
        "--no-comments", action="store_true",
        help="Disable inline column comments from column_meaning.json",
    )
    parser.add_argument(
        "--no-fewshots", action="store_true",
        help="Disable few-shot examples in user prompt",
    )
    parser.add_argument(
        "--no-stats", action="store_true",
        help="Disable column stats / examples in MSchema",
    )
    parser.add_argument(
        "--no-nullability", action="store_true",
        help="Disable Nullable / Not Null labels in rendered schema output.",
    )
    parser.add_argument(
        "--log-every", type=int, default=50,
        help="Print progress every N records (default: 50). Use 0 to disable interval logging.",
    )
    parser.add_argument(
        "--messages-only", action="store_true",
        help="Emit legacy message-only JSONL rows without top-level db_id/gold_sql/evidence/question fields.",
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # -- resolve paths ---------------------------------------------------
    default_paths = resolve_default_paths(args.base_dir, args.split)
    INPUT_FILE    = args.input_file or default_paths["input_file"]
    DB_DIR        = args.database_dir or default_paths["database_dir"]
    MEANINGS_FILE = args.meanings_file or default_paths["meanings_file"]
    OUTPUT_JSONL  = args.output or default_paths["output_file"]

    # -- load data -------------------------------------------------------
    print(f"Loading dataset from {INPUT_FILE} ...")
    data    = read_json(INPUT_FILE)
    records = data if args.n_examples == -1 else data[:args.n_examples]
    print(f"  → {len(records)} examples selected")

    # -- load column meanings (optional) --------------------------------
    meanings: Dict[str, str] = {}
    if not args.no_comments and os.path.exists(MEANINGS_FILE):
        print(f"Loading column meanings from {MEANINGS_FILE} ...")
        meanings = load_column_meanings(MEANINGS_FILE)
        print(f"  → {len(meanings)} column definitions loaded")
    elif args.no_comments:
        print("Column comments disabled (--no-comments).")
    else:
        print(f"[WARN] column_meaning.json not found at {MEANINGS_FILE}; skipping comments.")

    # -- schema cache ----------------------------------------------------
    schema_cache: Dict[str, str] = {}

    def get_mschema_str(db_id: str) -> str:
        if db_id in schema_cache:
            return schema_cache[db_id]
        db_path = os.path.join(DB_DIR, db_id, f"{db_id}.sqlite")
        if not os.path.exists(db_path):
            print(f"  [WARN] DB not found: {db_path}")
            schema_cache[db_id] = "Schema not available."
        else:
            # Schema generation is expensive relative to record formatting, so cache
            # per-db renders and reuse them across all examples hitting the same DB.
            print(f"  [CACHE MISS] Building MSchema for {db_id} ...")
            ms = build_mschema_from_db(
                db_path      = db_path,
                db_id        = db_id,
                meanings     = meanings,
                example_num  = args.example_num,
                include_stats= not args.no_stats,
            )
            schema_cache[db_id] = ms.to_mschema(
                example_num=args.example_num,
                include_nullability=not args.no_nullability,
            )
        return schema_cache[db_id]

    # -- write JSONL -----------------------------------------------------
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as out_f:
        for idx, record in enumerate(records):
            db_id    = record["db_id"]
            question = record["question"]
            hint     = record.get("evidence", "")
            gold_sql = record.get("SQL", "")
            fewshots = record.get("few_shot_examples", [])

            current_index = idx + 1
            # Emit the first, last, and periodic progress checkpoints so long runs
            # stay observable without flooding the terminal on large datasets.
            should_log = (
                current_index == 1
                or current_index == len(records)
                or (args.log_every > 0 and current_index % args.log_every == 0)
            )

            if should_log:
                print(f"\n[{current_index}/{len(records)}] question_id={record.get('question_id')}  db={db_id}")

            db_schema = get_mschema_str(db_id)

            user_msg = format_user_prompt(
                question=question,
                hint=hint,
                db_schema=db_schema,
                db_id=db_id,
                few_shot_examples=fewshots,
                include_fewshots=not args.no_fewshots,
            )

            entry = build_output_entry(
                db_id=db_id,
                gold_sql=gold_sql,
                hint=hint,
                question=question,
                user_msg=user_msg,
                messages_only=args.messages_only,
            )

            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            if should_log:
                print("  ✓ Written")

    print(f"\n{'='*60}")
    print(f"Done. JSONL saved to : {OUTPUT_JSONL}")
    print(f"Unique DBs introspected: {len(schema_cache)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()