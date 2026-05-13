"""Shared NL2SQL tool declarations and GRPO tool loading helpers."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any, Callable, Dict, List


def _object_schema(properties: Dict[str, Any], required: List[str]) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def get_tool_definitions() -> List[Dict[str, Any]]:
    """Return OpenAI/Transformers-style function declarations for Gemma tools."""

    return [
        {
            "type": "function",
            "function": {
                "name": "bm25_search_sqlite",
                "description": (
                    "Search a SQLite text column for likely literal values using BM25 "
                    "and substring matching. Use this when a question mentions a "
                    "name, category, location, code, or other value whose exact "
                    "database spelling is uncertain."
                ),
                "parameters": _object_schema(
                    {
                        "db_id": {
                            "type": "string",
                            "description": "Database identifier from <db_id>.",
                        },
                        "table": {
                            "type": "string",
                            "description": "Exact table name to search.",
                        },
                        "column": {
                            "type": "string",
                            "description": "Exact column name to search.",
                        },
                        "query": {
                            "type": "string",
                            "description": "Natural-language or literal search text.",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of best matches to return.",
                        },
                        "where": {
                            "type": "string",
                            "description": "Optional SQL WHERE fragment to restrict searched rows.",
                            "nullable": True,
                        },
                    },
                    ["db_id", "table", "column", "query"],
                ),
            },
        },
        {
            "type": "function",
            "function": {
                "name": "sqlite_peek",
                "description": (
                    "Preview values, SQLite types, null counts, distinct counts, and "
                    "small profiles for selected columns in one table."
                ),
                "parameters": _object_schema(
                    {
                        "db_id": {
                            "type": "string",
                            "description": "Database identifier from <db_id>.",
                        },
                        "table": {
                            "type": "string",
                            "description": "Exact table name to inspect.",
                        },
                        "columns": {
                            "type": "array",
                            "description": "Exact column names to inspect.",
                            "items": {"type": "string"},
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of sample values per column.",
                        },
                        "where": {
                            "type": "string",
                            "description": "Optional SQL WHERE fragment to restrict inspected rows.",
                            "nullable": True,
                        },
                    },
                    ["db_id", "table", "columns"],
                ),
            },
        },
        {
            "type": "function",
            "function": {
                "name": "sqlite_query",
                "description": (
                    "Execute a read-only SQLite SELECT or WITH...SELECT query against the target "
                    "database. Use this to verify the exact SQL before finalizing."
                ),
                "parameters": _object_schema(
                    {
                        "db_id": {
                            "type": "string",
                            "description": "Database identifier from <db_id>.",
                        },
                        "sql": {
                            "type": "string",
                            "description": "The exact SELECT or WITH query to execute.",
                        },
                        "max_return_rows": {
                            "type": "integer",
                            "description": "Maximum number of result rows to return.",
                        },
                    },
                    ["db_id", "sql"],
                ),
            },
        },
    ]


def tool_catalog_compact() -> str:
    """Compact human-readable catalog injected into the system prompt."""

    return "\n".join(
        [
            "- bm25_search_sqlite(db_id, table, column, query, top_k=10, where=None): find exact stored values.",
            "- sqlite_peek(db_id, table, columns, limit=10, where=None): inspect samples, types, nulls, and ranges.",
            "- sqlite_query(db_id, sql, max_return_rows=100): execute the exact read-only SELECT or WITH...SELECT for verification.",
        ]
    )


def configure_tool_db_roots(database_dir: str | None = None, extra_roots: str | None = None) -> str:
    """Populate BIRD_DB_ROOTS for gen_tools.py if the caller has not done so."""

    roots: List[str] = []

    def add(path: str | None) -> None:
        if not path:
            return
        expanded = str(Path(path).expanduser())
        if expanded not in roots:
            roots.append(expanded)

    for raw in (extra_roots or "").split(os.pathsep):
        add(raw)

    add(database_dir)
    if database_dir:
        add(os.path.join(database_dir, "train_databases"))
        add(os.path.join(database_dir, "dev_databases"))

    add("databases")
    add("databases/train_databases")
    add("databases/dev_databases")

    if not os.environ.get("BIRD_DB_ROOTS"):
        os.environ["BIRD_DB_ROOTS"] = os.pathsep.join(roots)

    return os.environ["BIRD_DB_ROOTS"]


def get_grpo_tool_functions() -> List[Callable[..., Any]]:
    """Load callable tools used by TRL's GRPO tool-call loop."""

    module = importlib.import_module("gen_tools")
    return [
        module.bm25_search_sqlite,
        module.sqlite_peek,
        module.sqlite_query,
    ]
