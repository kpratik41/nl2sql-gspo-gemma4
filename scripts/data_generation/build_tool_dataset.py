#!/usr/bin/env python3
"""Build NL2SQL tool-calling inference rows from a schema-built JSONL file."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nl2sql_gspo.sql_utils import extract_sql  # noqa: E402
from nl2sql_gspo.tool_calling import get_tool_definitions, tool_catalog_compact  # noqa: E402
from prompts import SYSTEM_PROMPT_TEMPLATES  # noqa: E402


DB_ID_TAG_RE = re.compile(r"<db_id>\s*([^<\n]+?)\s*</db_id>", re.IGNORECASE | re.DOTALL)
HINT_TAG_RE = re.compile(r"<hint>\s*(.*?)\s*</hint>", re.IGNORECASE | re.DOTALL)
QUESTION_TAG_RE = re.compile(r"<question>\s*(.*?)\s*</question>", re.IGNORECASE | re.DOTALL)


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def find_message(messages: List[Dict[str, Any]], role: str) -> Dict[str, Any]:
    for message in messages:
        if isinstance(message, dict) and message.get("role") == role:
            return message
    return {}


def extract_tag(pattern: re.Pattern[str], text: str) -> str:
    match = pattern.search(text or "")
    return match.group(1).strip() if match else ""


def build_tool_system_prompt() -> str:
    return SYSTEM_PROMPT_TEMPLATES.replace(
        "{TOOL_CATALOG_COMPACT}",
        tool_catalog_compact(),
    ).strip()


def convert_record(record: Dict[str, Any], system_prompt: str, tools: List[Dict[str, Any]]) -> Dict[str, Any]:
    messages = record.get("messages") or []
    if not isinstance(messages, list):
        raise ValueError("record has no messages list")

    user_message = find_message(messages, "user")
    assistant_message = find_message(messages, "assistant")
    user_content = str(user_message.get("content", ""))

    db_id = (
        record.get("db_id")
        or extract_tag(DB_ID_TAG_RE, user_content)
        or ""
    )
    evidence = (
        record.get("evidence")
        or extract_tag(HINT_TAG_RE, user_content)
        or ""
    )
    question = (
        record.get("question")
        or extract_tag(QUESTION_TAG_RE, user_content)
        or ""
    )
    gold_sql = extract_sql(
        record.get("gold_sql")
        or assistant_message.get("content", "")
        or record.get("SQL", "")
        or record.get("sql", "")
    )

    prompt_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    return {
        "db_id": db_id,
        "gold_sql": gold_sql,
        "evidence": evidence,
        "question": question,
        "tools": tools,
        "prompt": prompt_messages,
        "messages": prompt_messages,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert bare NL2SQL schema rows into native tool-calling inference prompts."
    )
    parser.add_argument("--input", required=True, help="Input bare schema JSONL.")
    parser.add_argument("--output", required=True, help="Output tool-calling JSONL.")
    parser.add_argument("--limit", type=int, default=-1, help="Rows to process (-1 = all).")
    parser.add_argument("--log-every", type=int, default=500, help="Progress interval.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    system_prompt = build_tool_system_prompt()
    tools = get_tool_definitions()

    written = 0
    missing: List[str] = []
    with output_path.open("w", encoding="utf-8") as out_f:
        for idx, record in enumerate(iter_jsonl(input_path), start=1):
            if args.limit >= 0 and written >= args.limit:
                break

            converted = convert_record(record, system_prompt, tools)
            missing_fields = [
                name
                for name in ("db_id", "question")
                if not converted.get(name)
            ]
            if missing_fields and len(missing) < 10:
                missing.append(f"line={idx} missing={','.join(missing_fields)}")

            out_f.write(json.dumps(converted, ensure_ascii=False) + "\n")
            written += 1

            if args.log_every > 0 and (written == 1 or written % args.log_every == 0):
                print(f"[{written}] db={converted.get('db_id')}")

    if missing:
        raise ValueError("Missing required fields: " + "; ".join(missing))

    print(f"Done. Wrote {written} rows to {output_path}")


if __name__ == "__main__":
    main()
