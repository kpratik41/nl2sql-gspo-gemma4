import re
from typing import Any, Dict, List


DB_ID_TAG_RE = re.compile(r"<db_id>\s*([^<\n]+?)\s*</db_id>", re.IGNORECASE | re.DOTALL)
SCHEMA_DB_ID_RE = re.compile(r"<database_schema>\s*`([^`\n]+)`", re.IGNORECASE | re.DOTALL)
HINT_TAG_RE = re.compile(r"<hint>\s*(.*?)\s*</hint>", re.IGNORECASE | re.DOTALL)


def _extract_from_messages(messages: List[Dict[str, Any]], *patterns: re.Pattern[str]) -> str:
    for message in messages:
        if not isinstance(message, dict):
            continue

        content = message.get("content", "")
        if not isinstance(content, str):
            continue

        for pattern in patterns:
            match = pattern.search(content)
            if match:
                return match.group(1).strip()

    return ""


def normalize_record(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts harmony/chat JSONL into the prompt format expected by TRL.

    Input example:
    {
      "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "SELECT ..."}
      ],
      "db_id": "...",
      "gold_sql": "...",
      "evidence": "..."
    }

    Output:
    {
      "prompt": [system/user messages only],
      "messages": [system/user messages only],
      "db_id": "...",
      "gold_sql": "...",
      "evidence": "..."
    }
    """

    messages = example.get("messages", [])
    prompt_messages: List[Dict[str, str]] = []

    gold_sql = (
        example.get("gold_sql")
        or example.get("query")
        or example.get("SQL")
        or example.get("sql")
        or ""
    )

    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue

            role = message.get("role")
            content = message.get("content", "")

            if role == "assistant":
                if not gold_sql:
                    gold_sql = content
                continue

            if role in {"system", "user"}:
                prompt_messages.append(
                    {
                        "role": role,
                        "content": content,
                    }
                )

    if not prompt_messages:
        question = example.get("question", "")
        schema = example.get("schema", "")
        evidence = (
            example.get("evidence")
            or example.get("external_knowledge")
            or example.get("hint")
            or ""
        )

        prompt_messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert Text-to-SQL assistant. "
                    "Return only valid SQLite SQL."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Schema:\n{schema}\n\n"
                    f"Evidence:\n{evidence}\n\n"
                    "Return only the SQL query."
                ),
            },
        ]

    inferred_db_id = _extract_from_messages(messages, DB_ID_TAG_RE, SCHEMA_DB_ID_RE)
    inferred_evidence = _extract_from_messages(messages, HINT_TAG_RE)

    return {
        "prompt": prompt_messages,
        "messages": prompt_messages,
        "db_id": (
            example.get("db_id")
            or example.get("database")
            or example.get("database_name")
            or inferred_db_id
            or ""
        ),
        "gold_sql": gold_sql,
        "evidence": (
            example.get("evidence")
            or example.get("external_knowledge")
            or example.get("hint")
            or inferred_evidence
            or ""
        ),
    }