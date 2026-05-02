import re
from typing import Dict, List, Set, Tuple


SQL_KEYWORDS = {
    "select", "from", "where", "join", "inner", "left", "right", "outer",
    "cross", "on", "and", "or", "not", "in", "between", "like", "is",
    "null", "group", "order", "by", "having", "limit", "union", "except",
    "intersect", "exists", "case", "when", "then", "else", "end", "as",
    "distinct", "all", "asc", "desc", "with", "recursive", "natural",
    "count", "sum", "avg", "max", "min", "coalesce", "ifnull", "round",
    "date", "time", "datetime", "strftime", "julianday", "true", "false",
}


def extract_schema_from_prompt(prompt_text: str) -> Tuple[Set[str], Set[str]]:
    table_names: Set[str] = set()
    column_names: Set[str] = set()

    if not prompt_text:
        return table_names, column_names

    for match in re.finditer(
        r"CREATE\s+TABLE\s+(?:[`\"\[]?)(\w+)(?:[`\"\]]?)",
        prompt_text,
        re.IGNORECASE,
    ):
        table_names.add(match.group(1).lower())

    for match in re.finditer(
        r"^Table:\s*(\w+)",
        prompt_text,
        re.IGNORECASE | re.MULTILINE,
    ):
        table_names.add(match.group(1).lower())

    for match in re.finditer(
        r"(?:^|\n)\s*[`\"]?([a-zA-Z_]\w*)[`\"]?\s+"
        r"(?:TEXT|INTEGER|REAL|NUMERIC|BLOB|VARCHAR|CHAR|INT|FLOAT|DOUBLE|BOOLEAN|DATE|DATETIME|TIMESTAMP)",
        prompt_text,
        re.IGNORECASE,
    ):
        col = match.group(1).lower()
        if col not in SQL_KEYWORDS:
            column_names.add(col)

    for match in re.finditer(r'"name"\s*:\s*"([a-zA-Z_]\w*)"', prompt_text):
        name = match.group(1).lower()
        if name not in SQL_KEYWORDS:
            column_names.add(name)

    for match in re.finditer(r"\b([a-zA-Z_]\w*)\.([a-zA-Z_]\w*)\b", prompt_text):
        table_names.add(match.group(1).lower())
        column_names.add(match.group(2).lower())

    return table_names, column_names


def extract_schema_items_from_sql(sql: str) -> Set[str]:
    if not sql:
        return set()

    sql_lower = sql.lower()
    sql_no_strings = re.sub(r"'[^']*'", "''", sql_lower)
    sql_no_strings = re.sub(r'"[^"]*"', '""', sql_no_strings)

    items: Set[str] = set()
    alias_map: Dict[str, str] = {}

    for match in re.finditer(
        r"(?:from|join)\s+([a-z_][a-z0-9_]*)(?:\s+(?:as\s+)?([a-z_][a-z0-9_]*))?",
        sql_no_strings,
    ):
        table = match.group(1)
        alias = match.group(2)

        if table not in SQL_KEYWORDS:
            items.add(table)

            if alias and alias not in SQL_KEYWORDS:
                alias_map[alias] = table

    for match in re.finditer(
        r"\b([a-z_][a-z0-9_]*)\.([a-z_][a-z0-9_]*)\b",
        sql_no_strings,
    ):
        qualifier = match.group(1)
        col = match.group(2)

        actual_table = alias_map.get(qualifier, qualifier)

        items.add(actual_table)
        items.add(col)
        items.add(f"{actual_table}.{col}")

    select_match = re.search(
        r"select\s+(.*?)\s+from\s",
        sql_no_strings,
        re.DOTALL,
    )

    if select_match:
        for col in re.findall(
            r"(?<!\.)(?:[a-z_][a-z0-9_]*)(?!\s*\()",
            select_match.group(1),
        ):
            if col not in SQL_KEYWORDS and not col.isdigit():
                items.add(col)

    for clause_kw in ["where", r"group\s+by", r"order\s+by", "having"]:
        clause_match = re.search(
            rf"{clause_kw}\s+(.*?)(?:group\s+by|order\s+by|having|limit|union|except|intersect|$)",
            sql_no_strings,
            re.DOTALL,
        )

        if clause_match:
            for col in re.findall(
                r"(?<!\.)(?:[a-z_][a-z0-9_]*)(?!\s*\()",
                clause_match.group(1),
            ):
                if col not in SQL_KEYWORDS and not col.isdigit():
                    items.add(col)

    return items


def tokenize_sql(sql: str) -> List[str]:
    sql = sql.lower().strip()
    sql = re.sub(r"\s+", " ", sql)

    return re.findall(
        r">=|<=|<>|!=|[a-z_][a-z0-9_]*|[-+]?\d*\.\d+|\d+|[(),.*=<>/+%-]",
        sql,
    )


def get_ngrams(tokens: List[str], n: int) -> Set[Tuple[str, ...]]:
    if len(tokens) < n:
        return set()

    return {
        tuple(tokens[i:i + n])
        for i in range(len(tokens) - n + 1)
    }


def jaccard(set_a: Set, set_b: Set) -> float:
    if not set_a and not set_b:
        return 1.0

    if not set_a or not set_b:
        return 0.0

    return len(set_a & set_b) / max(1, len(set_a | set_b))


def extract_tables_from_sql(sql: str) -> Set[str]:
    """Extract bare table names referenced in FROM/JOIN clauses."""
    if not sql:
        return set()

    sql_lower = sql.lower()
    sql_no_strings = re.sub(r"'[^']*'", "''", sql_lower)
    sql_no_strings = re.sub(r'"[^"]*"', '""', sql_no_strings)
    sql_no_strings = sql_no_strings.replace("`", "")

    tables: Set[str] = set()
    for match in re.finditer(
        r"(?:from|join)\s+([a-z_][a-z0-9_-]*)",
        sql_no_strings,
    ):
        table = match.group(1)
        if table not in SQL_KEYWORDS:
            tables.add(table)

    return tables


def extract_columns_from_sql(sql: str) -> Set[str]:
    """Extract column names referenced anywhere in the SQL (alias-resolved)."""
    if not sql:
        return set()

    sql_lower = sql.lower()
    sql_no_strings = re.sub(r"'[^']*'", "''", sql_lower)
    sql_no_strings = re.sub(r'"[^"]*"', '""', sql_no_strings)
    sql_no_strings = sql_no_strings.replace("`", "")

    # Build alias -> table map so we know which qualifiers are tables vs aliases.
    alias_to_table: Dict[str, str] = {}
    for match in re.finditer(
        r"(?:from|join)\s+([a-z_][a-z0-9_-]*)(?:\s+(?:as\s+)?([a-z_][a-z0-9_]*))?",
        sql_no_strings,
    ):
        table = match.group(1)
        alias = match.group(2)
        if alias and alias not in SQL_KEYWORDS and alias != table:
            alias_to_table[alias] = table

    columns: Set[str] = set()

    # Qualified columns: pick the column part only.
    for match in re.finditer(
        r"\b([a-z_][a-z0-9_]*)\.([a-z_][a-z0-9_]*)\b",
        sql_no_strings,
    ):
        col = match.group(2)
        if col not in SQL_KEYWORDS and col != "*":
            columns.add(col)

    # Unqualified columns from SELECT / WHERE / GROUP BY / ORDER BY / HAVING.
    table_tokens = set(alias_to_table.values()) | set(
        re.findall(r"(?:from|join)\s+([a-z_][a-z0-9_-]*)", sql_no_strings)
    )
    alias_tokens = set(alias_to_table.keys())

    clause_pattern = re.compile(
        r"(?:select|where|group\s+by|order\s+by|having|on)\s+(.*?)"
        r"(?=\b(?:from|where|group\s+by|order\s+by|having|limit|union|except|intersect|on|join)\b|$)",
        re.DOTALL,
    )

    for match in clause_pattern.finditer(sql_no_strings):
        chunk = match.group(1)
        # Strip qualified identifiers so we don't double count the table side.
        chunk = re.sub(r"\b[a-z_][a-z0-9_]*\.", "", chunk)
        for tok in re.findall(r"(?<![\w.])([a-z_][a-z0-9_]*)(?!\s*\()", chunk):
            if tok in SQL_KEYWORDS:
                continue
            if tok in table_tokens or tok in alias_tokens:
                continue
            if tok.isdigit():
                continue
            columns.add(tok)

    return columns