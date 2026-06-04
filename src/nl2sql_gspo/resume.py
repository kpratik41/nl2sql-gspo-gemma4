"""Small helpers for resumable BIRD inference runs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple


class ResumeManifestError(ValueError):
    """Raised when an existing run cannot be safely resumed."""


def add_resume_args(parser) -> None:
    parser.add_argument("--resume", action="store_true", help="Resume a compatible incremental run.")
    parser.add_argument(
        "--incremental_writes",
        action="store_true",
        help="Append generation records as they finish instead of writing only at the end.",
    )


def validate_resume_args(args) -> None:
    if getattr(args, "resume", False) and getattr(args, "overwrite", False):
        raise ValueError("--resume and --overwrite are mutually exclusive")


def incremental_enabled(args) -> bool:
    return bool(getattr(args, "resume", False) or getattr(args, "incremental_writes", False))


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def build_manifest(args, *, mode: str, fields: Sequence[str]) -> Dict[str, Any]:
    config = {"mode": mode}
    for field in fields:
        config[field] = _json_ready(getattr(args, field, None))
    return {"version": 1, "config": config}


def prepare_manifest(output_dir: Path, manifest: Dict[str, Any], *, resume: bool) -> None:
    path = output_dir / "run_manifest.json"
    if resume:
        if not path.exists():
            raise ResumeManifestError(f"Cannot resume because manifest is missing: {path}")
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ResumeManifestError(_manifest_mismatch_message(existing, manifest))
        return
    atomic_write_json(path, manifest)


def _manifest_mismatch_message(existing: Dict[str, Any], current: Dict[str, Any]) -> str:
    existing_config = existing.get("config", {})
    current_config = current.get("config", {})
    diffs = []
    for key in sorted(set(existing_config) | set(current_config)):
        if existing_config.get(key) != current_config.get(key):
            diffs.append(f"{key}: existing={existing_config.get(key)!r} current={current_config.get(key)!r}")
    preview = "; ".join(diffs[:12])
    if len(diffs) > 12:
        preview += f"; ... {len(diffs) - 12} more"
    return f"Existing run_manifest.json is not compatible with this resume request: {preview}"


def safe_read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    nonempty = [(line_number, line) for line_number, line in enumerate(lines, start=1) if line.strip()]
    rows: List[Dict[str, Any]] = []
    truncated_final_line = False
    for offset, (line_number, line) in enumerate(nonempty):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            if offset == len(nonempty) - 1:
                print(f"[resume] ignoring malformed final checkpoint line {line_number} in {path}: {exc}")
                truncated_final_line = True
                continue
            raise ValueError(f"Malformed JSONL checkpoint line {line_number} in {path}: {exc}") from exc
    if truncated_final_line:
        atomic_write_jsonl(path, rows)
    return rows


def checkpoint_map(
    path: Path,
    key_fn: Callable[[Dict[str, Any]], Tuple[Any, ...]],
) -> Dict[Tuple[Any, ...], Dict[str, Any]]:
    by_key: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for row in safe_read_jsonl(path):
        by_key[key_fn(row)] = row
    return by_key


def append_jsonl(path: Path, record: Dict[str, Any], *, fsync: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        if fsync:
            os.fsync(handle.fileno())


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def atomic_write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)
