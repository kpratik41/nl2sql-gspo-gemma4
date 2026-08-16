#!/usr/bin/env python3
"""Bake the Qwen tool-call syntax into an eval JSONL, once, offline.

The tool-format eval files embed the Gemma system prompt, which teaches
`call:name{...}` and forbids XML. The eval runner patches that at runtime via
run_inference_bird_qwen_server.rewrite_system_prompt_for_qwen, but that rewrite
was written for the OpenAI-server path: it tells the model "Do not print
tool-call JSON, XML, ... in assistant text", which is the opposite of what the
in-process async runner needs, and it leaves no XML examples at all.

Writing a Qwen-native file instead makes the data agree with the chat template
up front, so no runtime patching is needed (pass --no_prompt_rewrite).

The transformation is a single exact swap: every row in these files shares one
system prompt, byte-identical to build_tool_system_prompt("default"), so it is
replaced wholesale with build_tool_system_prompt("default_qwen"). Everything
else in the row -- user message, tools, db_id, gold_sql, evidence, question --
is untouched. The swap asserts on the exact source text, so an unexpected input
fails loudly instead of silently shipping a half-converted prompt.

Usage:
    python scripts/qwen/build_qwen_eval_data.py \
        --input  outputs/old-dev-schema-tool-unpatched.jsonl \
        --output outputs/qwen-old-dev-schema-tool-unpatched.jsonl
    # add --limit 1 to validate a single row first
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

from scripts.data_generation.build_tool_dataset import build_tool_system_prompt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--prompt-template",
        default="default_qwen",
        choices=["default_qwen", "consensus_qwen"],
        help="Qwen template to write into the converted file.",
    )
    parser.add_argument(
        "--source-template",
        default="default",
        choices=["default", "consensus"],
        help="Gemma template expected in the input file.",
    )
    parser.add_argument("--limit", type=int, default=-1, help="Convert only N rows (-1 = all).")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_path, out_path = Path(args.input), Path(args.output)
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"{out_path} exists; pass --overwrite")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    source_system = build_tool_system_prompt(args.source_template).strip()
    target_system = build_tool_system_prompt(args.prompt_template).strip()
    print(f"[build] source system prompt: {len(source_system)} chars ({args.source_template})")
    print(f"[build] target system prompt: {len(target_system)} chars ({args.prompt_template})")

    written = 0
    swapped = 0
    untouched_rows = 0
    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for lineno, line in enumerate(fin, start=1):
            if not line.strip():
                continue
            if args.limit >= 0 and written >= args.limit:
                break
            row = json.loads(line)

            row_swapped = False
            for key in ("prompt", "messages"):
                messages = row.get(key)
                if not isinstance(messages, list):
                    continue
                for message in messages:
                    if not isinstance(message, dict) or message.get("role") != "system":
                        continue
                    content = message.get("content")
                    if not isinstance(content, str):
                        continue
                    if content.strip() != source_system:
                        raise ValueError(
                            f"line {lineno}: system prompt in '{key}' does not match the "
                            f"expected {args.source_template} template "
                            f"({len(content)} chars vs {len(source_system)}). Refusing to "
                            "guess at a partial conversion."
                        )
                    message["content"] = target_system
                    row_swapped = True

            if row_swapped:
                swapped += 1
            else:
                untouched_rows += 1

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    print(f"[build] wrote {written} rows to {out_path}")
    print(f"[build] rows with system prompt swapped: {swapped}")
    if untouched_rows:
        print(f"[build] WARNING: {untouched_rows} rows had no system message to swap")


if __name__ == "__main__":
    main()
