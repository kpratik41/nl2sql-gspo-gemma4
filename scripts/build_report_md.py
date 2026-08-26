"""Assemble report.md from prose sections plus generated result tables.

The prose lives in ``report_src/`` with ``{{T<n>}}`` placeholders; the tables come
from ``scripts/make_tables.py``, which reads ``results/*.json``. Keeping the two
apart means the report's numbers are generated, never transcribed, so it cannot
drift from the data. Run this, then ``scripts/build_report.py`` for the HTML.
"""
import re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
S = ROOT / "report_src"

raw = subprocess.run([str(ROOT/".venv/bin/python"), str(ROOT/"scripts/make_tables.py")],
                     capture_output=True, text=True, cwd=ROOT).stdout

# Split into blocks keyed by "T<n>", dropping the "### Tn — ..." heading line so the
# report's own section headings are the only ones.
blocks = {}
cur, buf = None, []
for line in raw.splitlines():
    m = re.match(r"^### (T\d+) — (.*)$", line)
    if m:
        if cur:
            blocks[cur] = "\n".join(buf).strip()
        cur, buf = m.group(1), []
    elif cur:
        buf.append(line)
if cur:
    blocks[cur] = "\n".join(buf).strip()

missing = [k for k in [f"T{i}" for i in range(1, 14)] if k not in blocks or not blocks[k]]
if missing:
    print(f"WARNING: missing/empty tables: {missing}", file=sys.stderr)

head = (S/"report_head.md").read_text()
tail = (S/"report_tail.md").read_text()
body = (S/"report_body.md").read_text()

for key, block in blocks.items():
    body = body.replace(f"{{{{{key}}}}}", block)

left = re.findall(r"\{\{T\d+\}\}", body)
if left:
    print(f"WARNING: unreplaced placeholders: {left}", file=sys.stderr)

(ROOT/"report.md").write_text(head.rstrip() + "\n\n" + body.strip() + "\n\n" + tail.lstrip())
print(f"wrote report.md with tables: {sorted(blocks, key=lambda x: int(x[1:]))}")
