"""Render report.md to a self-contained, theme-aware report.html.

The HTML is a shadow of the Markdown: same content, styled for reading and for
handing to someone who will not open a repository.  Everything is inlined, so
the file works from a file:// URL, an email attachment, or an intranet share.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent

CSS = """
:root {
  --bg: #ffffff; --fg: #1a1d21; --muted: #5c6570; --rule: #e3e7ec;
  --accent: #12507b; --code-bg: #f5f7f9; --th-bg: #f0f3f6; --tr-alt: #fafbfc;
  --good: #1a6b3c; --warn: #8a5300; --bad: #97231f; --quote-bg: #f7f9fb;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #14171a; --fg: #e6e9ec; --muted: #99a2ac; --rule: #2b3238;
    --accent: #7cb8e0; --code-bg: #1c2126; --th-bg: #1f252b; --tr-alt: #181c20;
    --good: #6ed49b; --warn: #e0b063; --bad: #f08c86; --quote-bg: #1a1f24;
  }
}
* { box-sizing: border-box; }
body {
  background: var(--bg); color: var(--fg); margin: 0;
  font: 16px/1.65 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 900px; margin: 0 auto; padding: 3rem 1.5rem 6rem; }
h1, h2, h3, h4 { line-height: 1.25; font-weight: 650; margin: 2.4em 0 0.7em; }
h1 { font-size: 2.1rem; margin-top: 0; letter-spacing: -0.02em; }
h2 { font-size: 1.45rem; padding-bottom: 0.35rem; border-bottom: 2px solid var(--rule); }
h3 { font-size: 1.13rem; color: var(--accent); }
h4 { font-size: 1rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }
p, li { color: var(--fg); }
a { color: var(--accent); }
code {
  background: var(--code-bg); padding: 0.12em 0.38em; border-radius: 4px;
  font: 0.87em/1.5 ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
}
pre {
  background: var(--code-bg); border: 1px solid var(--rule); border-radius: 8px;
  padding: 1rem 1.1rem; overflow-x: auto;
}
pre code { background: none; padding: 0; font-size: 0.85rem; }
blockquote {
  margin: 1.4em 0; padding: 0.9rem 1.2rem; background: var(--quote-bg);
  border-left: 3px solid var(--accent); border-radius: 0 6px 6px 0;
}
blockquote p:first-child { margin-top: 0; } blockquote p:last-child { margin-bottom: 0; }
.table-scroll { overflow-x: auto; margin: 1.5em 0; }
table { border-collapse: collapse; width: 100%; font-size: 0.9rem; }
th, td { padding: 0.5rem 0.8rem; text-align: left; border-bottom: 1px solid var(--rule); }
th { background: var(--th-bg); font-weight: 600; white-space: nowrap; }
tbody tr:nth-child(even) { background: var(--tr-alt); }
td:not(:first-child), th:not(:first-child) { font-variant-numeric: tabular-nums; }
hr { border: none; border-top: 1px solid var(--rule); margin: 3rem 0; }
.subtitle { color: var(--muted); font-size: 1.05rem; margin-top: -0.4rem; }
.meta { color: var(--muted); font-size: 0.85rem; }
strong { font-weight: 650; }
.yes { color: var(--good); font-weight: 650; }
.no  { color: var(--bad); font-weight: 650; }
.part { color: var(--warn); font-weight: 650; }
ul, ol { padding-left: 1.4rem; }
li { margin: 0.3em 0; }
"""


def to_html(md_text: str, title: str) -> str:
    body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "toc", "attr_list", "sane_lists"],
    )
    # Wrap tables so wide ones scroll inside their own box rather than the page.
    body = body.replace("<table>", '<div class="table-scroll"><table>').replace(
        "</table>", "</table></div>"
    )
    # Colour the verdict glyphs used in the summary table.
    body = re.sub(r"✅", '<span class="yes">✅</span>', body)
    body = re.sub(r"❌", '<span class="no">❌</span>', body)
    body = re.sub(r"⚠️", '<span class="part">⚠️</span>', body)
    return (
        f"<!doctype html>\n<html lang=\"en\">\n<head>\n"
        f'<meta charset="utf-8">\n'
        f'<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{title}</title>\n<style>{CSS}</style>\n</head>\n"
        f'<body><div class="wrap">\n{body}\n</div></body>\n</html>\n'
    )


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "report.md"
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "report.html"
    md_text = src.read_text()
    m = re.search(r"^#\s+(.+)$", md_text, re.M)
    title = m.group(1).strip() if m else "Report"
    dst.write_text(to_html(md_text, title))
    print(f"wrote {dst} ({dst.stat().st_size / 1024:.0f} KB) from {src.name}")


if __name__ == "__main__":
    main()
