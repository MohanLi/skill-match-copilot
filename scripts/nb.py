#!/usr/bin/env python3
"""Build + execute notebooks from percent-format Python sources.

Notebook sources live in notebooks/src/*.py (cells separated by `# %%` / `# %% [markdown]`), which keeps
them diff-able in git. This script turns a source into notebooks/<name>.ipynb, executes it (outputs embedded)
and exports notebooks/html/<name>.html for reading without Jupyter.

Usage:
  python scripts/nb.py notebooks/src/01_eda_skillspan.py          # build + execute + html
  python scripts/nb.py notebooks/src/01_eda_skillspan.py --no-exec
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parents[1]
CELL_RE = re.compile(r"^# %%(.*)$")


def build(src: Path) -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    cells: list[tuple[str, list[str]]] = []
    kind, buf = "code", []
    for line in src.read_text(encoding="utf-8").splitlines():
        m = CELL_RE.match(line)
        if m:
            if buf and "".join(buf).strip():
                cells.append((kind, buf))
            kind = "markdown" if "[markdown]" in m.group(1) else "code"
            buf = []
        else:
            buf.append(line)
    if buf and "".join(buf).strip():
        cells.append((kind, buf))
    for kind, lines in cells:
        text = "\n".join(lines).strip("\n")
        if kind == "markdown":
            text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in text.splitlines())
            nb.cells.append(nbformat.v4.new_markdown_cell(text))
        else:
            nb.cells.append(nbformat.v4.new_code_cell(text))
    return nb


def run(src: Path, execute: bool = True, timeout: int = 7200) -> Path:
    name = src.stem
    out = ROOT / "notebooks" / f"{name}.ipynb"
    nb = build(src)
    if execute:
        t0 = time.time()
        client = NotebookClient(nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}, allow_errors=False)
        client.execute()
        print(f"[nb] executed {name} in {(time.time() - t0) / 60:.1f} min")
    nbformat.write(nb, out)
    html_dir = ROOT / "notebooks" / "html"
    html_dir.mkdir(exist_ok=True)
    exporter = HTMLExporter()
    exporter.exclude_input_prompt = True
    body, _ = exporter.from_notebook_node(nb)
    (html_dir / f"{name}.html").write_text(body, encoding="utf-8")
    print(f"[nb] wrote {out.relative_to(ROOT)} and notebooks/html/{name}.html")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--no-exec", action="store_true")
    ap.add_argument("--timeout", type=int, default=7200)
    a = ap.parse_args()
    run(Path(a.src), execute=not a.no_exec, timeout=a.timeout)
