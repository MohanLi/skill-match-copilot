"""Result writing with provenance. Every metrics table gets a sidecar .meta.json recording
when it was produced, from which commit, with which models / prompt versions / seeds."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ROOT, git_commit


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def write_results(df: pd.DataFrame, rel_path: str, meta: dict[str, Any] | None = None, index: bool = False) -> Path:
    out = ROOT / rel_path
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=index)
    side = out.with_suffix(out.suffix + ".meta.json")
    payload = {
        "file": _rel(out),
        "rows": int(len(df)),
        "columns": list(map(str, df.columns)),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": git_commit(),
        **(meta or {}),
    }
    side.write_text(json.dumps(payload, indent=2, default=str))
    return out


def write_json(obj: Any, rel_path: str) -> Path:
    out = ROOT / rel_path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(obj, indent=2, default=str, ensure_ascii=False))
    return out


def read_json(rel_path: str) -> Any:
    return json.loads((ROOT / rel_path).read_text())


def append_jsonl(obj: Any, rel_path: str) -> None:
    out = ROOT / rel_path
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, default=str, ensure_ascii=False) + "\n")


def read_jsonl(rel_path: str) -> list[Any]:
    p = ROOT / rel_path
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
