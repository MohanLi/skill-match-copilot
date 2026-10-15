"""open-apply-jobs daily snapshot (filtered to data/AI/ML titles by scripts/download_assets.py).
Files: data/raw/live_jobs/live_jobs_<date>_ai_data.parquet, *_sample20k.parquet, *_stats.json, snapshot_date.txt"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from ..utils.config import ROOT, data_cfg
from ..utils.text import strip_html

COLS = {
    "title": ["title", "job_title", "name"],
    "description_html": ["description_html", "description", "content", "body"],
    "company": ["company", "company_name", "organization", "org"],
    "locations": ["locations", "location", "city"],
    "remote": ["remote", "is_remote", "workplace_type", "remote_type"],
    "department": ["department", "team"],
    "employment_type": ["employment_type", "type"],
    "posted_at": ["posted_at", "published_at", "created_at", "date_posted", "first_seen"],
    "apply_url": ["apply_url", "url", "link", "job_url"],
    "source": ["source", "ats", "provider", "platform"],
}


def local_dir() -> Path:
    return ROOT / data_cfg()["datasets"]["live_jobs"]["local"]


def available() -> bool:
    return (local_dir() / "snapshot_date.txt").exists()


def snapshot_date() -> str | None:
    p = local_dir() / "snapshot_date.txt"
    return p.read_text().strip() if p.exists() else None


def stats() -> dict:
    d = snapshot_date()
    p = local_dir() / f"live_jobs_{d}_stats.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _pick(df: pd.DataFrame, cands: list[str]) -> str | None:
    for c in cands:
        if c in df.columns:
            return c
    return None


def _norm(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    used = {k: _pick(df, v) for k, v in COLS.items()}
    out = pd.DataFrame({k: (df[c] if c else pd.Series([None] * len(df))) for k, c in used.items()})
    if "id" in df.columns:
        out["job_id"] = "live-" + df["id"].astype(str)
    else:
        out["job_id"] = [f"live-{i}" for i in range(len(out))]
    out["description"] = out["description_html"].fillna("").astype(str).map(strip_html)
    out["title"] = out["title"].fillna("").astype(str).str.strip()
    out["locations"] = out["locations"].map(lambda v: ", ".join(map(str, v)) if isinstance(v, (list, tuple)) or hasattr(v, "tolist") and not isinstance(v, str) else ("" if v is None else str(v)))
    out["text"] = (out["title"] + "\n" + out["description"]).str.strip()
    out["n_chars"] = out["description"].str.len()
    return out, used


def load_filtered() -> tuple[pd.DataFrame, dict]:
    d = snapshot_date()
    df = pd.read_parquet(local_dir() / f"live_jobs_{d}_ai_data.parquet")
    out, used = _norm(df)
    return out, {"snapshot_date": d, "source_columns": list(df.columns), "mapping": used, "rows": len(out)}


def load_sample() -> tuple[pd.DataFrame, dict]:
    d = snapshot_date()
    df = pd.read_parquet(local_dir() / f"live_jobs_{d}_sample20k.parquet")
    out, used = _norm(df)
    return out, {"snapshot_date": d, "source_columns": list(df.columns), "mapping": used, "rows": len(out)}
