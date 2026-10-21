"""Djinni recruitment datasets (lang-uk, MIT): 141,897 English IT job descriptions and ~210k anonymised
candidate profiles (CVs). Loaded from the Hugging Face snapshot downloaded by scripts/download_assets.py.

The exact column names are verified at first load and written to docs/data_schema.md; this module
normalises them to a fixed schema:
  jobs: job_id, title, description, company, category (Primary Keyword), exp_years, english_level, published, text
  cvs : cv_id, title, category, exp_years, english_level, moreinfo, looking_for, highlights, cv, text (PII-scrubbed)"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import pandas as pd

from ..utils.config import ROOT, data_cfg
from ..utils.pii import scrub
from ..utils.text import strip_html

JOB_COLS = {
    "title": ["Position", "position", "title", "Title"],
    "description": ["Long Description", "long_description", "description", "Description", "Full Description"],
    "company": ["Company Name", "company_name", "company", "Company"],
    "category": ["Primary Keyword", "primary_keyword", "category"],
    "exp_years": ["Exp Years", "exp_years", "Experience Years", "experience_years"],
    "english_level": ["English Level", "english_level"],
    "published": ["Published", "published", "Date", "date"],
}
CV_COLS = {
    "title": ["Position", "position", "title"],
    "category": ["Primary Keyword", "primary_keyword", "category"],
    "exp_years": ["Experience Years", "experience_years", "Exp Years", "exp_years"],
    "english_level": ["English Level", "english_level"],
    "moreinfo": ["Moreinfo", "moreinfo", "More Info"],
    "looking_for": ["Looking For", "looking_for"],
    "highlights": ["Highlights", "highlights"],
    "cv": ["CV", "cv", "text", "Text"],
}


def _read_dir(local: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(local / "**" / "*.parquet"), recursive=True))
    if files:
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    files = sorted(glob.glob(str(local / "**" / "*.csv"), recursive=True))
    if files:
        return pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    files = sorted(glob.glob(str(local / "**" / "*.json*"), recursive=True))
    if files:
        return pd.concat([pd.read_json(f, lines=f.endswith("l")) for f in files], ignore_index=True)
    raise FileNotFoundError(f"no data files under {local}; run scripts/download_assets.py on the Mac first")


def _pick(df: pd.DataFrame, cands: list[str]) -> str | None:
    for c in cands:
        if c in df.columns:
            return c
    low = {c.lower().replace("_", " "): c for c in df.columns}
    for c in cands:
        if c.lower().replace("_", " ") in low:
            return low[c.lower().replace("_", " ")]
    return None


def _normalise(df: pd.DataFrame, mapping: dict[str, list[str]]) -> tuple[pd.DataFrame, dict[str, str | None]]:
    used = {k: _pick(df, v) for k, v in mapping.items()}
    out = pd.DataFrame({k: (df[c] if c else pd.Series([None] * len(df))) for k, c in used.items()})
    return out, used


def _years(x) -> float | None:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    m = re.search(r"\d+(?:\.\d+)?", str(x))
    return float(m.group(0)) if m else None


_STR = "string[pyarrow]"  # Arrow-backed strings: ~3x less memory than object columns for 350k long texts


def _clean_text_series(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str)
    # only run the HTML parser/unescaper where needed (0.1 % of Djinni texts carry tags or entities)
    mask = (s.str.contains("<", regex=False) & s.str.contains(">", regex=False)) | s.str.contains(r"&(?:#\d+|[a-z]+);", regex=True)
    if mask.any():
        s = s.copy()
        s.loc[mask] = s.loc[mask].map(strip_html)
    return s.str.strip().astype(_STR)


def load_jobs(raw: bool = False) -> tuple[pd.DataFrame, dict]:
    local = ROOT / data_cfg()["datasets"]["djinni_jobs"]["local"]
    df = _read_dir(local)
    if raw:
        return df, {"columns": list(df.columns)}
    out, used = _normalise(df, JOB_COLS)
    src_cols = list(df.columns)
    del df
    out["job_id"] = pd.Series([f"djinni-job-{i}" for i in range(len(out))], dtype=_STR)
    out["description"] = _clean_text_series(out["description"])
    out["title"] = out["title"].fillna("").astype(str).str.strip().astype(_STR)
    out["exp_years_num"] = out["exp_years"].map(_years)
    out["text"] = (out["title"].astype(str) + "\n" + out["description"].astype(str)).str.strip().astype(_STR)
    out["n_chars"] = out["description"].str.len().astype("int32")
    return out, {"source_columns": src_cols, "mapping": used, "rows": len(out)}


def load_cvs(raw: bool = False, scrub_pii: bool = True, keep_fields: bool = False) -> tuple[pd.DataFrame, dict]:
    """Memory-lean: the four free-text fields are merged into `text` and replaced by presence flags
    (has_moreinfo, ...) unless keep_fields=True. PII is scrubbed in place."""
    local = ROOT / data_cfg()["datasets"]["djinni_cvs"]["local"]
    df = _read_dir(local)
    if raw:
        return df, {"columns": list(df.columns)}
    out, used = _normalise(df, CV_COLS)
    src_cols = list(df.columns)
    del df
    out["cv_id"] = pd.Series([f"djinni-cv-{i}" for i in range(len(out))], dtype=_STR)
    out["title"] = out["title"].fillna("").astype(str).str.strip().astype(_STR)
    fields = ("moreinfo", "looking_for", "highlights", "cv")
    labels = {"moreinfo": "Experience", "looking_for": "Looking for", "highlights": "Highlights", "cv": ""}
    text = ("Position: " + out["title"].astype(str))
    for c in fields:
        col = _clean_text_series(out[c])
        out[f"has_{c}"] = (col.str.len() > 0)
        prefix = (labels[c] + ": ") if labels[c] else ""
        text = text + "\n\n" + (prefix + col.astype(str)).where(col.str.len() > 0, "")
        if keep_fields:
            out[c] = col
        else:
            out.drop(columns=[c], inplace=True)
    text = text.str.replace(r"\n{3,}", "\n\n", regex=True).str.strip()
    if scrub_pii:
        counts = []

        def _scrub(t: str) -> str:
            r = scrub(t)
            counts.append(sum(r.counts.values()))
            return r.text

        text = text.map(_scrub)
        out["pii_removed"] = pd.Series(counts, index=out.index, dtype="int32")
    out["text"] = text.astype(_STR)
    out["exp_years_num"] = out["exp_years"].map(_years)
    out["n_chars"] = out["text"].str.len().astype("int32")
    return out, {"source_columns": src_cols, "mapping": used, "rows": len(out)}
