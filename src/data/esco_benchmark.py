"""Skill-Extraction-benchmark (Decorte et al.) — TECH / HOUSE / TECHWOLF annotation CSVs from the
GitHub repo: columns sentence, span, sub_span, label. `label` is an ESCO 1.1.0 preferred label,
'LABEL NOT PRESENT' (no such ESCO skill) or 'UNDERSPECIFIED' (too vague to map)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..utils.config import ROOT, data_cfg

NIL_LABELS = {"LABEL NOT PRESENT", "UNDERSPECIFIED"}
FILES = {
    ("tech", "validation"): "tech_validation_annotations.csv",
    ("tech", "test"): "tech_test_annotations.csv",
    ("house", "validation"): "house_validation_annotations.csv",
    ("house", "test"): "house_test_annotations.csv",
    ("techwolf", "test"): "techwolf_test_annotations.csv",
}


def _dir() -> Path:
    return ROOT / data_cfg()["datasets"]["esco_benchmark"]["local_github"] / "data"


def load(subset: str, split: str) -> pd.DataFrame:
    df = pd.read_csv(_dir() / FILES[(subset, split)], keep_default_na=False)
    df["subset"], df["split"] = subset, split
    df["status"] = df["label"].map(lambda x: x if x in NIL_LABELS else "ESCO")
    return df


def load_all() -> pd.DataFrame:
    return pd.concat([load(s, sp) for (s, sp) in FILES], ignore_index=True)


def esco_label_list() -> list[str]:
    """The 13,895 ESCO 1.1.0 skill labels shipped with the benchmark (fallback vocabulary)."""
    p = _dir() / "skills_en_label.txt"
    return [l.rstrip("\n") for l in open(p, encoding="utf-8") if l.strip()]


def grouped_by_sentence(df: pd.DataFrame, drop_nil: bool = True) -> pd.DataFrame:
    """One row per sentence with the list of ESCO labels (what WorkRB's ranking tasks use)."""
    d = df[~df["label"].isin(NIL_LABELS)] if drop_nil else df
    g = d.groupby(["subset", "split", "sentence"])["label"].apply(lambda s: sorted(set(s))).reset_index()
    return g
