"""ESCO skills vocabulary (v1.1.0, English). Reads the official CSV when present
(data/raw/esco/1.1.0/en/skills_en.csv); otherwise falls back to the label list shipped with the
Skill-Extraction-benchmark (same 13.9k preferred labels, no alt labels / descriptions)."""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from ..utils.config import ROOT, data_cfg
from . import esco_benchmark


def csv_dir() -> Path:
    return ROOT / data_cfg()["datasets"]["esco"]["local"]


def available() -> bool:
    return (csv_dir() / "skills_en.csv").exists()


def load_skills() -> pd.DataFrame:
    """Columns: uri, label, alt_labels (list), description, skill_type, reuse_level, source."""
    if available():
        df = pd.read_csv(csv_dir() / "skills_en.csv", keep_default_na=False)
        out = pd.DataFrame({
            "uri": df["conceptUri"],
            "label": df["preferredLabel"],
            "alt_labels": df["altLabels"].map(lambda s: [a.strip() for a in str(s).split("\n") if a.strip()]),
            "description": df.get("description", ""),
            "skill_type": df.get("skillType", ""),
            "reuse_level": df.get("reuseLevel", ""),
        })
        out["source"] = "esco_csv"
        return out
    labels = esco_benchmark.esco_label_list()
    return pd.DataFrame({"uri": [""] * len(labels), "label": labels, "alt_labels": [[] for _ in labels], "description": "", "skill_type": "", "reuse_level": "", "source": "benchmark_label_list"})


def vocabulary() -> list[str]:
    return load_skills()["label"].tolist()


def workrb_cache_dir(version: str = "1.1.0", lang: str = "en") -> Path:
    import appdirs

    return Path(appdirs.user_cache_dir("wteb")) / "esco" / version / lang


def ensure_workrb_esco(version: str = "1.1.0", lang: str = "en") -> dict[str, Any]:
    """WorkRB looks for ESCO under appdirs.user_cache_dir('wteb')/esco/<version>/<lang>/ and would try to
    download it from the EU site (unreachable from the workspace). We install it ourselves:
      - the official CSVs when data/raw/esco/<version>/<lang>/ exists (final numbers), else
      - a STAND-IN built from the benchmark's 13.9k preferred-label list (no alt labels, no occupations),
        marked with a `_STANDIN` file so results are flagged provisional and re-run later."""
    dst = workrb_cache_dir(version, lang)
    dst.mkdir(parents=True, exist_ok=True)
    marker = dst / "_STANDIN"
    if available():
        for f in csv_dir().glob("*.csv"):
            if marker.exists() or not (dst / f.name).exists():
                shutil.copy2(f, dst / f.name)
        if marker.exists():
            marker.unlink()
        return {"esco_source": "esco_csv", "path": str(dst)}
    if not (dst / f"skills_{lang}.csv").exists() or marker.exists():
        labels = esco_benchmark.esco_label_list()
        pd.DataFrame({"conceptUri": [f"standin://skill/{i}" for i in range(len(labels))], "preferredLabel": labels,
                      "altLabels": [None] * len(labels), "description": [""] * len(labels)}).to_csv(dst / f"skills_{lang}.csv", index=False)
        pd.DataFrame({"conceptUri": [], "preferredLabel": []}).to_csv(dst / f"occupations_{lang}.csv", index=False)
        pd.DataFrame({"occupationUri": [], "skillUri": [], "relationType": []}).to_csv(dst / f"occupationSkillRelations_{lang}.csv", index=False)
        marker.write_text("stand-in ESCO built from the benchmark label list; replace with the official CSVs")
    return {"esco_source": "benchmark_label_list_standin", "path": str(dst)}
