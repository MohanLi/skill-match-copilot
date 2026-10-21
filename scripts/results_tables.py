"""Render the result tables used in REPORT.md / the slides as Markdown, straight from results/*.csv.

Usage: python scripts/results_tables.py [p1|p2|nil|p3|all]   (default: all; prints to stdout)
Every table names the file it was read from, so a reviewer can trace each number."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def md(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---:" if pd.api.types.is_numeric_dtype(df[c]) else "---" for c in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append("" if pd.isna(v) else floatfmt.format(v))
            else:
                cells.append("" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def p1() -> str:
    f = ROOT / "results/p1/metrics_summary_test.csv"
    d = pd.read_csv(f)
    names = {"tagger": "Fine-tuned JobBERT tagger (champion)", "llm_haiku": "Few-shot claude-haiku-5-5 (challenger)", "dict": "Dictionary baseline"}
    d["approach"] = d["approach"].map(names).fillna(d["approach"])
    d = d.sort_values(["column", "strict_f1_mean"], ascending=[True, False])
    out = d[["approach", "column", "strict_precision", "strict_recall", "strict_f1_mean", "loose_f1", "fp_spans_per_100_all_O", "strict_f1_tech", "strict_f1_house", "latency_ms", "hallucination_rate"]]
    out = out.rename(columns={"column": "span type", "strict_f1_mean": "strict F1", "strict_precision": "strict P", "strict_recall": "strict R", "loose_f1": "loose F1",
                              "fp_spans_per_100_all_O": "FP spans / 100 empty sentences", "strict_f1_tech": "F1 tech", "strict_f1_house": "F1 house", "latency_ms": "ms / sentence", "hallucination_rate": "hallucination rate"})
    return f"### Problem 1 — SkillSpan test split (`{f.relative_to(ROOT)}`)\n\n" + md(out)


def p2() -> str:
    parts = []
    for split in ("val", "test"):
        f = ROOT / f"results/p2/workrb_{split}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        d["task"] = d["task"].str.replace("Skill Extraction ", "")
        piv = d.pivot_table(index="model", columns="task", values=["map", "rp@10", "mrr"])
        piv.columns = [f"{m.upper()} {t}" for m, t in piv.columns]
        order = d.groupby("model")["map"].mean().sort_values(ascending=False).index
        piv = piv.loc[order].reset_index()
        cost = d.groupby("model")["llm_cost_usd"].first()
        piv["LLM cost (USD)"] = piv["model"].map(cost).round(3)
        parts.append(f"### Problem 2 — WorkRB {split} split (`{f.relative_to(ROOT)}`)\n\n" + md(piv))
    return "\n\n".join(parts)


def nil() -> str:
    parts = []
    for f in sorted((ROOT / "results/p2").glob("nil*.csv")):
        if "sweep" in f.name or "disagree" in f.name:
            continue
        d = pd.read_csv(f)
        meta = json.loads(f.with_suffix(".csv.meta.json").read_text()) if f.with_suffix(".csv.meta.json").exists() else {}
        out = d[["approach", "subset", "threshold", "encoder", "joint_accuracy", "nil_precision", "nil_recall", "nil_f1", "acc_on_mapped_phrases", "llm_cost_usd"]]
        parts.append(f"### 'Not in ESCO' — `{f.relative_to(ROOT)}` (threshold tuned on {meta.get('tuned_on', '?')})\n\n" + md(out))
    return "\n\n".join(parts)


def p3() -> str:
    parts = []
    for f in sorted((ROOT / "results/p3").glob("metrics_*.csv")):
        d = pd.read_csv(f)
        parts.append(f"### Problem 3 — `{f.relative_to(ROOT)}`\n\n" + md(d))
    k = ROOT / "data/labels/kappa.json"
    if k.exists():
        kk = json.loads(k.read_text())
        parts.append("### Gold labels — rater agreement (`data/labels/kappa.json`)\n\n" + md(pd.DataFrame([{kx: kk[kx] for kx in kk if not isinstance(kk[kx], (list, dict))}])))
    return "\n\n".join(parts) if parts else "(no Problem 3 results yet)"


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    fns = {"p1": p1, "p2": p2, "nil": nil, "p3": p3}
    for k, fn in fns.items():
        if which in ("all", k):
            print(fn(), "\n")
