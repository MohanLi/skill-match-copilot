"""Weekly drift + hallucination report (Model Ops). Produces one dashboard figure and a JSON summary:
  1. share of extracted phrases that map to NOT_IN_ESCO per period (new skills appearing)        — live + Djinni
  2. unsupported-claim rate per day with the alert threshold (verifier / faithfulness checker)   — API log + eval runs
  3. posting volume by category per period                                                       — live + Djinni
  4. similarity of new postings to the tagger's training data (mean nearest-neighbour cosine)     — drift proxy
  5. emerging-skills queue: most frequent NOT_IN_ESCO phrases (human review candidates)

  python -m src.monitoring.drift_report [--alert 0.02]
Writes results/monitoring/drift_report.json, results/monitoring/emerging_skills.csv, reports/figures/ops_dashboard.*"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter

import numpy as np
import pandas as pd

from ..utils.config import ROOT, models_cfg
from ..utils.io import read_jsonl, write_json, write_results


def _period(s: pd.Series, freq: str) -> pd.Series:
    d = pd.to_datetime(s, errors="coerce", utc=True)
    return d.dt.to_period(freq).astype(str)


def load_indexed(pool: str) -> pd.DataFrame | None:
    p = ROOT / "data" / "processed" / f"{pool}_jobs.parquet"
    return pd.read_parquet(p) if p.exists() else None


def nil_share_by_period(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    d = df.copy()
    d["period"] = _period(d["published"], freq)
    d = d[d["period"] != "NaT"]
    g = d.groupby("period").agg(postings=("job_id", "size"), esco=("n_esco", "sum"), nil=("n_not_in_esco", "sum")).reset_index()
    g["nil_share"] = g["nil"] / (g["esco"] + g["nil"]).replace(0, np.nan)
    return g


def similarity_to_training(df: pd.DataFrame, n: int = 400) -> float | None:
    """Mean cosine between live posting embeddings and their nearest SkillSpan training sentence."""
    try:
        from ..data import skillspan as ss
        from ..index.embedder import get_text_embedder
        from ..index.store import JobIndex

        train = [" ".join(r["tokens"]) for r in ss.load_split("train") if 5 <= len(r["tokens"]) <= 40][:3000]
        T = np.array(get_text_embedder().embed(train))  # same encoder as the stored job embeddings
        idx = JobIndex("live_jobs" if (df["source"] == "live").all() else "djinni_jobs")
        ids = df["job_id"].head(n).tolist()
        E = np.array(idx.col.get(ids=ids, include=["embeddings"])["embeddings"])
        if len(E) == 0:
            return None
        sims = (E @ T.T).max(axis=1)
        return float(np.mean(sims))
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alert", type=float, default=0.02, help="unsupported-claim rate alert threshold")
    args = ap.parse_args()
    os.environ["SMC_SOURCE"] = "src/monitoring/drift_report.py"
    live, dj = load_indexed("live_jobs"), load_indexed("djinni_jobs")
    report: dict = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "alert_threshold": args.alert}

    nil_live = nil_share_by_period(live, "W") if live is not None and len(live) else pd.DataFrame()
    nil_dj = nil_share_by_period(dj, "Q") if dj is not None and len(dj) else pd.DataFrame()
    report["nil_share_live_weekly"] = nil_live.to_dict("records") if len(nil_live) else []
    report["nil_share_djinni_quarterly"] = nil_dj.to_dict("records") if len(nil_dj) else []

    # unsupported-claim rate: API requests log + evaluation runs (as the launch baseline)
    reqs = pd.DataFrame([r for r in read_jsonl("logs/requests.jsonl") if r.get("n_claims")])
    series = []
    if len(reqs):
        reqs["day"] = pd.to_datetime(reqs["ts"], unit="s").dt.date.astype(str)
        g = reqs.groupby("day").apply(lambda x: pd.Series({"unsupported_rate": float((x["hallucination_rate"] * x["n_claims"]).sum() / x["n_claims"].sum()), "requests": len(x)}))
        series = g.reset_index().to_dict("records")
    for p in sorted((ROOT / "results" / "p3").glob("per_cv_*.csv")):
        d = pd.read_csv(p)
        for app, x in d[d["n_claims"] > 0].groupby("approach"):
            series.append({"day": f"eval:{p.stem.replace('per_cv_', '')}:{app}", "unsupported_rate": float(1 - x["n_supported"].sum() / x["n_claims"].sum()), "requests": int(len(x))})
    report["unsupported_claim_rate"] = series
    report["alerts"] = [s for s in series if s["unsupported_rate"] > args.alert]

    vol = pd.DataFrame()
    if live is not None and len(live):
        v = live.copy()
        v["period"] = _period(v["published"], "W")
        vol = v.groupby(["period", "category"]).size().reset_index(name="postings")
        vol = vol[vol["category"].isin(v["category"].value_counts().head(6).index)]
    report["volume_by_category_live"] = vol.to_dict("records") if len(vol) else []
    report["similarity_to_training_live"] = similarity_to_training(live) if live is not None and len(live) else None
    report["similarity_to_training_djinni"] = similarity_to_training(dj) if dj is not None and len(dj) else None

    # emerging skills queue
    src = live if live is not None and len(live) else dj
    emerging = pd.DataFrame()
    try:
        from ..p2_normalize.normalizer import get_normalizer

        nm = get_normalizer()
        phrases: Counter = Counter()
        if src is not None:
            for raw in src["raw_skills"].fillna("").head(3000):
                phrases.update(p.strip().lower() for p in raw.split("|") if 2 < len(p.strip()) < 40)
        top = [p for p, _ in phrases.most_common(600)]
        mapped = nm.map_phrases(top) if top else []
        emerging = pd.DataFrame([{"phrase": m["phrase"], "count": phrases[m["phrase"]], "best_esco": m["candidates"][0] if m.get("candidates") else None, "score": m["score"]} for m in mapped if not m["in_esco"]]).sort_values("count", ascending=False).head(40)
    except Exception as e:  # noqa: BLE001
        report["emerging_error"] = str(e)[:200]
    if len(emerging):
        write_results(emerging, "results/monitoring/emerging_skills.csv")
        report["emerging_skills_top"] = emerging.head(15).to_dict("records")
    write_json(report, "results/monitoring/drift_report.json")

    # ---- dashboard
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from ..utils import viz

    viz.setup()
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Share of extracted phrases NOT in ESCO (new skills appearing)", f"Unsupported-claim rate per day/run (alert > {args.alert:.0%})", "Posting volume by category (live, weekly)", "Emerging-skills queue (top NOT-in-ESCO phrases)"), vertical_spacing=0.16, horizontal_spacing=0.12)
    if len(nil_dj):
        fig.add_trace(go.Scatter(x=nil_dj["period"], y=nil_dj["nil_share"], mode="lines+markers", name="Djinni (quarterly)", line=dict(color=viz.PALETTE[0])), row=1, col=1)
    if len(nil_live):
        fig.add_trace(go.Scatter(x=nil_live["period"], y=nil_live["nil_share"], mode="lines+markers", name="live snapshot (weekly)", line=dict(color=viz.PALETTE[1])), row=1, col=1)
    if series:
        s = pd.DataFrame(series)
        fig.add_trace(go.Bar(x=s["day"], y=s["unsupported_rate"], name="unsupported-claim rate", marker_color=[viz.STATUS["critical"] if v > args.alert else viz.PALETTE[0] for v in s["unsupported_rate"]]), row=1, col=2)
        fig.add_hline(y=args.alert, line_dash="dot", line_color=viz.STATUS["critical"], row=1, col=2)
    if len(vol):
        for i, (cat, g) in enumerate(vol.groupby("category")):
            fig.add_trace(go.Bar(x=g["period"], y=g["postings"], name=str(cat)[:30], marker_color=viz.PALETTE[i % 8]), row=2, col=1)
    if len(emerging):
        e = emerging.head(12).iloc[::-1]
        fig.add_trace(go.Bar(x=e["count"], y=e["phrase"], orientation="h", name="NOT in ESCO", marker_color=viz.PALETTE[1]), row=2, col=2)
    fig.update_layout(title="Model Ops dashboard — drift and hallucination monitoring", height=820, barmode="stack", legend=dict(orientation="h", y=-0.08))
    fig.update_yaxes(tickformat=".0%", row=1, col=1)
    fig.update_yaxes(tickformat=".0%", row=1, col=2)
    viz.save_fig(fig, "ops_dashboard", "Model Ops dashboard: share of phrases outside ESCO per period (new skills), unsupported-claim rate with the alert line, posting volume by category and the emerging-skills review queue.", height=820)
    print(json.dumps({k: v for k, v in report.items() if k not in ("nil_share_live_weekly", "nil_share_djinni_quarterly", "volume_by_category_live", "unsupported_claim_rate", "emerging_skills_top")}, indent=1))
    print("alerts:", report["alerts"])


if __name__ == "__main__":
    main()
