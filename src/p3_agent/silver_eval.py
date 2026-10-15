"""Silver evaluation (large but rough): 1,000 Djinni CVs vs the whole djinni_jobs index, where a job is
'relevant' if it has the same Primary Keyword category as the CV. Baseline on all CVs; the two agents on a
subset (budget). Usage: python -m src.p3_agent.silver_eval [--n 1000] [--agents-n 60] [--workers 3]"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from ..data import djinni
from ..index.store import JobIndex
from ..utils.config import ROOT, data_cfg, models_cfg
from ..utils.io import write_results
from .baseline import EmbeddingBaseline
from .evaluate import LABELS, ndcg_at_k
from .multi_agent import MultiAgent
from .single_agent import SingleAgent
from .tools import ToolBox


def metrics(ranked_cats: list[str], cat: str, k: int = 10) -> dict[str, float]:
    rel = [1 if c == cat else 0 for c in ranked_cats[:k]]
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rel))
    idcg = sum(1 / math.log2(i + 2) for i in range(k))
    rr = next((1 / (i + 1) for i, r in enumerate(rel) if r), 0.0)
    return {"ndcg10": dcg / idcg, "p_at_10": sum(rel) / k, "mrr": rr}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=models_cfg()["p3_agent"]["silver_n_cvs"])
    ap.add_argument("--agents-n", type=int, default=60)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--model", default="agent")
    args = ap.parse_args()

    jobs = pd.read_parquet(ROOT / "data" / "processed" / "djinni_jobs.parquet")
    cat_of = dict(zip(jobs["job_id"], jobs["category"]))
    cats = jobs["category"].value_counts().head(models_cfg()["p3_agent"]["n_categories"]).index.tolist()
    cvs, _ = djinni.load_cvs()
    gold_ids = set(pd.read_csv(ROOT / "data" / "labels" / "pairs_to_label.csv")["cv_id"]) if (ROOT / "data" / "labels" / "pairs_to_label.csv").exists() else set()
    cvs = cvs[(cvs["n_chars"] >= data_cfg()["datasets"]["djinni_cvs"]["min_cv_chars"]) & cvs["category"].isin(cats) & ~cvs["cv_id"].isin(gold_ids)]
    sample = cvs.sample(n=min(args.n, len(cvs)), random_state=42).reset_index(drop=True)
    tb = ToolBox("djinni_jobs")
    _ = tb.extractor, tb.normalizer
    rows = []
    t0 = time.time()
    base = EmbeddingBaseline(tb, top_n=10)
    for i, r in sample.iterrows():
        run = base.run(r["text"])
        rows.append({"approach": "baseline", "cv_id": r["cv_id"], "category": r["category"], **metrics([cat_of.get(e["job_id"], "") for e in run.ranking], r["category"]), "cost_usd": 0.0, "llm_calls": 0})
    print(f"baseline on {len(sample)} CVs in {(time.time()-t0)/60:.1f} min")

    sub = sample.head(args.agents_n)

    def run_agents(r):
        out = []
        for key, cls in [("single", SingleAgent), ("multi", MultiAgent)]:
            box = ToolBox("djinni_jobs", extractor=tb.extractor, normalizer=tb.normalizer)
            ag = cls(box, model=args.model, top_n=10, task=f"p3_silver_{key}")
            run = ag.run(r["text"])
            out.append({"approach": key, "cv_id": r["cv_id"], "category": r["category"], **metrics([cat_of.get(e["job_id"], "") for e in run.ranking], r["category"]), "cost_usd": run.cost_usd, "llm_calls": run.llm_calls, "error": run.error})
        return out

    if args.agents_n > 0:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for k, res in enumerate(ex.map(run_agents, [r for _, r in sub.iterrows()]), 1):
                rows += res
                if k % 10 == 0:
                    print(f"  agents {k}/{len(sub)} ({(time.time()-t0)/60:.1f} min, spent ${sum(x['cost_usd'] for x in rows):.2f})", flush=True)
    df = pd.DataFrame(rows)
    write_results(df, "results/p3/silver_per_cv.csv", meta={"n_baseline": len(sample), "n_agents": args.agents_n, "relevance": "same Primary Keyword category"})
    agg = df.groupby("approach").agg(n_cvs=("cv_id", "nunique"), ndcg10=("ndcg10", "mean"), p_at_10=("p_at_10", "mean"), mrr=("mrr", "mean"), cost_usd=("cost_usd", "mean"), llm_calls=("llm_calls", "mean")).reset_index()
    agg["approach_label"] = agg["approach"].map({"baseline": LABELS["baseline"], "single": LABELS["single"], "multi": LABELS["multi"]})
    write_results(agg.round(4), "results/p3/silver_metrics.csv", meta={"relevance": "same category", "n_baseline": len(sample), "n_agents": args.agents_n})
    print(agg.round(3).to_string())

    os.environ["SMC_SOURCE"] = "src/p3_agent/silver_eval.py"
    import plotly.express as px

    from ..utils import viz

    viz.setup()
    m = agg.melt(id_vars=["approach_label"], value_vars=["ndcg10", "p_at_10", "mrr"], var_name="metric", value_name="value")
    m["metric"] = m["metric"].map({"ndcg10": "nDCG@10", "p_at_10": "Precision@10", "mrr": "MRR"})
    fig = px.bar(m, x="metric", y="value", color="approach_label", barmode="group", text=m["value"].map(lambda v: f"{v:.2f}"), title=f"Silver evaluation: same-category relevance over the full index (baseline n={len(sample)}, agents n={args.agents_n})")
    fig.update_traces(textposition="outside")
    fig.update_layout(yaxis_range=[0, 1], xaxis_title="", legend_title_text="")
    viz.save_fig(fig, "p3_silver_eval", "Problem 3 silver set: ranking quality with 'same job category' as a rough relevance signal, over the whole 6k-posting index (baseline on all sampled CVs, agents on a subset).")


if __name__ == "__main__":
    main()
