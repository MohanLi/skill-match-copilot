"""Problem 3 evaluation on the labelled pools: every approach ranks the same 10-job pool per CV.

Metrics: nDCG@10 (graded gains 0/1/2), Precision@3 and MRR (relevant = label 2), explanation
faithfulness (automatic checker), hallucination rate, LLM calls / tokens / cost / latency per resume.
Usage: python -m src.p3_agent.evaluate --approaches baseline single multi [multi_noverify] [--model agent] [--limit N]
Writes results/p3/per_cv_<tag>.csv, results/p3/metrics_<tag>.csv, results/p3/traces/<approach>/<cv_id>.json,
results/p3/failures_<tag>.csv and figures."""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import pandas as pd

from ..index.store import JobIndex
from ..utils.config import ROOT, models_cfg
from ..utils.io import write_json, write_results
from .baseline import EmbeddingBaseline
from .common import faithfulness_of_ranking, resume_profile
from .multi_agent import MultiAgent
from .single_agent import SingleAgent
from .tools import ToolBox

LABELS = {"baseline": "Embedding baseline (no LLM)", "single": "Single ReAct agent (champion)", "multi": "Multi-agent + verifier (challenger)", "multi_noverify": "Multi-agent without verifier (ablation)"}


def ndcg_at_k(ranked_labels: list[int], all_labels: list[int], k: int = 10) -> float | None:
    """Graded nDCG; None (excluded from averages) when the pool holds no relevant job at all."""
    dcg = sum((2 ** l - 1) / math.log2(i + 2) for i, l in enumerate(ranked_labels[:k]))
    ideal = sorted(all_labels, reverse=True)[:k]
    idcg = sum((2 ** l - 1) / math.log2(i + 2) for i, l in enumerate(ideal))
    return dcg / idcg if idcg > 0 else None


def precision_at_k(ranked_labels: list[int], k: int = 3, rel: int = 2) -> float:
    top = ranked_labels[:k]
    return sum(1 for l in top if l >= rel) / k


def mrr(ranked_labels: list[int], rel: int = 2) -> float | None:
    if not any(l >= rel for l in ranked_labels):
        return None
    for i, l in enumerate(ranked_labels):
        if l >= rel:
            return 1 / (i + 1)
    return None


def make_approach(key: str, tb: ToolBox, model: str):
    if key == "baseline":
        return EmbeddingBaseline(tb)
    if key == "single":
        return SingleAgent(tb, model=model)
    if key == "multi":
        return MultiAgent(tb, model=model)
    if key == "multi_noverify":
        return MultiAgent(tb, model=model, use_verifier=False, task="p3_multi_agent_noverify")
    raise ValueError(key)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--approaches", nargs="+", default=["baseline", "single", "multi"])
    ap.add_argument("--model", default="agent")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    tag = args.tag or ("haiku" if "haiku" in args.model else args.model)

    final = pd.read_csv(ROOT / "data" / "labels" / "final.csv")
    texts = pd.read_parquet(ROOT / "data" / "processed" / "p3_pairs_texts.parquet")
    cv_text = texts.drop_duplicates("cv_id").set_index("cv_id")["cv_text"].to_dict()
    pools = {cv: g.sort_values("pool_position") for cv, g in final.groupby("cv_id")}
    cv_ids = sorted(pools)
    if args.limit:
        cv_ids = cv_ids[: args.limit]
    index = JobIndex("djinni_jobs")
    tb_shared = ToolBox("djinni_jobs")  # loads models once; per-CV boxes reuse them
    _ = tb_shared.extractor, tb_shared.normalizer

    per_cv, traces_dir = [], ROOT / "results" / "p3" / "traces"
    for key in args.approaches:
        t_app = time.time()
        for n, cv in enumerate(cv_ids, 1):
            pool = pools[cv]
            pool_ids = pool["job_id"].tolist()
            labels = dict(zip(pool["job_id"], pool["label"].astype(int)))
            tb = ToolBox("djinni_jobs", pool_ids=pool_ids, extractor=tb_shared.extractor, normalizer=tb_shared.normalizer)
            app = make_approach(key, tb, args.model)
            resume = cv_text[cv]
            run = app.run(resume)
            ranked = [e["job_id"] for e in run.ranking if e["job_id"] in labels]
            missing = [j for j in pool_ids if j not in ranked]
            coverage = len(ranked) / len(pool_ids)
            # unranked pool jobs are appended in baseline (embedding) order so every approach ranks all 10
            if missing:
                fallback = [h["job_id"] for h in tb.search_jobs(resume[:4000], k=len(pool_ids)) if h["job_id"] in missing]
                ranked += fallback + [j for j in missing if j not in fallback]
            rl = [labels[j] for j in ranked]
            jobs = {j["job_id"]: j for j in index.get_many(pool_ids)}
            prof = resume_profile(tb, resume) if key != "baseline" else {"esco_skills": []}
            faith = faithfulness_of_ranking(run.ranking, resume, jobs, prof.get("esco_skills")) if key != "baseline" else {"n_claims": 0, "n_supported": 0, "hallucination_rate": None, "reasons": {}}
            rec = {"approach": key, "cv_id": cv, "cv_category": pool["cv_category"].iloc[0], "ndcg10": ndcg_at_k(rl, list(labels.values())), "p_at_3": precision_at_k(rl), "mrr": mrr(rl),
                   "p_at_3_rel1": precision_at_k(rl, rel=1), "mrr_rel1": mrr(rl, rel=1),  # relaxed: partial fit counts as relevant
                   "n_strong_in_pool": sum(1 for l in labels.values() if l == 2), "n_partial_or_better_in_pool": sum(1 for l in labels.values() if l >= 1), "coverage": coverage, "n_claims": faith["n_claims"], "n_supported": faith["n_supported"],
                   "hallucination_rate": faith["hallucination_rate"], "llm_calls": run.llm_calls, "input_tokens": run.input_tokens, "output_tokens": run.output_tokens,
                   "cost_usd": run.cost_usd, "latency_ms": run.latency_ms, "wall_ms": run.extra.get("wall_ms"), "tool_calls": run.tool_calls, "error": run.error,
                   "top1_label": rl[0] if rl else None, "ranked_order": "|".join(ranked), "reasons": json.dumps(faith["reasons"])}
            for k in ("verifier_flagged", "verifier_repaired", "verifier_dropped", "steps"):
                if k in run.extra:
                    rec[k] = run.extra[k]
            per_cv.append(rec)
            d = traces_dir / f"{key}_{tag}"
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{cv}.json").write_text(json.dumps({**run.to_dict(), "pool": [{"job_id": j, "label": labels[j], "title": jobs.get(j, {}).get("title")} for j in pool_ids], "metrics": {k: v for k, v in rec.items() if k not in ("ranked_order",)}, "faithfulness": faith}, indent=1, ensure_ascii=False, default=str))
            print(f"[{key}] {n}/{len(cv_ids)} {cv}: nDCG {rec['ndcg10'] if rec['ndcg10'] is None else round(rec['ndcg10'], 3)} P@3 {rec['p_at_3']:.2f} halluc {rec['hallucination_rate']} cost ${run.cost_usd:.3f} ({(time.time()-t_app)/60:.1f} min)", flush=True)

    df = pd.DataFrame(per_cv)
    write_results(df, f"results/p3/per_cv_{tag}.csv", meta={"model": args.model, "n_cvs": len(cv_ids)})
    agg = df.groupby("approach").agg(
        n_cvs=("cv_id", "nunique"), n_cvs_with_relevant=("ndcg10", "count"), ndcg10=("ndcg10", "mean"), ndcg10_std=("ndcg10", "std"), p_at_3=("p_at_3", "mean"), mrr=("mrr", "mean"),
        p_at_3_rel1=("p_at_3_rel1", "mean"), mrr_rel1=("mrr_rel1", "mean"),
        coverage=("coverage", "mean"), claims_per_cv=("n_claims", "mean"), supported_per_cv=("n_supported", "mean"),
        hallucination_rate=("hallucination_rate", lambda s: (1 - df.loc[s.index, "n_supported"].sum() / df.loc[s.index, "n_claims"].sum()) if df.loc[s.index, "n_claims"].sum() else None),
        llm_calls=("llm_calls", "mean"), input_tokens=("input_tokens", "mean"), output_tokens=("output_tokens", "mean"), cost_usd=("cost_usd", "mean"),
        latency_s=("latency_ms", lambda s: s.mean() / 1000), wall_s=("wall_ms", lambda s: s.mean() / 1000), errors=("error", lambda s: s.notna().sum()),
    ).reset_index()
    agg["approach_label"] = agg["approach"].map(LABELS)
    write_results(agg.round(4), f"results/p3/metrics_{tag}.csv", meta={"model": args.model, "labels": "data/labels/final.csv", "n_cvs": len(cv_ids)})
    print(agg.round(3).to_string())

    # failure analysis: wrong top-1 vs unsupported claims vs missed strong fits
    fails = []
    for _, r in df[df["approach"] != "baseline"].iterrows():
        kinds = []
        if r["top1_label"] is not None and r["top1_label"] < 2 and r["n_strong_in_pool"] > 0:
            kinds.append("wrong_top1")
        if (r["hallucination_rate"] or 0) > 0:
            kinds.append("unsupported_claim")
        if r["p_at_3"] < 1 and r["n_strong_in_pool"] >= 3:
            kinds.append("missed_strong_fit")
        if r["error"]:
            kinds.append("run_error")
        if kinds:
            fails.append({"approach": r["approach"], "cv_id": r["cv_id"], "failure_types": ",".join(kinds), "ndcg10": None if pd.isna(r["ndcg10"]) else round(r["ndcg10"], 3), "hallucination_rate": r["hallucination_rate"], "reasons": r["reasons"]})
    if fails:
        write_results(pd.DataFrame(fails), f"results/p3/failures_{tag}.csv")

    os.environ["SMC_SOURCE"] = "src/p3_agent/evaluate.py"
    import plotly.express as px

    from ..utils import viz

    viz.setup()
    m = agg.melt(id_vars=["approach_label"], value_vars=["ndcg10", "p_at_3", "mrr", "p_at_3_rel1", "mrr_rel1"], var_name="metric", value_name="value")
    m["metric"] = m["metric"].map({"ndcg10": "nDCG@10", "p_at_3": "P@3 (strong fit)", "mrr": "MRR (strong fit)", "p_at_3_rel1": "P@3 (partial or better)", "mrr_rel1": "MRR (partial or better)"})
    fig = px.bar(m, x="metric", y="value", color="approach_label", barmode="group", text=m["value"].map(lambda v: f"{v:.2f}"), title=f"Problem 3 — ranking quality on the labelled pools ({len(cv_ids)} CVs × 10 jobs)")
    fig.update_traces(textposition="outside")
    fig.update_layout(yaxis_range=[0, 1], xaxis_title="", legend_title_text="")
    viz.save_fig(fig, f"p3_ranking_quality_{tag}", f"Problem 3: nDCG@10, Precision@3 and MRR (relevant = strong fit, and relevant = partial fit or better) of the embedding baseline, the single ReAct agent and the multi-agent pipeline on the same {len(cv_ids)} labelled 10-job pools.", width=1200)
    a2 = agg[agg["approach"] != "baseline"]
    fig2 = px.bar(a2, x="approach_label", y="hallucination_rate", text=a2["hallucination_rate"].map(lambda v: f"{v:.1%}" if v is not None else ""), title="Explanation faithfulness: share of matched/missing-skill claims NOT supported by verbatim quotes")
    fig2.update_traces(textposition="outside", marker_color=viz.PALETTE[7])
    fig2.update_layout(yaxis_tickformat=".0%", xaxis_title="", yaxis_title="hallucination rate")
    viz.save_fig(fig2, f"p3_hallucination_{tag}", "Problem 3: hallucination rate of the explanations (unsupported claims ÷ all claims, automatic checker) — the verifier's effect.")
    c = agg.melt(id_vars=["approach_label"], value_vars=["llm_calls", "cost_usd", "latency_s"], var_name="metric", value_name="value")
    fig3 = px.bar(c, x="approach_label", y="value", facet_col="metric", facet_col_spacing=0.08, text=c["value"].map(lambda v: f"{v:.3g}"), title="Efficiency per resume: LLM calls, cost (USD) and model latency (s)")
    fig3.update_traces(textposition="outside", marker_color=viz.PALETTE[0])
    fig3.update_yaxes(matches=None, showticklabels=True)
    fig3.update_layout(xaxis_title="", showlegend=False)
    viz.save_fig(fig3, f"p3_efficiency_{tag}", "Problem 3: LLM calls, cost and latency per resume for each approach (same model and tools).")


if __name__ == "__main__":
    main()
