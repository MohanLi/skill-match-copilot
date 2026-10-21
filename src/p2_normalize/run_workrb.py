"""Run WorkRB skill-extraction ranking tasks for the baselines and our models; collect one table.

Usage:
  python -m src.p2_normalize.run_workrb --split val  --models bm25 minilm context ours
  python -m src.p2_normalize.run_workrb --split val  --models rtl --k 10 30 50        # tune k
  python -m src.p2_normalize.run_workrb --split test --models bm25 minilm context ours rtl rtl_context --k 50
Writes results/p2/workrb_<split>.csv (model, task, map, rp@10, mrr, n_queries, eval_time_s, esco_source,
llm_cost_usd, llm_calls) and reports/figures/p2_workrb_<split>.png|html"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time

import pandas as pd

from ..data import esco as esco_data
from ..utils.config import ROOT, models_cfg
from ..utils.io import write_results
from .workrb_tasks import dataset_sizes, make_tasks

logging.getLogger("workrb").setLevel(logging.WARNING)

LABELS = {
    "bm25": "BM25 (lexical baseline)",
    "minilm": "all-MiniLM-L6-v2 (zero-shot bi-encoder)",
    "context": "ConTeXT-match (TechWolf, published model; champion)",
    "ours": "Fine-tuned MiniLM on synthetic ESCO pairs (in-house)",
    "rtl": "Retrieve-then-LLM on fine-tuned MiniLM",
    "rtl_context": "Retrieve-then-LLM on ConTeXT (challenger)",
}


def build_model(key: str, k: int = 30):
    cfg = models_cfg()["p2_normalize"]
    if key == "bm25":
        from workrb.models import BM25Model

        return BM25Model()
    if key == "minilm":
        from .models import NamedBiEncoder

        return NamedBiEncoder(str(ROOT / cfg["base_encoder"]), LABELS["minilm"], max_seq_length=cfg["max_seq_length"])
    if key == "context":
        from workrb.models import ConTeXTMatchModel

        path = ROOT / cfg["context_model"]
        try:
            return ConTeXTMatchModel(model_name=str(path)) if path.exists() else ConTeXTMatchModel()
        except TypeError:
            return ConTeXTMatchModel()
    if key == "ours":
        from .models import NamedBiEncoder

        return NamedBiEncoder(str(ROOT / "models" / "p2_biencoder"), LABELS["ours"], max_seq_length=cfg["max_seq_length"])
    if key == "rtl":
        from .models import RetrieveThenLLM

        enc = ROOT / "models" / "p2_biencoder"
        if not enc.exists():
            enc = ROOT / cfg["base_encoder"]
        return RetrieveThenLLM(str(enc), f"{LABELS['rtl']} k={k}", k=k, max_seq_length=cfg["max_seq_length"])
    if key == "rtl_context":
        from .models import RetrieveThenLLM

        return RetrieveThenLLM(str(ROOT / cfg["context_model"]), f"{LABELS['rtl_context']} k={k}", k=k, max_seq_length=cfg["max_seq_length"])
    raise ValueError(key)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--models", nargs="+", default=["bm25", "minilm", "context", "ours", "rtl"])
    ap.add_argument("--k", nargs="*", type=int, default=[30])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    import workrb

    esco_info = esco_data.ensure_workrb_esco(models_cfg()["p2_normalize"]["esco_version"], "en")
    tasks = make_tasks(args.split)
    sizes = dataset_sizes(tasks)
    print("tasks:", sizes, "| ESCO:", esco_info["esco_source"])
    rows = []
    out_csv = f"results/p2/workrb_{args.split}.csv"
    prev = pd.read_csv(ROOT / out_csv) if (ROOT / out_csv).exists() and not args.force else pd.DataFrame()
    for key in args.models:
        for k in (args.k if key in ("rtl", "rtl_context") else [None]):
            model = build_model(key, k or 30)
            t0 = time.time()
            folder = ROOT / "results" / "p2" / "workrb_runs" / args.split / (f"{key}_k{k}" if k else key)
            res = workrb.evaluate(model, tasks, output_folder=str(folder), force_restart=True)
            llm_cost = sum(c.get("cost_usd", 0.0) for c in getattr(model, "calls", []))
            llm_calls = len(getattr(model, "calls", []))
            if hasattr(model, "dump_calls"):
                model.dump_calls(f"results/p2/{key}_calls_{args.split}_k{k}.jsonl")
            for tname, tr in res.task_results.items():
                for did, mr in tr.datasetid_results.items():
                    label = LABELS[key] + (f" k={k}" if k else "")
                    rows.append({"model_key": key, "k": k, "model": label, "task": tname, "split": args.split, **{m: round(float(v), 4) for m, v in mr.metrics_dict.items()},
                                 "n_queries": sizes.get(tname), "eval_time_s": round(mr.evaluation_time, 1), "esco_source": esco_info["esco_source"],
                                 "llm_cost_usd": round(llm_cost, 4), "llm_calls": llm_calls, "run_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            print(f"{label}: {[(r['task'], r['map']) for r in rows if r['model'] == label]}  ({time.time()-t0:.0f}s, LLM ${llm_cost:.3f})")
    df = pd.DataFrame(rows)
    if len(prev):
        prev = prev[~prev.set_index(["model_key", "k"]).index.isin(df.set_index(["model_key", "k"]).index)]
        df = pd.concat([prev, df], ignore_index=True)
    write_results(df, out_csv, meta={"split": args.split, "esco_source": esco_info["esco_source"], "tasks": sizes})
    print(df.assign(k=df["k"].fillna(0)).pivot_table(index=["model_key", "k", "model"], columns="task", values="map").round(3).to_string())

    os.environ["SMC_SOURCE"] = "src/p2_normalize/run_workrb.py"
    import plotly.express as px

    from ..utils import viz

    viz.setup()
    plot = df.copy()
    plot["task"] = plot["task"].str.replace("Skill Extraction ", "")
    for metric in ("map", "rp@10", "mrr"):
        fig = px.bar(plot, x="task", y=metric, color="model", barmode="group", text=plot[metric].map(lambda v: f"{v:.2f}"),
                     title=f"Problem 2 — WorkRB {metric.upper()} on the {args.split} split (ESCO 1.1.0, {'official' if esco_info['esco_source']=='esco_csv' else 'stand-in'} vocabulary)")
        fig.update_traces(textposition="outside")
        fig.update_layout(yaxis_range=[0, 1], xaxis_title="", legend_title_text="")
        viz.save_fig(fig, f"p2_workrb_{metric.replace('@','')}_{args.split}", f"Problem 2: WorkRB {metric.upper()} per task on the {args.split} split — lexical and dense baselines, the in-house fine-tuned MiniLM, the published ConTeXT encoder (champion) and retrieve-then-LLM re-ranking (challenger).")


if __name__ == "__main__":
    main()
