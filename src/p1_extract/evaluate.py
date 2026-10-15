"""Problem 1 evaluation — every approach is scored on exactly the same sentence ids.

Reads results/p1/preds/<approach>_<column>_seed<seed>_<split>.jsonl (one JSON per sentence:
sid, source, tokens, gold, pred) and writes
  results/p1/metrics.csv          long table: approach, column, seed, split, subset, metric, value
  results/p1/metrics_summary.csv  approach × column: strict F1 mean±std over seeds, loose F1, FP on all-O, by source
  results/p1/errors.csv           30 annotated error examples (15 per approach, champion vs challenger)
  reports/figures/p1_*.png|html   comparison charts
Usage: python -m src.p1_extract.evaluate --split test [--approaches tagger llm_haiku dict]"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from seqeval.metrics import f1_score, precision_score, recall_score

from ..utils.config import ROOT
from ..utils.io import read_jsonl, write_json, write_results
from ..utils.text import bio_to_spans

PRED_DIR = ROOT / "results" / "p1" / "preds"
FILE_RE = re.compile(r"^(?P<approach>.+?)_(?P<column>skill|knowledge|merged)_seed(?P<seed>\d+)_(?P<split>dev|test)\.jsonl$")


def list_runs(split: str) -> list[dict]:
    runs = []
    for p in sorted(PRED_DIR.glob(f"*_{split}.jsonl")):
        m = FILE_RE.match(p.name)
        if m and not p.name.startswith("smoketest"):
            runs.append({**m.groupdict(), "seed": int(m.group("seed")), "path": p})
    return runs


def _spans(tokens, tags):
    return [(t, s, e) for t, s, e, _ in bio_to_spans(tokens, tags)]


def loose_prf(golds: list[list[tuple]], preds: list[list[tuple]]) -> dict[str, float]:
    """Any-token-overlap matching (same type), one-to-one."""
    tp = fp = fn = 0
    for g, p in zip(golds, preds):
        used = set()
        for pt, ps, pe in p:
            hit = None
            for gi, (gt, gs, ge) in enumerate(g):
                if gi in used or gt != pt:
                    continue
                if ps < ge and gs < pe:
                    hit = gi
                    break
            if hit is None:
                fp += 1
            else:
                used.add(hit)
                tp += 1
        fn += len(g) - len(used)
    p_ = tp / (tp + fp) if tp + fp else 0.0
    r_ = tp / (tp + fn) if tp + fn else 0.0
    f_ = 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0
    return {"loose_precision": p_, "loose_recall": r_, "loose_f1": f_}


def score_rows(rows: list[dict]) -> dict[str, float]:
    y_true = [r["gold"] for r in rows]
    y_pred = [r["pred"] for r in rows]
    out = {"n_sentences": len(rows), "strict_precision": precision_score(y_true, y_pred), "strict_recall": recall_score(y_true, y_pred), "strict_f1": f1_score(y_true, y_pred)}
    golds = [_spans(r["tokens"], r["gold"]) for r in rows]
    preds = [_spans(r["tokens"], r["pred"]) for r in rows]
    out.update(loose_prf(golds, preds))
    out["n_gold_spans"] = sum(len(g) for g in golds)
    out["n_pred_spans"] = sum(len(p) for p in preds)
    all_o = [(g, p) for g, p in zip(golds, preds) if not g]
    out["n_all_O_sentences"] = len(all_o)
    out["fp_spans_per_100_all_O_sentences"] = 100 * sum(len(p) for _, p in all_o) / len(all_o) if all_o else 0.0
    out["share_all_O_sentences_with_fp"] = sum(1 for _, p in all_o if p) / len(all_o) if all_o else 0.0
    for src in ("tech", "house"):
        sub = [r for r in rows if r["source"] == src]
        if sub:
            out[f"strict_f1_{src}"] = f1_score([r["gold"] for r in sub], [r["pred"] for r in sub])
    return out


def error_examples(rows: list[dict], approach: str, column: str, n: int = 15, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    cands = []
    for r in rows:
        g, p = set(_spans(r["tokens"], r["gold"])), set(_spans(r["tokens"], r["pred"]))
        if g == p:
            continue
        kinds = []
        for pt, ps, pe in p - g:
            if any(gt == pt and ps < ge and gs < pe for gt, gs, ge in g):
                kinds.append("boundary")
            else:
                kinds.append("spurious")
        for gt, gs, ge in g - p:
            if not any(pt == gt and ps < ge and gs < pe for pt, ps, pe in p):
                kinds.append("missed")
        cands.append({"approach": approach, "column": column, "sid": r["sid"], "source": r["source"], "sentence": " ".join(r["tokens"]),
                      "gold_spans": " | ".join(" ".join(r["tokens"][s:e]) for _, s, e in sorted(g, key=lambda x: x[1])),
                      "pred_spans": " | ".join(" ".join(r["tokens"][s:e]) for _, s, e in sorted(p, key=lambda x: x[1])),
                      "error_types": ",".join(sorted(set(kinds)))})
    rng.shuffle(cands)
    # balance error types a bit
    by_type = defaultdict(list)
    for c in cands:
        by_type[c["error_types"].split(",")[0]].append(c)
    out, i = [], 0
    while len(out) < min(n, len(cands)):
        for k in sorted(by_type):
            if by_type[k] and len(out) < n:
                out.append(by_type[k].pop())
        i += 1
        if i > n:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--approaches", nargs="*", default=None)
    ap.add_argument("--champion", default="tagger")
    ap.add_argument("--challenger", default="llm_haiku")
    args = ap.parse_args()

    runs = list_runs(args.split)
    if args.approaches:
        runs = [r for r in runs if r["approach"] in args.approaches]
    if not runs:
        raise SystemExit("no prediction files found")
    data = {}
    for r in runs:
        rows = read_jsonl(str(r["path"].relative_to(ROOT)))
        data[(r["approach"], r["column"], r["seed"])] = {x["sid"]: x for x in rows}
    # common sentence ids across ALL runs of the same column (so approaches are compared on identical sentences)
    common = {}
    for col in {k[1] for k in data}:
        sets = [set(v) for k, v in data.items() if k[1] == col]
        common[col] = set.intersection(*sets)
    subset_note = {col: len(s) for col, s in common.items()}
    print("sentences in common per column:", subset_note)

    long_rows, summary_rows, errors = [], [], []
    for (approach, col, seed), by_sid in sorted(data.items()):
        rows = [by_sid[s] for s in sorted(common[col], key=lambda x: (x.split("-")[0], int(x.split("-")[1])))]
        m = score_rows(rows)
        meta_p = PRED_DIR / f"{approach}_{col}_seed{seed}_{args.split}.meta.json"
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        for k, v in m.items():
            long_rows.append({"approach": approach, "column": col, "seed": seed, "split": args.split, "n_common": len(rows), "metric": k, "value": v})
        for k in ("latency_ms_per_sentence", "cost_usd_per_1000_sentences", "hallucination_rate", "model"):
            if k in meta:
                long_rows.append({"approach": approach, "column": col, "seed": seed, "split": args.split, "n_common": len(rows), "metric": k, "value": meta[k]})
        summary_rows.append({"approach": approach, "column": col, "seed": seed, **{k: v for k, v in m.items()}, **{k: meta.get(k) for k in ("latency_ms_per_sentence", "cost_usd_per_1000_sentences", "hallucination_rate")}})
        if approach in (args.champion, args.challenger) and seed in (42, 0):
            errors += error_examples(rows, approach, col, n=8 if col != "merged" else 15)

    long_df = pd.DataFrame(long_rows)
    write_results(long_df, f"results/p1/metrics_{args.split}.csv", meta={"split": args.split, "common_sentences": subset_note})
    sm = pd.DataFrame(summary_rows)
    agg = sm.groupby(["approach", "column"]).agg(
        seeds=("seed", "nunique"), strict_f1_mean=("strict_f1", "mean"), strict_f1_std=("strict_f1", "std"),
        strict_precision=("strict_precision", "mean"), strict_recall=("strict_recall", "mean"), loose_f1=("loose_f1", "mean"),
        strict_f1_tech=("strict_f1_tech", "mean"), strict_f1_house=("strict_f1_house", "mean"),
        fp_spans_per_100_all_O=("fp_spans_per_100_all_O_sentences", "mean"), share_all_O_with_fp=("share_all_O_sentences_with_fp", "mean"),
        n_pred_spans=("n_pred_spans", "mean"), n_gold_spans=("n_gold_spans", "first"), n_sentences=("n_sentences", "first"),
        latency_ms=("latency_ms_per_sentence", "mean"), cost_usd_per_1000=("cost_usd_per_1000_sentences", "mean"), hallucination_rate=("hallucination_rate", "mean"),
    ).reset_index()
    agg["strict_f1_std"] = agg["strict_f1_std"].fillna(0.0)
    write_results(agg.round(4), f"results/p1/metrics_summary_{args.split}.csv", meta={"split": args.split, "common_sentences": subset_note})
    if errors:
        write_results(pd.DataFrame(errors), f"results/p1/errors_{args.split}.csv")
    print(agg.round(3).to_string())

    # ---- figures
    os.environ["SMC_SOURCE"] = "src/p1_extract/evaluate.py"
    import plotly.express as px
    from ..utils import viz

    viz.setup()
    plot = agg.copy()
    plot["approach_label"] = plot["approach"].map(lambda a: {"tagger": "Fine-tuned JobBERT tagger (champion)", "dict": "Dictionary lookup (baseline)"}.get(a, a.replace("llm_", "Few-shot LLM: ")))
    long_f = plot.melt(id_vars=["approach_label", "column"], value_vars=["strict_f1_mean", "loose_f1"], var_name="metric", value_name="f1")
    long_f["metric"] = long_f["metric"].map({"strict_f1_mean": "strict span F1", "loose_f1": "loose (overlap) F1"})
    fig = px.bar(long_f, x="column", y="f1", color="approach_label", facet_col="metric", barmode="group", text=long_f["f1"].map(lambda v: f"{v:.2f}"),
                 title=f"Problem 1 — span F1 on the SkillSpan {args.split} split (same {min(subset_note.values())} sentences for every approach)")
    fig.update_traces(textposition="outside")
    fig.update_layout(yaxis_range=[0, 1], xaxis_title="", legend_title_text="")
    viz.save_fig(fig, f"p1_f1_comparison_{args.split}", f"Problem 1: strict and loose span F1 per approach and span type on the SkillSpan {args.split} split (identical sentence set for all approaches; tagger = mean over seeds).")
    fig2 = px.bar(plot, x="column", y="fp_spans_per_100_all_O", color="approach_label", barmode="group", text=plot["fp_spans_per_100_all_O"].map(lambda v: f"{v:.1f}"),
                  title="Over-extraction: predicted spans per 100 sentences that contain no gold span")
    fig2.update_traces(textposition="outside")
    fig2.update_layout(xaxis_title="", yaxis_title="false-positive spans / 100 all-O sentences", legend_title_text="")
    viz.save_fig(fig2, f"p1_false_positives_{args.split}", "Problem 1: over-extraction on sentences without any gold span (headers, benefits) — how often each approach invents a span where annotators found none.")
    src_long = plot.melt(id_vars=["approach_label", "column"], value_vars=["strict_f1_tech", "strict_f1_house"], var_name="source", value_name="f1")
    src_long["source"] = src_long["source"].str.replace("strict_f1_", "")
    fig3 = px.bar(src_long, x="source", y="f1", color="approach_label", facet_col="column", barmode="group", text=src_long["f1"].map(lambda v: f"{v:.2f}"),
                  title="Robustness across sources: strict F1 on tech vs house sentences")
    fig3.update_traces(textposition="outside")
    fig3.update_layout(yaxis_range=[0, 1], legend_title_text="")
    viz.save_fig(fig3, f"p1_f1_by_source_{args.split}", "Problem 1: strict span F1 split by posting source (tech vs house) — the domain gap each approach suffers.")


if __name__ == "__main__":
    main()
