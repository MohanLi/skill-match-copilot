""" "Not in ESCO" detection (Problem 2, our own evaluation on the benchmark's raw span annotations).

For every annotated span the system outputs an ESCO skill or NOT_IN_ESCO.
  champion   : bi-encoder top-1 over the ESCO vocabulary, with a cosine threshold tuned on tech_validation
  challenger : LLM (prompts/p2_nil_v1.yaml) sees the top-k candidates and answers with one label or null
Scores: accuracy on ESCO-mapped phrases (top-1 label == gold), precision/recall/F1 of the NOT_IN_ESCO decision
(positive class = LABEL NOT PRESENT or UNDERSPECIFIED), and overall accuracy of the joint decision.

Usage: python -m src.p2_normalize.nil_eval --encoder models/p2_biencoder [--llm] [--k 10] [--subsets tech house]
Writes results/p2/nil_threshold_sweep.csv, results/p2/nil.csv, results/p2/nil_examples.csv and figures."""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch

from ..data import esco, esco_benchmark as bm
from ..utils.config import ROOT, models_cfg
from ..utils.io import write_results
from ..utils.llm import LLMClient, load_prompt

NIL = "NOT_IN_ESCO"


def prf(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"nil_precision": p, "nil_recall": r, "nil_f1": 2 * p * r / (p + r) if p + r else 0.0}


def score(df: pd.DataFrame, pred_col: str) -> dict[str, float]:
    is_nil_true = df["label"].isin(bm.NIL_LABELS).to_numpy().astype(int)
    is_nil_pred = (df[pred_col] == NIL).to_numpy().astype(int)
    mapped = df[~df["label"].isin(bm.NIL_LABELS)]
    acc_mapped = float((mapped[pred_col] == mapped["label"]).mean()) if len(mapped) else 0.0
    joint = float(((df[pred_col] == df["label"]) | (df["label"].isin(bm.NIL_LABELS) & (df[pred_col] == NIL))).mean())
    return {"n": len(df), "n_nil_true": int(is_nil_true.sum()), "acc_on_mapped_phrases": acc_mapped, "joint_accuracy": joint, **prf(is_nil_true, is_nil_pred)}


def main() -> None:
    cfg = models_cfg()["p2_normalize"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default=None, help="encoder path (default: the champion from configs/models.yaml, see normalizer.resolve_encoder)")
    ap.add_argument("--tag", default="", help="suffix for the output files, e.g. --tag p2_biencoder → results/p2/nil_p2_biencoder.csv")
    ap.add_argument("--llm", action="store_true", help="also run the LLM challenger")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--subsets", nargs="*", default=["tech", "house"])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    from .normalizer import resolve_encoder

    enc_path = str(resolve_encoder(args.encoder))
    suffix = f"_{args.tag}" if args.tag else ""
    print(f"encoder: {enc_path}")
    model = SentenceTransformer(enc_path, device="cpu")
    model.max_seq_length = cfg["max_seq_length"]
    vocab = esco.vocabulary()
    V = model.encode(vocab, convert_to_tensor=True, normalize_embeddings=True, batch_size=256, show_progress_bar=False)

    def run_encoder(df: pd.DataFrame) -> pd.DataFrame:
        Q = model.encode(df["span"].tolist(), convert_to_tensor=True, normalize_embeddings=True, batch_size=256, show_progress_bar=False)
        S = Q @ V.T
        top = S.topk(args.k, dim=1)
        df = df.copy()
        df["top1_label"] = [vocab[i] for i in top.indices[:, 0].tolist()]
        df["top1_score"] = top.values[:, 0].tolist()
        df["topk_labels"] = [[vocab[i] for i in row] for row in top.indices.tolist()]
        df["gold_in_topk"] = [g in c for g, c in zip(df["label"], df["topk_labels"])]
        return df

    val = run_encoder(bm.load("tech", "validation"))
    tests = {s: run_encoder(bm.load(s, "test")) for s in args.subsets}
    if args.limit:
        tests = {s: d.head(args.limit) for s, d in tests.items()}

    # ---- champion: threshold sweep on validation
    grid = np.arange(cfg["nil_threshold_grid_start"], cfg["nil_threshold_grid_stop"] + 1e-9, cfg["nil_threshold_grid_step"])
    sweep = []
    for th in grid:
        val["pred"] = np.where(val["top1_score"] < th, NIL, val["top1_label"])
        sweep.append({"threshold": round(float(th), 3), **score(val, "pred")})
    sweep_df = pd.DataFrame(sweep)
    best = sweep_df.sort_values(["joint_accuracy", "nil_f1"], ascending=False).iloc[0]
    th_best = float(best["threshold"])
    write_results(sweep_df.round(4), f"results/p2/nil_threshold_sweep{suffix}.csv", meta={"encoder": enc_path, "tuned_on": "tech_validation", "best_threshold": th_best})
    print(f"best threshold on tech_validation: {th_best:.2f} (joint acc {best['joint_accuracy']:.3f}, NIL F1 {best['nil_f1']:.3f})")

    rows, examples = [], []
    for s, df in tests.items():
        df["pred_encoder"] = np.where(df["top1_score"] < th_best, NIL, df["top1_label"])
        rows.append({"approach": "bi-encoder + threshold (champion)", "subset": s, "threshold": th_best, "encoder": os.path.basename(enc_path), **score(df, "pred_encoder"),
                     "recall_at_k_on_mapped": float(df[~df["label"].isin(bm.NIL_LABELS)]["gold_in_topk"].mean())})
        rows.append({"approach": "bi-encoder top-1, no NIL (reference)", "subset": s, "threshold": None, "encoder": os.path.basename(enc_path), **score(df, "top1_label")})

    # ---- challenger: LLM over top-k candidates
    if args.llm:
        client = LLMClient(task="p2_nil_llm")
        prompt = load_prompt("p2_nil_v1")

        def ask(row) -> tuple[str, dict]:
            cands = row["topk_labels"]
            cand_text = "\n".join(f"{i+1}. {c}" for i, c in enumerate(cands))
            try:
                obj, res = client.complete_json(prompt, model="bulk", max_tokens=200, phrase=row["span"], sentence=row["sentence"], candidates=cand_text)
                lab = obj.get("esco_label")
                pred = lab if isinstance(lab, str) and lab in cands else NIL
                return pred, {"cost_usd": res.cost_usd, "latency_ms": res.latency_ms, "confidence": obj.get("confidence"), "reason": obj.get("reason")}
            except Exception as e:  # noqa: BLE001
                return NIL, {"cost_usd": 0.0, "error": str(e)[:100]}

        for s, df in tests.items():
            with ThreadPoolExecutor(max_workers=4) as ex:
                outs = list(ex.map(ask, [r for _, r in df.iterrows()]))
            df["pred_llm"] = [o[0] for o in outs]
            df["llm_reason"] = [o[1].get("reason") for o in outs]
            cost = sum(o[1].get("cost_usd", 0.0) for o in outs)
            lat = [o[1]["latency_ms"] for o in outs if o[1].get("latency_ms")]
            rows.append({"approach": f"retrieve top-{args.k} + LLM (challenger)", "subset": s, "threshold": None, "encoder": os.path.basename(enc_path), **score(df, "pred_llm"),
                         "llm_cost_usd": round(cost, 4), "llm_latency_ms_median": float(np.median(lat)) if lat else None, "recall_at_k_on_mapped": float(df[~df["label"].isin(bm.NIL_LABELS)]["gold_in_topk"].mean())})
            dis = df[df["pred_llm"] != df["pred_encoder"]].head(20)
            for _, r in dis.iterrows():
                examples.append({"subset": s, "sentence": r["sentence"], "span": r["span"], "gold": r["label"], "encoder_pred": r["pred_encoder"], "encoder_score": round(float(r["top1_score"]), 3), "llm_pred": r["pred_llm"], "llm_reason": r["llm_reason"]})

    res = pd.DataFrame(rows)
    write_results(res.round(4), f"results/p2/nil{suffix}.csv", meta={"encoder": enc_path, "threshold": th_best, "k": args.k, "tuned_on": "tech_validation"})
    if examples:
        write_results(pd.DataFrame(examples), f"results/p2/nil_disagreements{suffix}.csv")
    print(res.round(3).to_string())

    os.environ["SMC_SOURCE"] = "src/p2_normalize/nil_eval.py"
    import plotly.express as px
    import plotly.graph_objects as go

    from ..utils import viz

    viz.setup()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sweep_df["threshold"], y=sweep_df["joint_accuracy"], mode="lines", name="joint accuracy"))
    fig.add_trace(go.Scatter(x=sweep_df["threshold"], y=sweep_df["nil_f1"], mode="lines", name="F1 of the 'not in ESCO' decision"))
    fig.add_trace(go.Scatter(x=sweep_df["threshold"], y=sweep_df["acc_on_mapped_phrases"], mode="lines", name="accuracy on mapped phrases"))
    fig.add_vline(x=th_best, line_dash="dot", line_color=viz.INK["muted"], annotation_text=f"chosen {th_best:.2f}", annotation_position="top left")
    enc_name = os.path.basename(enc_path)
    fig.update_layout(title=f"'Not in ESCO' threshold sweep on tech_validation — cosine of {enc_name} to the best ESCO skill", xaxis_title="cosine threshold", yaxis_title="score", yaxis_range=[0, 1])
    viz.save_fig(fig, f"p2_nil_threshold_sweep{suffix}", f"Tuning the 'not in ESCO' cosine threshold of the {enc_name} encoder on the TECH validation spans: joint accuracy, NIL-decision F1 and accuracy on mapped phrases vs threshold.")
    m = res.melt(id_vars=["approach", "subset"], value_vars=["joint_accuracy", "nil_f1", "acc_on_mapped_phrases"], var_name="metric", value_name="value")
    m["metric"] = m["metric"].map({"joint_accuracy": "joint accuracy", "nil_f1": "F1 of the 'not in ESCO' decision", "acc_on_mapped_phrases": "accuracy on mapped phrases"})
    fig2 = px.bar(m, x="metric", y="value", color="approach", facet_col="subset", barmode="group", text=m["value"].map(lambda v: f"{v:.2f}"),
                  title=f"'Not in ESCO' detection on the test spans — {enc_name} encoder, threshold {th_best:.2f}")
    fig2.update_traces(textposition="outside")
    fig2.update_layout(yaxis_range=[0, 1], legend_title_text="", xaxis_title="", yaxis_title="score")
    fig2.update_xaxes(title_text="")
    viz.save_fig(fig2, f"p2_nil_results{suffix}", f"'Not in ESCO' detection on TECH and HOUSE test spans with the {enc_name} encoder: joint accuracy, F1 of the NIL decision and accuracy on mapped phrases for the thresholded encoder (champion) vs retrieve-then-LLM over its top-10 (challenger).")


if __name__ == "__main__":
    main()
