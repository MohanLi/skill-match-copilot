"""Gold-standard fit labels (decision #6): two independent LLM raters (Opus primary, Sonnet second),
weighted Cohen's kappa, adjudication of disagreements by the primary rater with written rationale.

Usage: python -m src.p3_agent.label_pairs [--workers 4] [--limit N]
Writes data/labels/labels_opus.csv, labels_sonnet.csv, final.csv, kappa.json (all committed; CV text is not)."""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from ..utils.config import ROOT, models_cfg
from ..utils.io import write_json
from ..utils.llm import LLMClient, load_prompt


def rate(client: LLMClient, prompt, model_role: str, row: dict) -> dict:
    try:
        obj, res = client.complete_json(prompt, model=model_role, max_tokens=1000, cv_text=row["cv_text"], job_text=row["job_text"])
        lab = int(obj.get("label")) if obj.get("label") in (0, 1, 2, "0", "1", "2") else None
        return {"pair_id": row["pair_id"], "label": lab, "rationale": str(obj.get("rationale", ""))[:600], "role_family_match": obj.get("role_family_match"),
                "present": json.dumps(obj.get("present", []))[:500], "missing": json.dumps(obj.get("missing", []))[:500], "seniority_gap_levels": obj.get("seniority_gap_levels"),
                "model": res.model, "cost_usd": res.cost_usd, "latency_ms": res.latency_ms, "cached": res.cached}
    except Exception as e:  # noqa: BLE001
        return {"pair_id": row["pair_id"], "label": None, "rationale": f"ERROR {str(e)[:150]}", "model": model_role, "cost_usd": 0.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    cfg = models_cfg()["llm"]
    pairs = pd.read_csv(ROOT / "data" / "labels" / "pairs_to_label.csv")
    texts = pd.read_parquet(ROOT / "data" / "processed" / "p3_pairs_texts.parquet")
    df = pairs.merge(texts, on=["pair_id", "cv_id", "job_id"])
    if args.limit:
        df = df.head(args.limit)
    rows = df.to_dict("records")
    prompt = load_prompt("label_fit_v1")
    out = {}
    for role, fname in [("label_primary", "labels_opus.csv"), ("label_second", "labels_sonnet.csv")]:
        client = LLMClient(task=f"p3_gold_{role}")
        print(f"rating {len(rows)} pairs with {client.resolve_model(role)} …", flush=True)
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            res = list(ex.map(lambda r: rate(client, prompt, role, r), rows))
        rdf = pd.DataFrame(res)
        rdf.to_csv(ROOT / "data" / "labels" / fname, index=False)
        out[role] = rdf
        print(f"  done in {(time.time()-t0)/60:.1f} min, cost ${rdf['cost_usd'].sum():.3f}, label counts {rdf['label'].value_counts(dropna=False).to_dict()}")

    a, b = out["label_primary"].set_index("pair_id"), out["label_second"].set_index("pair_id")
    both = a.join(b, lsuffix="_a", rsuffix="_b").dropna(subset=["label_a", "label_b"])
    la, lb = both["label_a"].astype(int), both["label_b"].astype(int)
    kappa = {"n_pairs": int(len(both)), "agreement_rate": float((la == lb).mean()),
             "cohen_kappa_unweighted": float(cohen_kappa_score(la, lb)), "cohen_kappa_linear": float(cohen_kappa_score(la, lb, weights="linear")),
             "cohen_kappa_quadratic": float(cohen_kappa_score(la, lb, weights="quadratic")),
             "confusion_a_rows_b_cols": pd.crosstab(la, lb).reindex(index=[0, 1, 2], columns=[0, 1, 2], fill_value=0).values.tolist(),
             "label_dist_a": la.value_counts().sort_index().to_dict(), "label_dist_b": lb.value_counts().sort_index().to_dict(),
             "rater_a": str(a["model"].iloc[0]), "rater_b": str(b["model"].iloc[0])}
    print(json.dumps(kappa, indent=1))

    # adjudication
    adj_prompt = load_prompt("label_adjudicate_v1")
    client = LLMClient(task="p3_gold_adjudicate")
    tx = texts.set_index("pair_id")
    disagree = both[la != lb]

    def adjudicate(pid: str) -> dict:
        r = both.loc[pid]
        try:
            obj, res = client.complete_json(adj_prompt, model="label_primary", max_tokens=500, cv_text=tx.loc[pid, "cv_text"], job_text=tx.loc[pid, "job_text"],
                                            label_a=int(r["label_a"]), rationale_a=r["rationale_a"], label_b=int(r["label_b"]), rationale_b=r["rationale_b"])
            return {"pair_id": pid, "label": int(obj.get("label")), "rationale": str(obj.get("rationale", ""))[:600], "agree_with": obj.get("agree_with"), "cost_usd": res.cost_usd}
        except Exception as e:  # noqa: BLE001
            return {"pair_id": pid, "label": int(r["label_a"]), "rationale": f"adjudication failed ({str(e)[:80]}); primary rater kept", "agree_with": "A", "cost_usd": 0.0}

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        adj = list(ex.map(adjudicate, disagree.index.tolist()))
    adj_df = pd.DataFrame(adj).set_index("pair_id") if adj else pd.DataFrame(columns=["label", "rationale", "agree_with", "cost_usd"])
    final_rows = []
    for pid, r in both.iterrows():
        if pid in adj_df.index:
            final_rows.append({"pair_id": pid, "label": int(adj_df.loc[pid, "label"]), "source": "adjudicated", "rationale": adj_df.loc[pid, "rationale"], "label_a": int(r["label_a"]), "label_b": int(r["label_b"])})
        else:
            final_rows.append({"pair_id": pid, "label": int(r["label_a"]), "source": "agreed", "rationale": r["rationale_a"], "label_a": int(r["label_a"]), "label_b": int(r["label_b"])})
    final = pd.DataFrame(final_rows).merge(pairs, on="pair_id")
    final.to_csv(ROOT / "data" / "labels" / "final.csv", index=False)
    kappa["adjudicated"] = int(len(adj_df))
    kappa["adjudication_agree_with"] = adj_df["agree_with"].value_counts().to_dict() if len(adj_df) else {}
    kappa["final_label_dist"] = final["label"].value_counts().sort_index().to_dict()
    kappa["final_label_by_relation"] = final.groupby("relation")["label"].mean().round(3).to_dict()
    kappa["cost_usd_total"] = round(float(out["label_primary"]["cost_usd"].sum() + out["label_second"]["cost_usd"].sum() + (adj_df["cost_usd"].sum() if len(adj_df) else 0.0)), 4)
    write_json(kappa, "data/labels/kappa.json")
    print(f"final labels: {kappa['final_label_dist']}  adjudicated {kappa['adjudicated']}  total cost ${kappa['cost_usd_total']}")


if __name__ == "__main__":
    main()
