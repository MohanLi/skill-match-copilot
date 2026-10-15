"""Sample the Problem 3 evaluation pairs: 30 Djinni CVs (3 per category × 10 categories) × a fixed pool
of 10 indexed Djinni jobs each (4 same category, 3 related, 3 random) = 300 pairs, seed 42.

Requires the djinni_jobs index (python -m src.index.build_job_index --pool djinni).
Writes data/labels/pairs_to_label.csv (ids only; committed), data/processed/p3_pairs_texts.parquet
(CV + job texts, PII-scrubbed; not committed) and data/processed/labeling_sheet.xlsx for human raters."""
from __future__ import annotations

import argparse
import json
import random

import numpy as np
import pandas as pd

from ..data import djinni
from ..index.store import JobIndex
from ..utils.config import ROOT, data_cfg, models_cfg
from ..utils.io import write_json


def related_categories(index: JobIndex, jobs: pd.DataFrame, cats: list[str], n_related: int = 2) -> dict[str, list[str]]:
    cents = {}
    for c in cats:
        ids = jobs[jobs["category"] == c]["job_id"].tolist()[:400]
        res = index.col.get(ids=ids, include=["embeddings"])
        E = np.array(res["embeddings"])
        v = E.mean(axis=0)
        cents[c] = v / (np.linalg.norm(v) + 1e-9)
    out = {}
    for c in cats:
        sims = sorted(((float(cents[c] @ cents[o]), o) for o in cats if o != c), reverse=True)
        out[c] = [o for _, o in sims[:n_related]]
    return out


def main() -> None:
    cfg = models_cfg()["p3_agent"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=models_cfg()["seed"])
    ap.add_argument("--n-cvs", type=int, default=cfg["n_cvs"])
    ap.add_argument("--jobs-per-cv", type=int, default=cfg["jobs_per_cv"])
    args = ap.parse_args()
    rng = random.Random(args.seed)

    jobs = pd.read_parquet(ROOT / "data" / "processed" / "djinni_jobs.parquet")
    cats = jobs["category"].value_counts().head(cfg["n_categories"]).index.tolist()
    index = JobIndex("djinni_jobs")
    related = related_categories(index, jobs, cats)
    print("categories:", cats)
    print("related:", related)

    cvs, _ = djinni.load_cvs()
    min_chars = data_cfg()["datasets"]["djinni_cvs"]["min_cv_chars"]
    cvs = cvs[(cvs["n_chars"] >= min_chars) & (cvs["category"].isin(cats))]
    per_cat = args.n_cvs // len(cats)
    chosen = pd.concat([cvs[cvs["category"] == c].sample(n=per_cat, random_state=args.seed) for c in cats]).reset_index(drop=True)

    job_texts = {r["job_id"]: r for _, r in jobs.iterrows()}
    pairs, texts = [], []
    for ci, cv in chosen.iterrows():
        c = cv["category"]
        same = jobs[jobs["category"] == c]["job_id"].tolist()
        rel = jobs[jobs["category"].isin(related[c])]["job_id"].tolist()
        other = jobs[~jobs["category"].isin([c] + related[c])]["job_id"].tolist()
        r = random.Random(f"{args.seed}-{cv['cv_id']}")
        pool = [(j, "same") for j in r.sample(same, 4)] + [(j, "related") for j in r.sample(rel, 3)] + [(j, "random") for j in r.sample(other, 3)]
        r.shuffle(pool)
        full = index.get_many([j for j, _ in pool])
        full_by = {j["job_id"]: j for j in full}
        for k, (jid, relation) in enumerate(pool):
            pid = f"{cv['cv_id']}__{jid}"
            pairs.append({"pair_id": pid, "cv_id": cv["cv_id"], "cv_category": c, "job_id": jid, "job_category": job_texts[jid]["category"], "relation": relation, "pool_position": k})
            texts.append({"pair_id": pid, "cv_id": cv["cv_id"], "job_id": jid, "cv_text": cv["text"][:6000], "job_text": full_by[jid]["text"][:6000], "job_title": full_by[jid].get("title", "")})
    pdf, tdf = pd.DataFrame(pairs), pd.DataFrame(texts)
    (ROOT / "data" / "labels").mkdir(parents=True, exist_ok=True)
    pdf.to_csv(ROOT / "data" / "labels" / "pairs_to_label.csv", index=False)
    tdf.to_parquet(ROOT / "data" / "processed" / "p3_pairs_texts.parquet", index=False)
    sheet = tdf.merge(pdf[["pair_id", "relation"]], on="pair_id")
    sheet = sheet[["pair_id", "cv_id", "job_id", "job_title", "cv_text", "job_text"]].assign(label="", notes="")
    try:
        sheet.to_excel(ROOT / "data" / "processed" / "labeling_sheet.xlsx", index=False)
    except Exception as e:  # noqa: BLE001
        print("xlsx not written:", e)
    write_json({"seed": args.seed, "categories": cats, "related": related, "n_cvs": len(chosen), "n_pairs": len(pdf), "cv_ids": chosen["cv_id"].tolist(),
                "cv_chars": {"min": int(chosen["n_chars"].min()), "median": float(chosen["n_chars"].median()), "max": int(chosen["n_chars"].max())}},
               "data/labels/pairs_meta.json")
    print(f"{len(chosen)} CVs, {len(pdf)} pairs; relation counts: {pdf['relation'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
