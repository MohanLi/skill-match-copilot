"""Build the job indexes: clean → extract (P1 champion) → normalize (P2 champion) → embed (all-MiniLM-L6-v2, 256 tokens) → Chroma.

  python -m src.index.build_job_index --pool djinni --sample 6000      # evaluation pool (stratified by category)
  python -m src.index.build_job_index --pool live                      # this week's data/AI/ML postings
Idempotent: already-indexed job ids are skipped. Writes data/processed/<pool>_jobs.parquet (ids, metadata,
skills) and results/index/<pool>_build.json (counts, timing, spot-check sample)."""
from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np
import pandas as pd

from ..data import djinni, live_jobs
from ..p1_extract.predict import SkillExtractor
from ..p2_normalize.normalizer import get_normalizer
from ..utils.config import ROOT, models_cfg
from ..utils.io import write_json
from .store import JobIndex


def select_djinni_sample(jobs: pd.DataFrame, n: int, seed: int, n_categories: int) -> pd.DataFrame:
    jobs = jobs[(jobs["n_chars"] >= 300) & jobs["category"].notna() & (jobs["category"].astype(str).str.strip() != "")]
    jobs = jobs.drop_duplicates(["title", "description"])
    top_cats = jobs["category"].value_counts().head(n_categories).index.tolist()
    per_cat = n // n_categories
    parts = [jobs[jobs["category"] == c].sample(n=min(per_cat, (jobs["category"] == c).sum()), random_state=seed) for c in top_cats]
    out = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def export_parquet(index: JobIndex, name: str) -> pd.DataFrame:
    """data/processed/<name>.parquet = every indexed posting's metadata (no text): ids, title, category, skills, counts."""
    ids = index.all_ids()
    rows = []
    for b in range(0, len(ids), 1000):
        res = index.col.get(ids=ids[b : b + 1000], include=["metadatas"])
        rows += [{"job_id": i, **(m or {})} for i, m in zip(res["ids"], res["metadatas"])]
    out = pd.DataFrame(rows)
    proc = ROOT / "data" / "processed" / f"{name}.parquet"
    proc.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(proc, index=False)
    return out


def main() -> None:
    cfg = models_cfg()
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", choices=["djinni", "live"], required=True)
    ap.add_argument("--sample", type=int, default=cfg["index"]["djinni_sample_size"])
    ap.add_argument("--seed", type=int, default=cfg["seed"])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--live-sample", type=int, default=None, help="live pool: keep the N most recent postings (default configs/models.yaml index.live_sample_size)")
    ap.add_argument("--prune", action="store_true", help="remove indexed postings that are no longer in the selection (e.g. after the selection rule changed)")
    args = ap.parse_args()

    t0 = time.time()
    if args.pool == "djinni":
        jobs, info = djinni.load_jobs()
        jobs = select_djinni_sample(jobs, args.sample, args.seed, cfg["p3_agent"]["n_categories"])
        jobs["source"] = "djinni"
        jobs["url"] = ""
    else:
        jobs, info = live_jobs.load_filtered()
        jobs = jobs[jobs["n_chars"] >= 200].copy()
        jobs = jobs.drop_duplicates(["title", "description"])
        live_n = args.live_sample if args.live_sample is not None else cfg["index"].get("live_sample_size", 4000)
        if live_n and len(jobs) > live_n:  # most recent postings first (the demo should show this week's jobs)
            jobs["_posted"] = live_jobs.parse_dates(jobs["posted_at"])
            jobs = jobs.sort_values("_posted", ascending=False, na_position="last").head(live_n).drop(columns=["_posted"])
        jobs["category"] = jobs["department"].fillna("").astype(str)
        jobs["company"] = jobs["company"].fillna("").astype(str)
        jobs["source"] = "live"
        jobs["url"] = jobs["apply_url"].fillna("").astype(str)
        jobs["exp_years"] = None
        jobs["english_level"] = None
        jobs["published"] = jobs["posted_at"].astype(str)
    if args.limit:
        jobs = jobs.head(args.limit)
    name = f"{args.pool}_jobs"
    index = JobIndex(name)
    done = set(index.all_ids())
    if args.prune:
        stale = sorted(done - set(jobs["job_id"]))
        if stale:
            for b in range(0, len(stale), 500):
                index.col.delete(ids=stale[b : b + 500])
            print(f"{name}: pruned {len(stale)} postings outside the current selection")
            done = set(index.all_ids())
    todo = jobs[~jobs["job_id"].isin(done)].reset_index(drop=True)
    print(f"{name}: {len(jobs)} postings selected, {len(done)} already indexed, {len(todo)} to do")

    extractor = SkillExtractor()
    normalizer = get_normalizer()
    rows_out = []
    for b in range(0, len(todo), args.batch):
        chunk = todo.iloc[b : b + args.batch]
        docs = []
        for _, j in chunk.iterrows():
            ex = extractor.extract(j["text"][:6000])
            raw = ex["skills"] + ex["knowledge"]
            nm = normalizer.normalize(raw[:80])
            doc = {
                "job_id": j["job_id"], "text": j["text"][:6000], "title": j["title"][:200], "company": str(j.get("company") or "")[:100],
                "category": str(j.get("category") or "")[:80], "exp_years": str(j.get("exp_years") or ""), "english_level": str(j.get("english_level") or ""),
                "published": str(j.get("published") or "")[:40], "source": j["source"], "url": str(j.get("url") or "")[:300],
                "esco_skills": "|".join(nm["esco_skills"])[:4000], "raw_skills": "|".join(dict.fromkeys(raw))[:4000], "n_chars": int(j["n_chars"]),
                "n_skill_phrases": len(ex["skills"]), "n_knowledge_phrases": len(ex["knowledge"]), "n_esco": len(nm["esco_skills"]), "n_not_in_esco": len(nm["not_in_esco"]),
            }
            docs.append(doc)
        emb = index.embed([f"{d['title']}. {d['text'][:1500]}" for d in docs])
        index.upsert(docs, emb)
        rows_out += [{k: v for k, v in d.items() if k != "text"} for d in docs]
        print(f"  indexed {min(b + args.batch, len(todo))}/{len(todo)}  ({(time.time()-t0)/60:.1f} min)", flush=True)

    out = export_parquet(index, name)  # the whole collection, so interrupted or pruned builds stay consistent
    rng = random.Random(args.seed)
    spot = out.sample(n=min(20, len(out)), random_state=args.seed)[["job_id", "title", "category", "esco_skills", "raw_skills"]].to_dict("records")
    write_json({"pool": name, "selected": int(len(jobs)), "indexed_total": index.count(), "newly_indexed": len(todo), "build_time_min": round((time.time() - t0) / 60, 1),
                "mean_esco_per_job": float(out["n_esco"].mean()) if len(out) else None, "mean_not_in_esco_per_job": float(out["n_not_in_esco"].mean()) if len(out) else None,
                "categories": out["category"].value_counts().head(15).to_dict() if len(out) else {}, "spot_check_20": spot, "source_info": {k: v for k, v in info.items() if k != "source_columns"}},
               f"results/index/{name}_build.json")
    print(f"done: {index.count()} in {name}; search test:", [r["title"] for r in index.search("machine learning engineer python", k=5)])


if __name__ == "__main__":
    main()
