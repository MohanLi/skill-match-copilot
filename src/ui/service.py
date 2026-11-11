"""Run a UI search without loading extraction models for the embedding baseline."""
from __future__ import annotations

import os
import time
from pathlib import Path

from src.utils.config import ROOT, load_env, models_cfg
from src.utils.pii import scrub


def index_status(name: str) -> tuple[int, str | None]:
    if not (ROOT / "chroma").exists():
        return 0, "The local job indexes are missing. Restore the team's chroma/ folder or build an index first."
    try:
        from src.index.store import JobIndex

        count = JobIndex(name).count()
        return count, None if count else "This job index is empty. Build or restore it before searching."
    except Exception:
        return 0, "The job index could not be opened. Check the local Chroma installation and index files."


def setup_issues(mode: str, root: Path = ROOT, cfg: dict | None = None) -> list[str]:
    cfg = cfg or models_cfg()
    issues = []
    encoder = root / cfg["index"]["text_embedding_model"]
    if not (encoder / "config.json").is_file():
        issues.append(f"The sentence encoder is missing: {encoder.relative_to(root)}. Run scripts/download_assets.py or restore the team's model bundle.")
    if mode != "baseline":
        taggers = root / "models" / "p1_tagger"
        separate = all((taggers / f"{kind}_seed42" / "config.json").is_file() for kind in ("skill", "knowledge"))
        merged = (taggers / "merged_seed42" / "config.json").is_file()
        if not (separate or merged):
            issues.append("The trained skill taggers are missing from models/p1_tagger/. Restore the team's trained checkpoints; downloading the base models alone is not enough.")
        load_env()
        if not os.getenv("ANTHROPIC_API_KEY"):
            issues.append("Add ANTHROPIC_API_KEY to your local .env file to use an agent. Semantic search does not need an API key.")
    return issues


_FACTS: dict[tuple[str, int], list[dict]] = {}
_FACT_KEYS = ("job_id", "title", "company", "locations", "remote", "published")


def job_facts(index) -> list[dict]:
    """Title + job facts of every posting in the index (no texts, no skills), read once per index size."""
    key = (getattr(index, "name", ""), index.count())
    if key not in _FACTS:
        _FACTS.clear()
        _FACTS[key] = [{k: m[k] for k in _FACT_KEYS if k in m} for m in index.all_metadata()]
    return _FACTS[key]


def clear_cache() -> None:
    """Forget the cached job facts (after the index changed without changing size, e.g. python -m src.index.job_facts)."""
    _FACTS.clear()


def requirement_options(index_name: str) -> dict:
    """What the form needs to know about a job source: the most common places (as suggestions) and whether the
    index has location facts at all. Never raises — a missing index just means no suggestions."""
    try:
        from collections import Counter

        from src.index.store import JobIndex
        from src.p3_agent.requirements import has_location_facts

        facts = job_facts(JobIndex(index_name))
        places = Counter()
        for m in facts:
            for place in str(m.get("locations") or "").split(" | "):
                city = place.split(",")[0].strip()
                if city and "remote" not in city.lower() and len(city) <= 40:
                    places[city] += 1
        return {"locations": [c for c, _ in places.most_common(40)], "location_data": any(has_location_facts(m) for m in facts), "jobs": len(facts)}
    except Exception:
        return {"locations": [], "location_data": True, "jobs": 0}


def similar_jobs(index_name: str, job_id: str, requirements=None, exclude=(), k: int = 5) -> list[dict]:
    """Postings closest to one posting (same embedding space as the search, no LLM), inside the same requirements."""
    from src.index.store import JobIndex
    from src.p3_agent.common import REPEAT_COSINE, same_job
    from src.p3_agent.requirements import Requirements, select_pool

    index = JobIndex(index_name)
    res = index.col.get(ids=[job_id], include=["embeddings"])
    if not len(res["ids"]):
        return []
    req = Requirements.from_any(requirements)
    skip = set(exclude) | {job_id}
    allowed = None
    if req.filters:
        pool = select_pool(job_facts(index), req, need=k + len(skip))
        allowed = [i for i in pool["ids"] if i not in skip] if pool["restricted"] else None
        if allowed is not None and not allowed:
            return []
    kwargs = {"query_embeddings": [list(map(float, res["embeddings"][0]))], "include": ["metadatas", "distances"]}
    if allowed is not None:
        kwargs.update(ids=allowed, n_results=min(2 * k + 5, len(allowed)))
    else:
        kwargs.update(n_results=min(2 * k + 5 + len(skip), index.count()))
    out = index.col.query(**kwargs)
    hits: list[dict] = []
    for i, m, d in zip(out["ids"][0], out["metadatas"][0], out["distances"][0]):
        hit = {"job_id": i, "score": round(1 - float(d), 4), **(m or {})}
        if i in skip or hit["score"] >= REPEAT_COSINE:                       # the posting itself or a repeat of it
            continue
        if any(same_job(hit, h) and abs(hit["score"] - h["score"]) < 0.005 for h in hits):   # the same job listed twice
            continue
        hits.append(hit)
    return hits[:k]


def run_search(resume: str, preferences: str, index_name: str, mode: str, top_n: int,
               model: str = "agent", model_loader=None, requirements=None) -> dict:
    """`requirements` (role / location / remote, see src/p3_agent/requirements.py) restrict the job pool before the
    chosen approach ranks it; `preferences` is free text that only the agents read."""
    from src.p3_agent.baseline import EmbeddingBaseline
    from src.p3_agent.common import faithfulness_of_ranking, resume_profile
    from src.p3_agent.match import match_jobs
    from src.p3_agent.requirements import Requirements, pool_summary
    from src.p3_agent.tools import ToolBox

    clean = scrub(resume).text
    clean_preferences = scrub(preferences).text
    req = Requirements.from_any(requirements)
    req.notes = clean_preferences
    started = time.perf_counter()
    if mode == "baseline":
        tb = ToolBox(index_name)
        agent = EmbeddingBaseline(tb, top_n=top_n + 5)   # spare candidates: repeat postings are merged afterwards
    else:
        from src.p3_agent.multi_agent import MultiAgent
        from src.p3_agent.single_agent import SingleAgent
        from src.p3_agent.verified import VerifiedSingleAgent

        extractor, normalizer = model_loader()
        tb = ToolBox(index_name, extractor=extractor, normalizer=normalizer)
        cls = {"single": SingleAgent, "single_verified": VerifiedSingleAgent, "multi": MultiAgent}[mode]
        agent = cls(tb, model=model, top_n=top_n, task=f"ui_{mode}")
    if tb.index.count() == 0:
        raise ValueError("The selected job index is empty. Build or restore it before searching.")
    run, jobs, checks, pool = match_jobs(agent, tb, clean, req, top_n, facts=job_facts(tb.index) if req.filters else None, baseline=mode == "baseline")
    profile = resume_profile(tb, clean) if mode != "baseline" else None
    faith = faithfulness_of_ranking(run.ranking, clean, jobs, profile["esco_skills"]) if profile is not None else None
    return {"run": run.to_dict(), "jobs": jobs, "profile": profile, "faithfulness": faith,
            "elapsed": time.perf_counter() - started, "mode": mode, "index": index_name,
            "preferences": clean_preferences, "top_n": top_n, "model": model,
            "requirements": req.to_dict(), "pool": pool_summary(pool), "checks": checks}
