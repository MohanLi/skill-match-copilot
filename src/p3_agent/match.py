"""Run one approach for a resume *within the candidate's requirements* — the code path the UI and the API share.

1. requirements → eligible job pool (src/p3_agent/requirements.py); the toolbox is restricted to it
2. the approach ranks (single agent, verified agent, multi-agent or embedding baseline — unchanged)
3. each returned job is checked against the requirements, full matches are listed first,
   and repeat postings of one job are folded into a single result"""
from __future__ import annotations

from typing import Any

from .common import AgentRun, job_vectors, merge_repeats
from .requirements import Requirements, check_job, order_by_requirements, select_pool


def _baseline_entries(tb, resume: str, job_ids: list[str]) -> list[dict[str, Any]]:
    """Similarity-ranked entries for specific postings, in the embedding baseline's own format."""
    previous = getattr(tb, "allowed_ids", None)
    tb.restrict_to(job_ids)
    hits = tb.search_jobs(resume[:4000], k=len(job_ids))
    tb.restrict_to(previous)
    return [{"job_id": h["job_id"], "rank": 0, "fit_score": round(100 * max(0.0, h["score"]), 1), "matched_skills": [], "missing_skills": [], "summary": ""} for h in hits]


def match_jobs(approach, tb, resume: str, req: Requirements, top_n: int, facts: list[dict[str, Any]] | None = None,
               baseline: bool = False) -> tuple[AgentRun, dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, Any] | None]:
    """→ (run, jobs by id, requirement checks by id, pool). `facts` = the index's job facts if the caller has them cached."""
    pool = None
    if req.filters:
        pool = select_pool(facts if facts is not None else tb.index.all_metadata(), req, need=top_n)
        if pool["restricted"]:
            tb.restrict_to(pool["ids"])
    run = approach.run(resume, req.to_prompt(pool) if req.filters else req.notes)
    if baseline and pool and pool["must_include"]:   # the few full matches must not lose their place to closer-sounding partial ones
        have = {e["job_id"] for e in run.ranking}
        missing = [job_id for job_id in pool["must_include"] if job_id not in have]
        if missing:
            run.ranking = run.ranking + _baseline_entries(tb, resume, missing)
    jobs = {j["job_id"]: j for j in tb.index.get_many([e["job_id"] for e in run.ranking])} if run.ranking else {}
    checks: dict[str, list[dict[str, Any]]] = {}
    if req.filters:
        run.ranking = [e for e in run.ranking if e["job_id"] in jobs]              # an id the index does not have cannot be checked
        checks = {job_id: check_job(job, req) for job_id, job in jobs.items()}
        run.ranking = order_by_requirements(run.ranking, checks)
    run.ranking = merge_repeats(run.ranking, jobs, job_vectors(tb.index, [e["job_id"] for e in run.ranking]))[:top_n]
    return run, jobs, checks, pool
