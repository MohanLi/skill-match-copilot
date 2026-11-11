"""Shared pieces for the Problem 3 approaches: run result container, resume skill profile, output
normalisation and the deterministic quote check used by the Verifier and the evaluation."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from .faithfulness import check_explanation, quote_in_text


@dataclass
class AgentRun:
    approach: str
    ranking: list[dict[str, Any]]
    trace: list[dict[str, Any]] = field(default_factory=list)
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    tool_calls: int = 0
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def add_usage(self, res) -> None:
        self.llm_calls += 1
        self.input_tokens += res.input_tokens
        self.output_tokens += res.output_tokens
        self.cost_usd += res.cost_usd
        self.latency_ms += res.latency_ms

    def to_dict(self) -> dict[str, Any]:
        return {"approach": self.approach, "ranking": self.ranking, "trace": self.trace, "llm_calls": self.llm_calls, "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens, "cost_usd": round(self.cost_usd, 6), "latency_ms": round(self.latency_ms, 1), "tool_calls": self.tool_calls, "error": self.error, **self.extra}


def resume_profile(toolbox, resume: str) -> dict[str, Any]:
    """Extract + normalize the resume once (code path shared by the multi-agent retriever and the baseline)."""
    ex = toolbox.extract_skills(resume)
    nm = toolbox.normalize_skills(ex["skills"] + ex["knowledge"])
    return {"skills": ex["skills"], "knowledge": ex["knowledge"], "esco_skills": nm["esco_skills"], "not_in_esco": nm["not_in_esco"],
            "mapped": nm.get("mapped", [])}  # per-phrase (phrase, esco_label, score) for display


def normalise_ranking(obj: Any, top_n: int) -> list[dict[str, Any]]:
    """Coerce whatever the LLM returned into [{job_id, rank, fit_score, matched_skills, missing_skills, summary}]."""
    items = obj.get("ranking") if isinstance(obj, dict) else obj
    if not isinstance(items, list):
        return []
    out = []
    for i, it in enumerate(items):
        if not isinstance(it, dict) or not it.get("job_id"):
            continue
        ms = [c for c in (it.get("matched_skills") or []) if isinstance(c, dict)]
        mi = [c for c in (it.get("missing_skills") or []) if isinstance(c, dict)]
        try:
            fit = float(it.get("fit_score", 0) or 0)
        except (TypeError, ValueError):
            fit = 0.0
        out.append({"job_id": str(it["job_id"]), "rank": i + 1, "fit_score": fit, "matched_skills": ms, "missing_skills": mi, "summary": str(it.get("summary") or "")[:600]})
    seen, dedup = set(), []
    for it in out:
        if it["job_id"] not in seen:
            seen.add(it["job_id"])
            dedup.append(it)
    for i, it in enumerate(dedup):
        it["rank"] = i + 1
    return dedup[:top_n]


def flag_claims(entry: dict[str, Any], resume: str, job_text: str, resume_esco: list[str] | None = None, job_esco: list[str] | None = None) -> list[dict[str, Any]]:
    """Deterministic half of the Verifier. Applies exactly the checks the faithfulness metric applies
    (src/p3_agent/faithfulness.py): quotes must be verbatim, a matched skill must be named in its quotes (or be an
    ESCO skill of both sides), a missing skill must be named in the job and absent from the resume."""
    from .faithfulness import check_matched, check_missing

    claims = []
    for c in entry.get("matched_skills", []):
        chk = check_matched(c, resume, job_text, resume_esco, job_esco)
        claims.append({"kind": "matched", **c, "flag": not chk.supported, "reasons": chk.reasons})
    for c in entry.get("missing_skills", []):
        chk = check_missing(c, resume, job_text, resume_esco, job_esco)
        claims.append({"kind": "missing", **c, "flag": not chk.supported, "reasons": chk.reasons})
    return claims


def faithfulness_of_ranking(ranking: list[dict[str, Any]], resume: str, jobs: dict[str, dict[str, Any]], resume_esco: list[str] | None = None) -> dict[str, Any]:
    n_claims = n_sup = 0
    reasons: dict[str, int] = {}
    per_job = []
    for e in ranking:
        j = jobs.get(e["job_id"])
        if not j:
            continue
        from ..index.store import split_skills

        rep = check_explanation(e, resume, j.get("text", ""), resume_esco, split_skills(j.get("esco_skills")))
        s = rep.summary()
        n_claims += s["n_claims"]
        n_sup += s["n_supported"]
        for k, v in s["reasons"].items():
            reasons[k] = reasons.get(k, 0) + v
        per_job.append({"job_id": e["job_id"], **s})
    return {"n_claims": n_claims, "n_supported": n_sup, "hallucination_rate": (1 - n_sup / n_claims) if n_claims else 0.0, "reasons": reasons, "per_job": per_job}


REPEAT_COSINE = 0.98   # same company + title: repeats of one posting sit at >= 0.98, different jobs with one title below 0.97


def same_job(a: dict, b: dict) -> bool:
    name = lambda job, key: " ".join(str(job.get(key) or "").lower().split())  # noqa: E731
    return bool(name(a, "title")) and name(a, "title") == name(b, "title") and name(a, "company") == name(b, "company")


def merge_repeats(ranking: list[dict], jobs: dict, vectors: dict) -> list[dict]:
    """One card per job. Employers post the same job once per location (6.5% of the live index); a later result with
    the same company, title and near-identical text is folded into the earlier one as `also` instead of taking a slot."""
    kept: list[dict] = []
    for entry in ranking:
        job = jobs.get(entry["job_id"], {})
        vec = vectors.get(entry["job_id"])
        twin = None
        for first in kept:
            other = vectors.get(first["job_id"])
            if vec is not None and other is not None and same_job(job, jobs.get(first["job_id"], {})) \
                    and float(sum(x * y for x, y in zip(vec, other))) >= REPEAT_COSINE:
                twin = first
                break
        if twin is None:
            kept.append(entry)
        else:
            twin.setdefault("also", []).append({"job_id": entry["job_id"], "locations": job.get("locations"), "url": job.get("url")})
    for i, entry in enumerate(kept):
        entry["rank"] = i + 1
    return kept


def job_vectors(index, job_ids: list[str]) -> dict:
    """Stored embeddings of a few postings, by job id (for merge_repeats)."""
    if not job_ids:
        return {}
    res = index.col.get(ids=job_ids, include=["embeddings"])
    return {i: [float(x) for x in e] for i, e in zip(res["ids"], res["embeddings"])}
