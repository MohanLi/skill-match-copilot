"""FastAPI service — Model Ops deployment surface (loads the fine-tuned models once at startup).

  uvicorn src.api.app:app --host 0.0.0.0 --port 8000
Endpoints: GET /health · POST /extract · POST /normalize · POST /search · POST /match
Every /match request logs the verifier's unsupported-claim rate to logs/requests.jsonl (monitoring input)."""
from __future__ import annotations

import time
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..utils.config import ROOT, git_commit, models_cfg
from ..utils.io import append_jsonl
from ..utils.pii import scrub

app = FastAPI(title="Skill-Match Copilot API", version="0.1.0", description="Skill extraction, ESCO normalization and explainable resume–job matching.")
STATE: dict[str, Any] = {}


class TextIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=20000)


class PhrasesIn(BaseModel):
    phrases: list[str] = Field(..., max_length=200)


class SearchIn(BaseModel):
    query: str
    k: int = 10
    index: str = "live_jobs"
    filters: dict[str, Any] | None = None


class RequirementsIn(BaseModel):
    """Role / location / remote requirements (src/p3_agent/requirements.py): they restrict the job pool before ranking."""
    roles: list[str] = Field(default_factory=list, max_length=10)
    locations: list[str] = Field(default_factory=list, max_length=20)
    remote_ok: bool = False
    remote_within: list[str] = Field(default_factory=list, max_length=10)


class MatchIn(BaseModel):
    resume: str = Field(..., min_length=50, max_length=20000)
    preferences: str = ""                                    # free text for the agent
    requirements: RequirementsIn | None = None
    mode: str = Field("single_verified", pattern="^(single_verified|single|multi|baseline)$")  # champion first
    index: str = "live_jobs"
    top_n: int = 5
    model: str = "agent"


@app.on_event("startup")
def _load() -> None:
    from ..p1_extract.predict import SkillExtractor
    from ..p2_normalize.normalizer import get_normalizer

    STATE["extractor"] = SkillExtractor()
    STATE["normalizer"] = get_normalizer()
    STATE["started"] = time.time()


@app.get("/health")
def health() -> dict[str, Any]:
    from ..index.store import JobIndex

    counts = {}
    for name in ("djinni_jobs", "live_jobs"):
        try:
            counts[name] = JobIndex(name).count()
        except Exception:  # noqa: BLE001
            counts[name] = None
    return {"status": "ok", "git_commit": git_commit(), "uptime_s": round(time.time() - STATE.get("started", time.time()), 1), "indexes": counts, "llm": models_cfg()["llm"]["agent"], "normalizer_threshold": STATE["normalizer"].threshold}


@app.post("/extract")
def extract(inp: TextIn) -> dict[str, Any]:
    t0 = time.perf_counter()
    out = STATE["extractor"].extract(inp.text)
    return {**out, "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}


@app.post("/normalize")
def normalize(inp: PhrasesIn) -> dict[str, Any]:
    t0 = time.perf_counter()
    out = STATE["normalizer"].normalize(inp.phrases)
    return {**out, "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}


@app.post("/search")
def search(inp: SearchIn) -> dict[str, Any]:
    from ..p3_agent.tools import ToolBox

    tb = ToolBox(inp.index, extractor=STATE["extractor"], normalizer=STATE["normalizer"])
    return {"results": tb.search_jobs(inp.query, k=inp.k, filters=inp.filters)}


@app.post("/match")
def match(inp: MatchIn) -> dict[str, Any]:
    from ..p3_agent.baseline import EmbeddingBaseline
    from ..p3_agent.common import faithfulness_of_ranking, resume_profile
    from ..p3_agent.match import match_jobs
    from ..p3_agent.multi_agent import MultiAgent
    from ..p3_agent.requirements import Requirements, pool_summary
    from ..p3_agent.single_agent import SingleAgent
    from ..p3_agent.tools import ToolBox
    from ..p3_agent.verified import VerifiedSingleAgent

    t0 = time.perf_counter()
    resume = scrub(inp.resume).text
    tb = ToolBox(inp.index, extractor=STATE["extractor"], normalizer=STATE["normalizer"])
    if tb.index.count() == 0:
        raise HTTPException(status_code=503, detail=f"index {inp.index} is empty; run src.index.build_job_index")
    approach = {"single_verified": lambda: VerifiedSingleAgent(tb, model=inp.model, top_n=inp.top_n, task="api_single_verified"),
                "single": lambda: SingleAgent(tb, model=inp.model, top_n=inp.top_n, task="api_single"), "multi": lambda: MultiAgent(tb, model=inp.model, top_n=inp.top_n, task="api_multi"),
                "baseline": lambda: EmbeddingBaseline(tb, top_n=inp.top_n + 5)}[inp.mode]()   # spare candidates: repeat postings are merged
    req = Requirements.from_any(inp.requirements.model_dump() if inp.requirements else None)
    req.notes = inp.preferences.strip()
    # only postings that meet the requirements are searchable; too few → topped up and flagged per job (src/p3_agent/match.py)
    run, jobs, checks, pool = match_jobs(approach, tb, resume, req, inp.top_n, baseline=inp.mode == "baseline")
    prof = resume_profile(tb, resume) if inp.mode != "baseline" else {"esco_skills": []}
    faith = faithfulness_of_ranking(run.ranking, resume, jobs, prof.get("esco_skills")) if inp.mode != "baseline" else None
    for e in run.ranking:
        j = jobs.get(e["job_id"], {})
        e["title"], e["company"], e["url"], e["category"] = j.get("title"), j.get("company"), j.get("url"), j.get("category")
        e["locations"], e["remote"], e["published"] = j.get("locations"), j.get("remote"), j.get("published")
        if req.filters:
            e["requirements"] = checks.get(e["job_id"], [])
    rec = {"ts": time.time(), "mode": inp.mode, "index": inp.index, "model": inp.model, "n_results": len(run.ranking), "llm_calls": run.llm_calls, "cost_usd": run.cost_usd,
           "latency_ms": round((time.perf_counter() - t0) * 1000, 1), "hallucination_rate": faith["hallucination_rate"] if faith else None, "n_claims": faith["n_claims"] if faith else 0, "error": run.error}
    append_jsonl(rec, "logs/requests.jsonl")
    return {"ranking": run.ranking, "requirements": {"asked": req.to_dict(), "pool": pool_summary(pool)} if req.filters else None, "profile": prof, "faithfulness": {k: v for k, v in faith.items() if k != "per_job"} if faith else None, "usage": {k: v for k, v in run.to_dict().items() if k in ("llm_calls", "input_tokens", "output_tokens", "cost_usd", "latency_ms", "tool_calls", "error")},
            "trace": run.trace, "latency_ms": rec["latency_ms"]}
