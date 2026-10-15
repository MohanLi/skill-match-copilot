"""Wiring tests for the Problem 3 approaches with a fake index, fake extractor/normalizer and a scripted
LLM — no models, no API key, no network. Catches graph/tool/parsing errors early."""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pytest

from src.p3_agent import common
from src.p3_agent.baseline import EmbeddingBaseline
from src.p3_agent.multi_agent import MultiAgent
from src.p3_agent.single_agent import SingleAgent
from src.p3_agent.tools import ToolBox
from src.utils.llm import LLMResult

JOBS = {
    "j1": {"title": "Senior Python Backend Engineer", "category": "Python", "text": "Senior Python Backend Engineer. Requirements: 5+ years Python, Django, PostgreSQL, Docker and Kubernetes. Nice: AWS.", "esco": ["Python (computer programming)", "Docker", "Kubernetes"], "raw": ["Python", "Django", "PostgreSQL", "Docker", "Kubernetes", "AWS"]},
    "j2": {"title": "Machine Learning Engineer", "category": "Data Science", "text": "Machine Learning Engineer. We need PyTorch, Python, MLOps with Docker, and experience with LLM applications.", "esco": ["Python (computer programming)", "Docker", "use machine learning"], "raw": ["PyTorch", "Python", "MLOps", "Docker", "LLM applications"]},
    "j3": {"title": "Manual QA Engineer", "category": "QA", "text": "Manual QA Engineer. Test cases, bug reports, Jira. No coding required.", "esco": ["perform software tests"], "raw": ["Test cases", "bug reports", "Jira"]},
    "j4": {"title": "React Frontend Developer", "category": "JavaScript", "text": "React Frontend Developer: React, TypeScript, CSS, REST APIs.", "esco": ["JavaScript"], "raw": ["React", "TypeScript", "CSS", "REST APIs"]},
}
RESUME = "Jordan Reyes, Machine Learning Engineer. Skills: Python, PyTorch, Docker, Kubernetes, FastAPI. Built a RAG assistant on top of Claude and shipped LLM applications to production."


def fake_vec(text: str, dim: int = 16) -> list[float]:
    rng = np.random.default_rng(int(hashlib.md5(text.lower().encode()).hexdigest()[:8], 16))
    v = rng.normal(size=dim)
    # make vectors with shared keywords closer: add keyword hashes
    for kw in ("python", "pytorch", "docker", "kubernetes", "react", "qa", "machine learning", "llm"):
        if kw in text.lower():
            r2 = np.random.default_rng(int(hashlib.md5(kw.encode()).hexdigest()[:8], 16))
            v += 3 * r2.normal(size=dim)
    return (v / np.linalg.norm(v)).tolist()


class FakeNormalizer:
    def embed(self, texts):
        return [fake_vec(t) for t in texts]

    def normalize(self, phrases):
        table = {"python": "Python (computer programming)", "docker": "Docker", "kubernetes": "Kubernetes", "pytorch": "use machine learning", "react": "JavaScript"}
        mapped = [{"phrase": p, "esco_label": table.get(p.lower(), "NOT_IN_ESCO"), "score": 0.9 if p.lower() in table else 0.2, "in_esco": p.lower() in table} for p in phrases]
        return {"mapped": mapped, "esco_skills": sorted({m["esco_label"] for m in mapped if m["in_esco"]}), "not_in_esco": [m["phrase"] for m in mapped if not m["in_esco"]]}


class FakeExtractor:
    def extract(self, text):
        kws = ["Python", "PyTorch", "Docker", "Kubernetes", "FastAPI", "React", "TypeScript", "Django", "PostgreSQL", "Jira"]
        found = [k for k in kws if k.lower() in text.lower()]
        return {"skills": [], "knowledge": found}


class FakeCol:
    def __init__(self, jobs, emb):
        self.jobs, self.emb = jobs, emb

    def get(self, ids=None, include=None, **kw):
        ids = ids or list(self.jobs)
        out = {"ids": ids, "metadatas": [{k: v for k, v in self.jobs[i].items() if k not in ("job_id", "text")} for i in ids], "documents": [self.jobs[i]["text"] for i in ids]}
        if include and "embeddings" in include:
            out["embeddings"] = [self.emb[i] for i in ids]
        return out


class FakeIndex:
    """Mimics the JobIndex API used by ToolBox (col.get with embeddings, get, get_many, search)."""

    def __init__(self):
        self.jobs = {jid: {"job_id": jid, "title": j["title"], "category": j["category"], "company": "ACME", "esco_skills": "|".join(j["esco"]), "raw_skills": "|".join(j["raw"]), "text": j["text"]} for jid, j in JOBS.items()}
        self.emb = {jid: fake_vec(j["title"] + ". " + j["text"]) for jid, j in JOBS.items()}
        self.col = FakeCol(self.jobs, self.emb)

    def get(self, job_id):
        return self.jobs.get(job_id)

    def get_many(self, ids):
        return [self.jobs[i] for i in ids if i in self.jobs]

    def search(self, query, k=30, where=None):
        q = np.array(fake_vec(query))
        scored = sorted(((float(np.array(e) @ q), i) for i, e in self.emb.items()), reverse=True)[:k]
        return [{"job_id": i, "score": round(s, 4), **{kk: v for kk, v in self.jobs[i].items() if kk != "job_id"}} for s, i in scored]

    def count(self):
        return len(self.jobs)


class ScriptedLLM:
    """Replaces LLMClient: tool-use first, then the final JSON (single agent); JSON answers for the multi-agent prompts."""

    def __init__(self):
        self.n = 0

    def resolve_model(self, m):
        return "fake-model"

    def _res(self, content, stop):
        text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
        return LLMResult(text=text, content=content, stop_reason=stop, model="fake-model", prompt_version="test", input_tokens=100, output_tokens=50, latency_ms=5.0, cost_usd=0.0001, cached=False)

    def chat(self, messages, system="", model="agent", prompt_version="x", tools=None, max_tokens=None, temperature=None, tool_choice=None):
        self.n += 1
        if tools and self.n == 1:
            return self._res([{"type": "text", "text": "Let me search."}, {"type": "tool_use", "id": "t1", "name": "search_jobs", "input": {"query": "machine learning engineer python", "k": 5}},
                              {"type": "tool_use", "id": "t2", "name": "extract_skills", "input": {"text": RESUME}}], "tool_use")
        if tools and self.n == 2:
            return self._res([{"type": "tool_use", "id": "t3", "name": "get_job", "input": {"job_id": "j2"}}, {"type": "tool_use", "id": "t4", "name": "compare_skills", "input": {"resume_skills": ["Python", "Docker"], "job_skills": ["Python", "PyTorch"]}}], "tool_use")
        final = {"ranking": [
            {"job_id": "j2", "rank": 1, "fit_score": 90, "matched_skills": [{"skill": "PyTorch", "resume_quote": "PyTorch, Docker", "job_quote": "We need PyTorch"}, {"skill": "Spark", "resume_quote": "Python", "job_quote": "Spark experience"}], "missing_skills": [{"skill": "MLOps", "job_quote": "MLOps with Docker"}], "summary": "good"},
            {"job_id": "j1", "rank": 2, "fit_score": 60, "matched_skills": [{"skill": "Docker", "resume_quote": "Docker, Kubernetes", "job_quote": "Docker and Kubernetes"}], "missing_skills": [{"skill": "Django", "job_quote": "Django"}], "summary": "partial"},
            {"job_id": "j4", "rank": 3, "fit_score": 10, "matched_skills": [], "missing_skills": [], "summary": "no"},
            {"job_id": "j3", "rank": 4, "fit_score": 5, "matched_skills": [], "missing_skills": [], "summary": "no"}]}
        return self._res([{"type": "text", "text": "FINAL ANSWER\n" + json.dumps(final)}], "end_turn")

    def complete_json(self, prompt, model="agent", max_tokens=None, **kw):
        self.n += 1
        if prompt.id == "p3_planner":
            return {"role_family": "machine learning", "seniority": "middle", "target_titles": ["Machine Learning Engineer"], "must_have_skills": ["Python", "PyTorch"], "search_queries": ["machine learning engineer python pytorch"]}, self._res([{"type": "text", "text": "{}"}], "end_turn")
        if prompt.id == "p3_matcher":
            return json.loads(self.chat([], tools=None).text.split("FINAL ANSWER\n")[1]), self._res([{"type": "text", "text": "{}"}], "end_turn")
        if prompt.id == "p3_verifier":
            claims = json.loads(kw["claims"])
            keep = [c for c in claims if not c["flag"]]
            return {"job_id": "x", "matched_skills": [{k: c[k] for k in ("skill", "resume_quote", "job_quote") if k in c} for c in keep if c["kind"] == "matched"],
                    "missing_skills": [{k: c[k] for k in ("skill", "job_quote") if k in c} for c in keep if c["kind"] == "missing"], "dropped": [{"skill": c["skill"], "reason": "no quote"} for c in claims if c["flag"]]}, self._res([{"type": "text", "text": "{}"}], "end_turn")
        raise AssertionError(prompt.id)


@pytest.fixture
def toolbox():
    tb = ToolBox.__new__(ToolBox)
    tb.index = FakeIndex()
    tb.pool_ids = list(JOBS)
    tb._extractor, tb._normalizer = FakeExtractor(), FakeNormalizer()
    tb.snippet_chars, tb.job_text_chars = 300, 3500
    tb.calls, tb._pool_cache = [], None
    return tb


def test_tools(toolbox):
    hits = toolbox.search_jobs("machine learning engineer python", k=3)
    assert len(hits) == 3 and hits[0]["job_id"] in ("j2", "j1")
    assert toolbox.get_job("j2")["title"].startswith("Machine")
    assert "error" in toolbox.get_job("nope")
    out = json.loads(toolbox.dispatch("compare_skills", {"resume_skills": ["Python"], "job_skills": ["python", "Java"]}))
    assert out["matched"] == ["python"] and out["missing"] == ["Java"]


def test_baseline(toolbox):
    run = EmbeddingBaseline(toolbox, top_n=4).run(RESUME)
    assert [e["job_id"] for e in run.ranking][0] in ("j2", "j1") and len(run.ranking) == 4


def test_single_agent(toolbox):
    ag = SingleAgent.__new__(SingleAgent)
    ag.tb, ag.model, ag.top_n, ag.candidate_k, ag.max_steps = toolbox, "agent", 4, 5, 6
    ag.client = ScriptedLLM()
    from src.utils.llm import load_prompt

    ag.prompt = load_prompt("p3_single_agent_v1")
    ag.graph = ag._build()
    run = ag.run(RESUME)
    assert run.error is None, run.error
    assert [e["job_id"] for e in run.ranking] == ["j2", "j1", "j4", "j3"]
    assert run.tool_calls == 4 and run.llm_calls == 3
    faith = common.faithfulness_of_ranking(run.ranking, RESUME, toolbox.index.jobs, ["Python (computer programming)"])
    assert faith["n_claims"] == 5 and faith["n_supported"] == 4  # the invented Spark claim is caught


def test_multi_agent(toolbox):
    ag = MultiAgent.__new__(MultiAgent)
    ag.tb, ag.model, ag.top_n, ag.candidate_k, ag.matcher_max, ag.use_verifier = toolbox, "agent", 4, 5, 15, True
    ag.client = ScriptedLLM()
    from src.utils.llm import load_prompt

    ag.p_plan, ag.p_match, ag.p_verify = load_prompt("p3_planner_v1"), load_prompt("p3_matcher_v1"), load_prompt("p3_verifier_v1")
    ag.graph = ag._build()
    run = ag.run(RESUME)
    assert run.error is None, run.error
    assert [e["job_id"] for e in run.ranking] == ["j2", "j1", "j4", "j3"]
    assert run.extra["verifier_flagged"] == 1 and run.extra["verifier_dropped"] == 1
    faith = common.faithfulness_of_ranking(run.ranking, RESUME, toolbox.index.jobs, ["Python (computer programming)"])
    assert faith["n_claims"] == 4 and faith["n_supported"] == 4  # verifier removed the unsupported claim


def test_metrics():
    from src.p3_agent.evaluate import mrr, ndcg_at_k, precision_at_k

    assert math.isclose(ndcg_at_k([2, 1, 0], [2, 1, 0]), 1.0)
    assert ndcg_at_k([0, 1, 2], [2, 1, 0]) < 1.0
    assert precision_at_k([2, 2, 0]) == pytest.approx(2 / 3)
    assert mrr([0, 1, 2]) == pytest.approx(1 / 3) and mrr([0, 1]) is None
