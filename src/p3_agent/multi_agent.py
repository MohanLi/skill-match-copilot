"""Problem 3 challenger — multi-agent pipeline (LangGraph): Planner → Retriever → Matcher → Verifier.

Planner (LLM) writes target titles, must-have skills and search queries. Retriever (code) runs the
searches, de-duplicates, fetches each candidate and computes compare_skills against the resume profile.
Matcher (LLM) scores and ranks with quotes. Verifier = deterministic quote check + one LLM repair call
per job with flagged claims; unsupported claims are dropped."""
from __future__ import annotations

import json
import time
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from ..utils.config import models_cfg
from ..utils.llm import LLMClient, extract_json, load_prompt
from .common import AgentRun, flag_claims, normalise_ranking, resume_profile
from .tools import ToolBox


class MState(TypedDict):
    resume: str
    preferences: str
    plan: dict[str, Any]
    profile: dict[str, Any]
    candidates: list[dict[str, Any]]
    ranking: list[dict[str, Any]]
    verified: list[dict[str, Any]]


class MultiAgent:
    def __init__(self, toolbox: ToolBox, model: str = "agent", top_n: int | None = None, candidate_k: int | None = None, matcher_max_candidates: int = 15, task: str = "p3_multi_agent", use_verifier: bool = True):
        cfg = models_cfg()["p3_agent"]
        self.tb = toolbox
        self.model = model
        self.top_n = top_n or cfg["top_n"]
        self.max_output_tokens = int(cfg.get("max_output_tokens", 8000))
        self.candidate_k = candidate_k or cfg["candidate_k"]
        self.matcher_max = matcher_max_candidates
        self.use_verifier = use_verifier
        self.client = LLMClient(task=task)
        self.p_plan, self.p_match, self.p_verify = load_prompt("p3_planner_v1"), load_prompt("p3_matcher_v1"), load_prompt("p3_verifier_v1")
        self.graph = self._build()

    # ------------------------------------------------------------------ nodes
    def planner(self, s: MState) -> dict[str, Any]:
        t0 = time.perf_counter()
        try:
            obj, res = self.client.complete_json(self.p_plan, model=self.model, max_tokens=800, resume=s["resume"], preferences=s["preferences"] or "none")
            self._run.add_usage(res)
            plan = obj if isinstance(obj, dict) else {}
        except Exception as e:  # noqa: BLE001
            plan = {"error": str(e)[:200]}
        plan.setdefault("search_queries", [])
        plan.setdefault("target_titles", [])
        self._run.trace.append({"node": "planner", "plan": plan, "ms": round((time.perf_counter() - t0) * 1000, 1)})
        return {"plan": plan}

    def retriever(self, s: MState) -> dict[str, Any]:
        t0 = time.perf_counter()
        prof = resume_profile(self.tb, s["resume"])
        self._run.tool_calls += 2
        queries = [q for q in (s["plan"].get("search_queries") or []) if isinstance(q, str)][:5] + [t for t in (s["plan"].get("target_titles") or []) if isinstance(t, str)][:3]
        if not queries:
            queries = [s["resume"][:300]]
        best: dict[str, dict[str, Any]] = {}
        for q in queries:
            for h in self.tb.search_jobs(q, k=self.candidate_k):
                self._run.tool_calls += 1
                if h["job_id"] not in best or h["score"] > best[h["job_id"]]["score"]:
                    best[h["job_id"]] = h
        if self.tb.pool_ids:  # evaluation: every pool job is a candidate
            for jid in self.tb.pool_ids:
                best.setdefault(jid, {"job_id": jid, "score": 0.0})
        cands = sorted(best.values(), key=lambda h: -h["score"])[: max(self.candidate_k, len(self.tb.pool_ids or []))]
        out = []
        for h in cands:
            j = self.tb.get_job(h["job_id"])
            self._run.tool_calls += 1
            if "error" in j:
                continue
            cmp = self.tb.compare_skills(prof["esco_skills"] + prof["skills"] + prof["knowledge"], j["esco_skills"] + j["raw_skills"][:40])
            self._run.tool_calls += 1
            out.append({**j, "search_score": h["score"], "compare": cmp})
        out.sort(key=lambda c: (-len(c["compare"]["matched"]), -c["search_score"]))
        self._run.trace.append({"node": "retriever", "queries": queries, "n_candidates": len(out), "profile": {k: v[:20] for k, v in prof.items()}, "ms": round((time.perf_counter() - t0) * 1000, 1)})
        return {"profile": prof, "candidates": out}

    def matcher(self, s: MState) -> dict[str, Any]:
        t0 = time.perf_counter()
        cands = s["candidates"][: max(self.matcher_max, len(self.tb.pool_ids or []))]
        blocks = []
        for c in cands:
            blocks.append(json.dumps({"job_id": c["job_id"], "title": c["title"], "company": c["company"], "category": c["category"],
                                      "compare_skills": {"matched": c["compare"]["matched"][:20], "missing": c["compare"]["missing"][:15]},
                                      "job_text": c["text"][:1600]}, ensure_ascii=False))
        try:
            obj, res = self.client.complete_json(self.p_match, model=self.model, max_tokens=self.max_output_tokens, resume=s["resume"], plan=json.dumps(s["plan"], ensure_ascii=False)[:1500], candidates="\n\n".join(blocks), top_n=self.top_n)
            self._run.extra["truncated_output"] = res.stop_reason == "max_tokens"
            self._run.add_usage(res)
            ranking = normalise_ranking(obj, self.top_n)
        except Exception as e:  # noqa: BLE001
            ranking = []
            self._run.error = f"matcher: {str(e)[:200]}"
        valid = {c["job_id"] for c in cands}
        ranking = [r for r in ranking if r["job_id"] in valid]
        self._run.trace.append({"node": "matcher", "n_in": len(cands), "n_ranked": len(ranking), "ms": round((time.perf_counter() - t0) * 1000, 1)})
        return {"ranking": ranking}

    def verifier(self, s: MState) -> dict[str, Any]:
        t0 = time.perf_counter()
        jobs = {c["job_id"]: c for c in s["candidates"]}
        resume_esco = list((s.get("profile") or {}).get("esco_skills") or [])
        verified, n_flagged, n_repaired, n_dropped = [], 0, 0, 0
        for e in s["ranking"]:
            job = jobs.get(e["job_id"], {})
            job_esco = list(job.get("esco_skills") or [])
            claims = flag_claims(e, s["resume"], job.get("text", ""), resume_esco, job_esco)
            flagged = [c for c in claims if c["flag"]]
            n_flagged += len(flagged)
            llm_dropped: list[dict] = []
            if flagged and self.use_verifier:
                try:
                    obj, res = self.client.complete_json(self.p_verify, model=self.model, max_tokens=2000, resume=s["resume"], job_text=job.get("text", "")[:3500], claims=json.dumps(claims, ensure_ascii=False))
                    self._run.add_usage(res)
                    e = {**e, "matched_skills": [c for c in (obj.get("matched_skills") or []) if isinstance(c, dict)], "missing_skills": [c for c in (obj.get("missing_skills") or []) if isinstance(c, dict)]}
                    llm_dropped = [c for c in (obj.get("dropped") or []) if isinstance(c, dict)]
                except Exception as ex:  # noqa: BLE001
                    llm_dropped = [{"skill": c.get("skill"), "reason": f"verifier error: {str(ex)[:80]}"} for c in flagged]
            e["dropped"] = llm_dropped
            # final deterministic pass: anything still unsupported is removed
            claims2 = flag_claims(e, s["resume"], job.get("text", ""), resume_esco, job_esco)
            keep_m = [c for c in claims2 if c["kind"] == "matched" and not c["flag"]]
            keep_x = [c for c in claims2 if c["kind"] == "missing" and not c["flag"]]
            dropped_now = [c for c in claims2 if c["flag"]]
            n_dropped += len(dropped_now) + len(llm_dropped)
            n_repaired += max(0, len(flagged) - len(dropped_now) - len(llm_dropped))
            e["matched_skills"] = [{k: c[k] for k in ("skill", "resume_quote", "job_quote") if k in c} for c in keep_m]
            e["missing_skills"] = [{k: c[k] for k in ("skill", "job_quote") if k in c} for c in keep_x]
            e["dropped"] = (e.get("dropped") or []) + [{"skill": c.get("skill"), "reason": ", ".join(c.get("reasons") or ["unsupported"])} for c in dropped_now]
            verified.append(e)
        self._run.trace.append({"node": "verifier", "flagged": n_flagged, "repaired": n_repaired, "dropped": n_dropped, "ms": round((time.perf_counter() - t0) * 1000, 1)})
        self._run.extra.update({"verifier_flagged": n_flagged, "verifier_repaired": n_repaired, "verifier_dropped": n_dropped})
        return {"verified": verified}

    def _build(self):
        g = StateGraph(MState)
        g.add_node("planner", self.planner)
        g.add_node("retriever", self.retriever)
        g.add_node("matcher", self.matcher)
        g.add_node("verifier", self.verifier)
        g.add_edge(START, "planner")
        g.add_edge("planner", "retriever")
        g.add_edge("retriever", "matcher")
        g.add_edge("matcher", "verifier")
        g.add_edge("verifier", END)
        return g.compile()

    def run(self, resume: str, preferences: str = "") -> AgentRun:
        self._run = AgentRun(approach="multi_agent" if self.use_verifier else "multi_agent_no_verifier", ranking=[])
        t0 = time.perf_counter()
        try:
            final = self.graph.invoke({"resume": resume, "preferences": preferences, "plan": {}, "profile": {}, "candidates": [], "ranking": [], "verified": []})
            self._run.ranking = final["verified"]
            self._run.extra["plan"] = final["plan"]
        except Exception as e:  # noqa: BLE001
            self._run.error = f"{type(e).__name__}: {str(e)[:300]}"
        self._run.extra["wall_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return self._run
