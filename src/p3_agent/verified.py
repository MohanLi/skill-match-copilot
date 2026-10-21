"""Single ReAct agent + the multi-agent Verifier as a post-processor.

The evaluation showed the two arms win on different axes (the ReAct agent ranks best, the planner→matcher→verifier
pipeline is the only one with zero unsupported claims). The verifier is a modular step, so this approach checks
whether the two strengths combine: run the single agent, then verify/repair/drop its claims with exactly the same
verifier node (deterministic check + one LLM repair call per flagged job)."""
from __future__ import annotations

import time
from typing import Any

from ..index.store import split_skills
from .common import AgentRun, resume_profile
from .multi_agent import MultiAgent
from .single_agent import SingleAgent
from .tools import ToolBox


class VerifiedSingleAgent:
    def __init__(self, toolbox: ToolBox, model: str = "agent", top_n: int | None = None, task: str = "p3_single_verified"):
        self.tb = toolbox
        self.inner = SingleAgent(toolbox, model=model, top_n=top_n, task="p3_single_agent")  # same cache namespace as the plain agent
        self.host = MultiAgent(toolbox, model=model, top_n=top_n, task=task)  # only its verifier node is used

    def run(self, resume: str, preferences: str = "") -> AgentRun:
        t0 = time.perf_counter()
        run = self.inner.run(resume, preferences)
        run.approach = "single_agent_verified"
        if run.ranking:
            jobs = self.tb.index.get_many([e["job_id"] for e in run.ranking])
            for j in jobs:
                j["esco_skills"] = split_skills(j.get("esco_skills"))
            prof = resume_profile(self.tb, resume)
            self.host._run = run  # usage, trace and verifier counters accumulate on the same run
            out = self.host.verifier({"resume": resume, "preferences": preferences, "plan": {}, "profile": prof, "candidates": jobs, "ranking": run.ranking, "verified": []})
            run.ranking = out["verified"]
        run.extra["wall_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return run
