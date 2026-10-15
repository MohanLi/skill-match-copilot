"""Problem 3 baseline (no LLM): embed the whole resume with the bi-encoder and rank jobs by cosine
similarity. No explanations, so faithfulness is not applicable."""
from __future__ import annotations

import time

from ..utils.config import models_cfg
from .common import AgentRun
from .tools import ToolBox


class EmbeddingBaseline:
    def __init__(self, toolbox: ToolBox, top_n: int | None = None):
        self.tb = toolbox
        self.top_n = top_n or models_cfg()["p3_agent"]["top_n"]

    def run(self, resume: str, preferences: str = "") -> AgentRun:
        t0 = time.perf_counter()
        hits = self.tb.search_jobs(resume[:4000], k=max(self.top_n, len(self.tb.pool_ids or [])))
        ranking = [{"job_id": h["job_id"], "rank": i + 1, "fit_score": round(100 * max(0.0, h["score"]), 1), "matched_skills": [], "missing_skills": [], "summary": ""} for i, h in enumerate(hits[: self.top_n])]
        run = AgentRun(approach="embedding_baseline", ranking=ranking, tool_calls=1)
        run.extra["wall_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return run
