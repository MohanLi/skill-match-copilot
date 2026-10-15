"""The five tools every Problem 3 approach shares (same code for the single agent, the multi-agent
pipeline and the API). A ToolBox can be restricted to a fixed job pool (evaluation) and logs every call.

Tools: extract_skills(text), normalize_skills(phrases), search_jobs(query, k, filters), get_job(job_id),
compare_skills(resume_skills, job_skills)."""
from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
from rapidfuzz import fuzz

from ..index.store import JobIndex, split_skills
from ..p1_extract.predict import SkillExtractor
from ..p2_normalize.normalizer import get_normalizer

TOOL_SCHEMAS = [
    {"name": "extract_skills", "description": "Extract skill phrases (things a person can do) and knowledge phrases (technologies, domains, degrees) from a text using the fine-tuned tagger. Input: text. Output: {skills: [...], knowledge: [...]}.",
     "input_schema": {"type": "object", "properties": {"text": {"type": "string", "description": "Resume or job text (up to ~6000 characters)."}}, "required": ["text"]}},
    {"name": "normalize_skills", "description": "Map free-text skill phrases to ESCO taxonomy skills. Output: {esco_skills: [...], not_in_esco: [...], mapped: [{phrase, esco_label, score}]}.",
     "input_schema": {"type": "object", "properties": {"phrases": {"type": "array", "items": {"type": "string"}}}, "required": ["phrases"]}},
    {"name": "search_jobs", "description": "Semantic search over the job index. Returns up to k candidate postings with job_id, title, company, category, similarity score, ESCO skills and a snippet. Call it 2-4 times with different queries (titles, stack, domain).",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}, "k": {"type": "integer", "default": 10, "minimum": 1, "maximum": 50},
                                                       "filters": {"type": "object", "description": "Optional metadata filters, e.g. {\"category\": \"Python\"} or {\"remote\": true}.", "additionalProperties": True}}, "required": ["query"]}},
    {"name": "get_job", "description": "Full text and extracted skills of one posting. Output: {job_id, title, company, category, text, esco_skills, raw_skills}.",
     "input_schema": {"type": "object", "properties": {"job_id": {"type": "string"}}, "required": ["job_id"]}},
    {"name": "compare_skills", "description": "Compare two skill lists (ESCO labels or raw phrases). Output: {matched: [...], missing: [...], extra: [...]} where missing = in the job but not in the resume.",
     "input_schema": {"type": "object", "properties": {"resume_skills": {"type": "array", "items": {"type": "string"}}, "job_skills": {"type": "array", "items": {"type": "string"}}}, "required": ["resume_skills", "job_skills"]}},
]


def _norm(s: str) -> str:
    return " ".join(str(s).lower().replace("-", " ").split())


def compare_skill_lists(resume_skills: list[str], job_skills: list[str], fuzzy_threshold: int = 90) -> dict[str, list[str]]:
    r = {_norm(s): s for s in resume_skills if s}
    j = {_norm(s): s for s in job_skills if s}
    matched, missing = [], []
    for jk, jv in j.items():
        if jk in r:
            matched.append(jv)
            continue
        best = max(((fuzz.token_set_ratio(jk, rk), rk) for rk in r), default=(0, None))
        if best[0] >= fuzzy_threshold:
            matched.append(jv)
        else:
            missing.append(jv)
    extra = [rv for rk, rv in r.items() if rk not in j and not any(fuzz.token_set_ratio(rk, jk) >= fuzzy_threshold for jk in j)]
    return {"matched": matched, "missing": missing, "extra": extra}


class ToolBox:
    def __init__(self, index_name: str = "djinni_jobs", pool_ids: list[str] | None = None, extractor: SkillExtractor | None = None, normalizer=None, snippet_chars: int = 300, job_text_chars: int = 3500):
        self.index = JobIndex(index_name)
        self.pool_ids = list(pool_ids) if pool_ids else None
        self._extractor = extractor
        self._normalizer = normalizer
        self.snippet_chars, self.job_text_chars = snippet_chars, job_text_chars
        self.calls: list[dict[str, Any]] = []
        self._pool_cache: dict[str, Any] | None = None

    @property
    def extractor(self) -> SkillExtractor:
        if self._extractor is None:
            self._extractor = SkillExtractor()
        return self._extractor

    @property
    def normalizer(self):
        if self._normalizer is None:
            self._normalizer = get_normalizer()
        return self._normalizer

    # ------------------------------------------------------------------ tools
    def extract_skills(self, text: str) -> dict[str, Any]:
        return self.extractor.extract(str(text)[:6000])

    def normalize_skills(self, phrases: list[str]) -> dict[str, Any]:
        out = self.normalizer.normalize([str(p) for p in phrases][:120])
        out["mapped"] = [{"phrase": m["phrase"], "esco_label": m["esco_label"], "score": m["score"]} for m in out["mapped"]]
        return out

    def _pool(self) -> dict[str, Any]:
        if self._pool_cache is None:
            res = self.index.col.get(ids=self.pool_ids, include=["metadatas", "documents", "embeddings"])
            self._pool_cache = {"ids": res["ids"], "metas": res["metadatas"], "docs": res["documents"], "emb": np.array(res["embeddings"])}
        return self._pool_cache

    def search_jobs(self, query: str, k: int = 10, filters: dict | None = None) -> list[dict[str, Any]]:
        k = max(1, min(int(k or 10), 50))
        if self.pool_ids:
            pool = self._pool()
            q = np.array(self.normalizer.embed([str(query)])[0])
            sims = pool["emb"] @ q
            order = np.argsort(-sims)[:k]
            hits = [{"job_id": pool["ids"][i], "score": round(float(sims[i]), 4), **(pool["metas"][i] or {}), "text": pool["docs"][i]} for i in order]
        else:
            where = None
            if filters:
                clean = {kk: v for kk, v in filters.items() if isinstance(v, (str, int, float, bool)) and kk in ("category", "source", "company", "remote", "employment_type")}
                where = clean if len(clean) == 1 else ({"$and": [{kk: v} for kk, v in clean.items()]} if clean else None)
            try:
                hits = self.index.search(str(query), k=k, where=where)
            except Exception:  # noqa: BLE001  (bad filter → unfiltered)
                hits = self.index.search(str(query), k=k)
        return [{"job_id": h["job_id"], "title": h.get("title"), "company": h.get("company"), "category": h.get("category"), "score": h.get("score"),
                 "esco_skills": split_skills(h.get("esco_skills"))[:15], "snippet": (h.get("text") or "")[: self.snippet_chars]} for h in hits]

    def get_job(self, job_id: str) -> dict[str, Any]:
        if self.pool_ids and job_id not in self.pool_ids:
            return {"error": f"job {job_id} is not in the candidate pool"}
        j = self.index.get(str(job_id))
        if not j:
            return {"error": f"job {job_id} not found"}
        return {"job_id": j["job_id"], "title": j.get("title"), "company": j.get("company"), "category": j.get("category"), "url": j.get("url"),
                "text": (j.get("text") or "")[: self.job_text_chars], "esco_skills": split_skills(j.get("esco_skills")), "raw_skills": split_skills(j.get("raw_skills"))[:60]}

    def compare_skills(self, resume_skills: list[str], job_skills: list[str]) -> dict[str, list[str]]:
        return compare_skill_lists([str(s) for s in resume_skills], [str(s) for s in job_skills])

    # --------------------------------------------------------------- dispatch
    def anthropic_tools(self) -> list[dict[str, Any]]:
        return TOOL_SCHEMAS

    def dispatch(self, name: str, inp: dict[str, Any]) -> str:
        t0 = time.perf_counter()
        try:
            fn = getattr(self, name)
            out = fn(**inp)
            err = None
        except TypeError as e:
            out, err = {"error": f"bad arguments for {name}: {e}"}, str(e)
        except Exception as e:  # noqa: BLE001
            out, err = {"error": f"{type(e).__name__}: {str(e)[:200]}"}, str(e)
        s = json.dumps(out, ensure_ascii=False, default=str)
        self.calls.append({"tool": name, "input": {k: (v if len(json.dumps(v, default=str)) < 300 else "…") for k, v in inp.items()}, "ms": round((time.perf_counter() - t0) * 1000, 1), "output_chars": len(s), "error": err})
        return s
