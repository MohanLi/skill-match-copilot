"""Chroma-backed job index (one collection per job pool: djinni_jobs, live_jobs).

Each job is stored once: id, document = cleaned text (title + description), embedding = all-MiniLM-L6-v2 text embedding (src/index/embedder.py)
vector of 'title. description[:1500]', metadata = title, company, category, exp_years, english_level,
published, source, url, esco_skills ('|'-joined), raw_skills ('|'-joined), n_chars."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..utils.config import ROOT

CHROMA_DIR = ROOT / "chroma"


def _client():
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


class JobIndex:
    def __init__(self, name: str, embed_fn=None):
        self.name = name
        self.client = _client()
        self.col = self.client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
        self._embed = embed_fn

    @property
    def embed(self):
        if self._embed is None:
            from .embedder import get_text_embedder

            self._embed = get_text_embedder().embed
        return self._embed

    def count(self) -> int:
        return self.col.count()

    def upsert(self, jobs: list[dict[str, Any]], embeddings: list[list[float]]) -> None:
        self.col.upsert(
            ids=[j["job_id"] for j in jobs],
            documents=[j["text"] for j in jobs],
            embeddings=embeddings,
            metadatas=[{k: (v if isinstance(v, (str, int, float, bool)) else json.dumps(v, ensure_ascii=False)) for k, v in j.items() if k not in ("job_id", "text") and v is not None} for j in jobs],
        )

    def search(self, query: str, k: int = 30, where: dict | None = None, ids: list[str] | None = None) -> list[dict[str, Any]]:
        q = self.embed([query])[0]
        kwargs: dict[str, Any] = {"query_embeddings": [q], "n_results": min(k, max(1, self.count())), "include": ["metadatas", "documents", "distances"]}
        if where:
            kwargs["where"] = where
        if ids:
            kwargs["ids"] = ids
        res = self.col.query(**kwargs)
        out = []
        for jid, meta, doc, dist in zip(res["ids"][0], res["metadatas"][0], res["documents"][0], res["distances"][0]):
            out.append({"job_id": jid, "score": round(1 - float(dist), 4), **(meta or {}), "text": doc})
        return out

    def get(self, job_id: str) -> dict[str, Any] | None:
        res = self.col.get(ids=[job_id], include=["metadatas", "documents"])
        if not res["ids"]:
            return None
        return {"job_id": job_id, **(res["metadatas"][0] or {}), "text": res["documents"][0]}

    def get_many(self, job_ids: list[str]) -> list[dict[str, Any]]:
        res = self.col.get(ids=job_ids, include=["metadatas", "documents"])
        by = {i: {"job_id": i, **(m or {}), "text": d} for i, m, d in zip(res["ids"], res["metadatas"], res["documents"])}
        return [by[i] for i in job_ids if i in by]

    def all_ids(self) -> list[str]:
        return self.col.get(include=[])["ids"]


def split_skills(meta_value: str | None) -> list[str]:
    return [s for s in (meta_value or "").split("|") if s]
