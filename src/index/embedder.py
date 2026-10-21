"""Sentence embeddings for *job texts and search queries* (the Chroma index).

This is deliberately a different model from the Problem-2 skill normaliser: the normaliser encoders are
tuned for short skill phrases (64 tokens), whereas the index must embed a title plus the opening of a posting
and match it against resume summaries, so it uses the general-purpose all-MiniLM-L6-v2 at 256 tokens."""
from __future__ import annotations

from functools import lru_cache

import torch

from ..utils.config import ROOT, cpu_threads, models_cfg


class TextEmbedder:
    def __init__(self, model_path: str | None = None, max_seq_length: int | None = None):
        from sentence_transformers import SentenceTransformer

        cfg = models_cfg()["index"]
        torch.set_num_threads(cpu_threads())
        path = model_path or str(ROOT / cfg.get("text_embedding_model", "models/base/all-MiniLM-L6-v2"))
        self.model_path = path
        self.model = SentenceTransformer(path, device="cpu")
        self.model.max_seq_length = int(max_seq_length or cfg.get("text_max_seq_length", 256))

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode([str(t) for t in texts], normalize_embeddings=True, batch_size=32, show_progress_bar=False).tolist()


@lru_cache(maxsize=1)
def get_text_embedder() -> TextEmbedder:
    return TextEmbedder()
