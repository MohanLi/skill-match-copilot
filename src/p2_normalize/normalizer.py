"""Problem 2 champion as a service: phrase → ESCO skill (or NOT_IN_ESCO) with the fine-tuned bi-encoder
and the threshold tuned by nil_eval.py. ESCO embeddings are cached on disk next to the model."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from ..data import esco
from ..utils.config import ROOT, cpu_threads, models_cfg

NIL = "NOT_IN_ESCO"


class Normalizer:
    def __init__(self, encoder_path: str | None = None, threshold: float | None = None, k: int = 10):
        from sentence_transformers import SentenceTransformer

        cfg = models_cfg()["p2_normalize"]
        torch.set_num_threads(cpu_threads())
        cand = Path(encoder_path) if encoder_path else ROOT / "models" / "p2_biencoder"
        if not cand.exists():
            cand = ROOT / cfg["base_encoder"]
        self.encoder_path = str(cand)
        self.model = SentenceTransformer(self.encoder_path, device="cpu")
        self.model.max_seq_length = cfg["max_seq_length"]
        self.k = k
        self.threshold = threshold if threshold is not None else self._tuned_threshold()
        self.vocab = esco.vocabulary()
        self.V = self._esco_embeddings()

    def _tuned_threshold(self) -> float:
        meta = ROOT / "results" / "p2" / "nil_threshold_sweep.csv.meta.json"
        if meta.exists():
            return float(json.loads(meta.read_text()).get("best_threshold", 0.5))
        return 0.5

    def _esco_embeddings(self) -> torch.Tensor:
        cache = Path(self.encoder_path) / "esco_embeddings.pt" if Path(self.encoder_path).is_dir() and Path(self.encoder_path).is_relative_to(ROOT) else ROOT / ".cache" / f"esco_emb_{Path(self.encoder_path).name}.pt"
        try:
            if cache.exists():
                d = torch.load(cache)
                if d.get("n") == len(self.vocab):
                    return d["emb"]
        except Exception:  # noqa: BLE001
            pass
        emb = self.model.encode(self.vocab, convert_to_tensor=True, normalize_embeddings=True, batch_size=256, show_progress_bar=False)
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"n": len(self.vocab), "emb": emb}, cache)
        return emb

    def map_phrases(self, phrases: list[str]) -> list[dict[str, Any]]:
        phrases = [p for p in phrases if isinstance(p, str) and p.strip()]
        if not phrases:
            return []
        Q = self.model.encode(phrases, convert_to_tensor=True, normalize_embeddings=True, batch_size=128, show_progress_bar=False)
        S = Q @ self.V.T
        top = S.topk(min(self.k, len(self.vocab)), dim=1)
        out = []
        for i, p in enumerate(phrases):
            score = float(top.values[i, 0])
            label = self.vocab[int(top.indices[i, 0])]
            out.append({"phrase": p, "esco_label": label if score >= self.threshold else NIL, "score": round(score, 4), "in_esco": score >= self.threshold,
                        "candidates": [self.vocab[int(j)] for j in top.indices[i].tolist()]})
        return out

    def normalize(self, phrases: list[str]) -> dict[str, Any]:
        mapped = self.map_phrases(phrases)
        esco_skills = sorted({m["esco_label"] for m in mapped if m["in_esco"]})
        return {"mapped": [{k: v for k, v in m.items() if k != "candidates"} for m in mapped], "esco_skills": esco_skills, "not_in_esco": [m["phrase"] for m in mapped if not m["in_esco"]]}

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False).tolist()


@lru_cache(maxsize=1)
def get_normalizer() -> Normalizer:
    return Normalizer()
