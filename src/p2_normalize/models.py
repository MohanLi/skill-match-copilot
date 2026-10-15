"""Problem 2 models plugged into WorkRB's ModelInterface.

- NamedBiEncoder: sentence-transformers encoder from a local path with a readable name (base MiniLM,
  our fine-tuned model, ...). Target (ESCO vocabulary) embeddings are cached per model so the three
  tasks do not re-encode 13.9k labels each time.
- RetrieveThenLLM (challenger): the encoder retrieves the top-k ESCO skills; an LLM (prompt
  prompts/p2_select_v1.yaml) picks the ones the sentence really mentions, which are moved to the top."""
from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from workrb.models.base import ModelInterface

from ..utils.config import ROOT, cpu_threads
from ..utils.llm import LLMClient, load_prompt

_EMB_CACHE: dict[str, torch.Tensor] = {}


def _key(model_path: str, targets: list[str]) -> str:
    h = hashlib.sha256(("\n".join(targets)).encode()).hexdigest()[:16]
    return f"{Path(model_path).name}:{len(targets)}:{h}"


class NamedBiEncoder(ModelInterface):
    def __init__(self, model_path: str, label: str, max_seq_length: int = 64, batch_size: int = 128):
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(cpu_threads())
        self.model_path, self.label = str(model_path), label
        self.model = SentenceTransformer(self.model_path, device="cpu")
        self.model.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.model.eval()

    @property
    def name(self) -> str:
        return self.label

    @property
    def description(self) -> str:
        return f"sentence-transformers bi-encoder ({self.model_path})"

    def encode(self, texts: list[str]) -> torch.Tensor:
        return self.model.encode(texts, convert_to_tensor=True, normalize_embeddings=True, batch_size=self.batch_size, show_progress_bar=False)

    def encode_targets(self, targets: list[str]) -> torch.Tensor:
        k = _key(self.model_path, targets)
        if k not in _EMB_CACHE:
            _EMB_CACHE[k] = self.encode(targets)
        return _EMB_CACHE[k]

    def _compute_rankings(self, queries, targets, query_input_type=None, target_input_type=None) -> torch.Tensor:
        q = self.encode(list(queries))
        t = self.encode_targets(list(targets))
        return q @ t.T

    def _compute_classification(self, texts, targets, input_type, target_input_type=None) -> torch.Tensor:
        return self._compute_rankings(texts, targets, input_type, target_input_type or input_type)

    @property
    def classification_label_space(self):
        return None


class RetrieveThenLLM(NamedBiEncoder):
    def __init__(self, model_path: str, label: str, k: int = 30, llm_model: str = "bulk", task: str = "p2_retrieve_llm", workers: int = 4, boost: float = 10.0, **kw):
        super().__init__(model_path, label, **kw)
        self.k, self.boost, self.workers = k, boost, workers
        self.llm_model = llm_model
        self.client = LLMClient(task=task)
        self.prompt = load_prompt("p2_select_v1")
        self.calls: list[dict] = []

    @property
    def description(self) -> str:
        return f"bi-encoder top-{self.k} then LLM selection ({self.llm_model})"

    def _ask(self, sentence: str, cands: list[str]) -> tuple[set[str], dict]:
        cand_text = "\n".join(f"{i+1}. {c}" for i, c in enumerate(cands))
        try:
            obj, res = self.client.complete_json(self.prompt, model=self.llm_model, max_tokens=300, sentence=sentence, candidates=cand_text)
            sel = obj.get("selected") or []
            chosen = {s for s in sel if isinstance(s, str) and s in cands}
            invalid = [s for s in sel if not (isinstance(s, str) and s in cands)]
            info = {"sentence": sentence, "k": self.k, "selected": sorted(chosen), "invalid": invalid, "none": bool(obj.get("none")), "input_tokens": res.input_tokens, "output_tokens": res.output_tokens, "latency_ms": res.latency_ms, "cost_usd": res.cost_usd, "cached": res.cached, "model": res.model}
        except Exception as e:  # noqa: BLE001
            chosen, info = set(), {"sentence": sentence, "k": self.k, "error": f"{type(e).__name__}: {str(e)[:120]}", "selected": [], "cost_usd": 0.0}
        return chosen, info

    def _compute_rankings(self, queries, targets, query_input_type=None, target_input_type=None) -> torch.Tensor:
        targets = list(targets)
        scores = super()._compute_rankings(queries, targets).clone()
        topk = scores.topk(self.k, dim=1).indices.tolist()
        jobs = [(i, queries[i], [targets[j] for j in topk[i]]) for i in range(len(queries))]
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            outs = list(ex.map(lambda job: self._ask(job[1], job[2]), jobs))
        for (i, _, cands), (chosen, info) in zip(jobs, outs):
            self.calls.append(info)
            for j, c in zip(topk[i], cands):
                if c in chosen:
                    scores[i, j] += self.boost
        return scores

    def dump_calls(self, rel_path: str) -> None:
        p = ROOT / rel_path
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            for c in self.calls:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
