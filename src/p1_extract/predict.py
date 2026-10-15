"""Inference for the fine-tuned taggers: word-level BIO tags with sliding windows for long sentences,
plus a convenience extractor that turns free text into skill / knowledge phrases (used by the agent
tool `extract_skills` and the job-index builder)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..utils.config import ROOT, cpu_threads, pick_device
from ..utils.text import bio_to_spans, split_sentences, tokenize


class Tagger:
    def __init__(self, model_dir: str, device: str | None = None):
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        self.model_dir = model_dir
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
        self.device = device or pick_device()
        self.model.to(self.device).eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        torch.set_num_threads(cpu_threads())

    @torch.no_grad()
    def tag_batch(self, sentences: list[list[str]], max_length: int = 128, batch_size: int = 32, stride_words: int = 96) -> list[list[str]]:
        """Returns one BIO label per input word. Sentences whose sub-token length exceeds max_length are
        processed in overlapping word windows; each word takes the label from the window where it is
        most central."""
        out: list[list[str]] = [None] * len(sentences)  # type: ignore[list-item]
        # split long sentences into word windows
        items: list[tuple[int, int, list[str]]] = []  # (sent_idx, word_offset, words)
        for i, words in enumerate(sentences):
            n_sub = len(self.tok(words, is_split_into_words=True, add_special_tokens=True)["input_ids"]) if words else 2
            if n_sub <= max_length or len(words) <= stride_words:
                items.append((i, 0, words))
            else:
                start = 0
                while start < len(words):
                    items.append((i, start, words[start : start + stride_words]))
                    start += stride_words // 2
        votes: dict[int, dict[int, tuple[float, str]]] = {}
        for b in range(0, len(items), batch_size):
            chunk = items[b : b + batch_size]
            enc = self.tok([c[2] for c in chunk], is_split_into_words=True, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
            logits = self.model(**{k: v.to(self.device) for k, v in enc.items()}).logits.cpu().numpy()
            pred_ids = logits.argmax(-1)
            for j, (si, off, words) in enumerate(chunk):
                word_ids = enc.word_ids(batch_index=j)
                seen = set()
                for pos, wid in enumerate(word_ids):
                    if wid is None or wid in seen:
                        continue
                    seen.add(wid)
                    centrality = -abs((wid / max(1, len(words))) - 0.5)  # prefer windows where the word is central
                    label = self.id2label[int(pred_ids[j, pos])]
                    d = votes.setdefault(si, {})
                    gw = off + wid
                    if gw not in d or centrality > d[gw][0]:
                        d[gw] = (centrality, label)
        for i, words in enumerate(sentences):
            d = votes.get(i, {})
            labs = [d.get(w, (0, "O"))[1] for w in range(len(words))]
            out[i] = _repair_bio(labs)
        return out

    def extract(self, text: str) -> list[dict[str, Any]]:
        """Free text → list of {type, text, sentence_idx, start, end}."""
        sents = split_sentences(text)
        toks = [tokenize(s) for s in sents]
        toks_nonempty = [(i, t) for i, t in enumerate(toks) if t]
        preds = self.tag_batch([t for _, t in toks_nonempty]) if toks_nonempty else []
        spans = []
        for (i, t), p in zip(toks_nonempty, preds):
            for typ, s, e, phrase in bio_to_spans(t, p):
                spans.append({"type": typ, "text": phrase, "sentence_idx": i, "start": s, "end": e})
        return spans


def _repair_bio(labels: list[str]) -> list[str]:
    """I-X without a preceding B-X/I-X becomes B-X (seqeval-compatible)."""
    out = []
    prev = "O"
    for l in labels:
        if l.startswith("I-") and not (prev.endswith(l[2:]) and prev != "O"):
            l = "B-" + l[2:]
        out.append(l)
        prev = l
    return out


@lru_cache(maxsize=4)
def load_tagger(model_dir: str) -> Tagger:
    return Tagger(model_dir)


class SkillExtractor:
    """Champion extractor = one tagger per column (skill, knowledge), or one merged tagger."""

    def __init__(self, skill_dir: str | None = None, knowledge_dir: str | None = None, merged_dir: str | None = None):
        base = ROOT / "models" / "p1_tagger"
        if merged_dir or (not skill_dir and (base / "merged_seed42").exists() and not (base / "skill_seed42").exists()):
            self.taggers = {"merged": load_tagger(str(merged_dir or base / "merged_seed42"))}
        else:
            self.taggers = {"skill": load_tagger(str(skill_dir or base / "skill_seed42")), "knowledge": load_tagger(str(knowledge_dir or base / "knowledge_seed42"))}

    def extract(self, text: str) -> dict[str, list[str]]:
        skills, knowledge = [], []
        for name, tg in self.taggers.items():
            for sp in tg.extract(text):
                (skills if sp["type"] == "SKILL" else knowledge).append(sp["text"])
        dedup = lambda xs: list(dict.fromkeys(x.strip() for x in xs if x.strip()))
        return {"skills": dedup(skills), "knowledge": dedup(knowledge)}
