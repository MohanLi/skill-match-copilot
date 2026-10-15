"""SkillSpan → token-classification datasets (label alignment, label spaces, chunked inference helpers)."""
from __future__ import annotations

import json
from typing import Any

from ..data import skillspan as ss
from ..utils.config import ROOT

LABEL_SPACES = {
    "skill": ["O", "B-SKILL", "I-SKILL"],
    "knowledge": ["O", "B-KNOW", "I-KNOW"],
    "merged": ["O", "B-SKILL", "I-SKILL", "B-KNOW", "I-KNOW"],
}


def label_scheme_from_eda(default: str = "separate") -> str:
    p = ROOT / "results" / "eda" / "skillspan_summary.json"
    if p.exists():
        return json.loads(p.read_text()).get("label_scheme_decision", default)
    return default


def columns_for_scheme(scheme: str) -> list[str]:
    return ["skill", "knowledge"] if scheme == "separate" else ["merged"]


def records_with_labels(split: str, column: str) -> list[dict[str, Any]]:
    rows = ss.load_split(split)
    for r in rows:
        r["labels"] = ss.record_labels(r, column)
    return rows


def encode_batch(tokenizer, batch: dict[str, list], label2id: dict[str, int], max_length: int = 128) -> dict[str, Any]:
    """HF-style batch encoder: labels on the first sub-token of each word, -100 elsewhere."""
    enc = tokenizer(batch["tokens"], is_split_into_words=True, truncation=True, max_length=max_length)
    all_labels = []
    for i, labs in enumerate(batch["labels"]):
        word_ids = enc.word_ids(batch_index=i)
        prev = None
        out = []
        for wid in word_ids:
            if wid is None:
                out.append(-100)
            elif wid != prev:
                out.append(label2id[labs[wid]])
            else:
                out.append(-100)
            prev = wid
        all_labels.append(out)
    enc["labels"] = all_labels
    return enc
