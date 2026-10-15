"""SkillSpan loader (Zhang et al., NAACL 2022). Primary source: the GitHub copy (kris927b/SkillSpan,
CC BY 4.0) which has train/dev/test JSON with per-sentence `source` (tech/house).

Record schema: idx, tokens, tags_skill, tags_knowledge, source (+ we add: split, sid).
Label scheme (configs/models.yaml → p1_extract.label_scheme):
  merged   : one BIO sequence with types SKILL and KNOW; on a token tagged in both, SKILL wins
  separate : two taggers, one per column
`tag_overlap_stats()` is what decides 'auto' (merge if < 1% of tagged tokens overlap)."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from ..utils.config import ROOT, data_cfg
from ..utils.text import bio_to_spans

SPLITS = {"train": "train", "dev": "dev", "validation": "dev", "test": "test"}
MERGED_LABELS = ["O", "B-SKILL", "I-SKILL", "B-KNOW", "I-KNOW"]


def _json_dir() -> Path:
    return ROOT / data_cfg()["datasets"]["skillspan"]["local_github"] / "data" / "json"


def load_split(split: str) -> list[dict[str, Any]]:
    name = SPLITS[split]
    rows = []
    with open(_json_dir() / f"{name}.json", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            r = json.loads(line)
            r["split"] = name
            r["sid"] = f"{name}-{i}"
            rows.append(r)
    return rows


def load_all() -> dict[str, list[dict[str, Any]]]:
    return {s: load_split(s) for s in ["train", "dev", "test"]}


def norm_tag(t: str, typ: str) -> str:
    """'B' / 'B-Skill' / 'B-SKILL' → 'B-<typ>'; 'O' stays 'O'. The JSON files use bare B/I/O."""
    if t == "O" or not t:
        return "O"
    return f"{t[0].upper()}-{typ}"


def merge_tags(tags_skill: list[str], tags_knowledge: list[str]) -> list[str]:
    """Merged BIO: skill has priority on overlapping tokens; knowledge spans are re-chunked so a
    knowledge span interrupted by a skill span restarts with B-KNOW."""
    out = []
    prev_know = False
    for ts, tk in zip(tags_skill, tags_knowledge):
        if ts != "O":
            out.append("B-SKILL" if ts[0].upper() == "B" else "I-SKILL")
            prev_know = False
        elif tk != "O":
            if tk[0].upper() == "B" or not prev_know:
                out.append("B-KNOW")
            else:
                out.append("I-KNOW")
            prev_know = True
        else:
            out.append("O")
            prev_know = False
    return out


def record_labels(r: dict[str, Any], scheme: str = "merged") -> list[str]:
    if scheme == "merged":
        return merge_tags(r["tags_skill"], r["tags_knowledge"])
    if scheme == "skill":
        return [norm_tag(t, "SKILL") for t in r["tags_skill"]]
    if scheme == "knowledge":
        return [norm_tag(t, "KNOW") for t in r["tags_knowledge"]]
    raise ValueError(scheme)


def gold_phrases(r: dict[str, Any]) -> dict[str, list[str]]:
    """{'skills': [...], 'knowledge': [...]} verbatim phrases for a record."""
    sk = [s[3] for s in bio_to_spans(r["tokens"], [norm_tag(t, "SKILL") for t in r["tags_skill"]])]
    kn = [s[3] for s in bio_to_spans(r["tokens"], [norm_tag(t, "KNOW") for t in r["tags_knowledge"]])]
    return {"skills": sk, "knowledge": kn}


def tag_overlap_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """How often is a token tagged in BOTH columns? Decides merged vs separate."""
    both = sk = kn = tagged_any = 0
    for r in rows:
        for ts, tk in zip(r["tags_skill"], r["tags_knowledge"]):
            s, k = ts != "O", tk != "O"
            sk += s
            kn += k
            both += s and k
            tagged_any += s or k
    return {
        "tokens_tagged_skill": sk,
        "tokens_tagged_knowledge": kn,
        "tokens_tagged_both": both,
        "tokens_tagged_any": tagged_any,
        "overlap_share_of_tagged": (both / tagged_any) if tagged_any else 0.0,
    }


def span_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n_sent = len(rows)
    all_o = sum(1 for r in rows if all(t == "O" for t in r["tags_skill"]) and all(t == "O" for t in r["tags_knowledge"]))
    n_skill = n_know = 0
    len_skill: Counter = Counter()
    len_know: Counter = Counter()
    vocab: set[str] = set()
    n_tok = 0
    for r in rows:
        n_tok += len(r["tokens"])
        vocab.update(t.lower() for t in r["tokens"])
        for typ, s, e, _ in bio_to_spans(r["tokens"], record_labels(r, "merged")):
            if typ == "SKILL":
                n_skill += 1
                len_skill[e - s] += 1
            else:
                n_know += 1
                len_know[e - s] += 1
    return {
        "sentences": n_sent, "tokens": n_tok, "vocab_size": len(vocab), "all_O_sentences": all_o,
        "all_O_share": all_o / n_sent if n_sent else 0.0,
        "skill_spans": n_skill, "knowledge_spans": n_know,
        "skill_span_len": dict(sorted(len_skill.items())), "knowledge_span_len": dict(sorted(len_know.items())),
        "avg_tokens_per_sentence": n_tok / n_sent if n_sent else 0.0,
    }
