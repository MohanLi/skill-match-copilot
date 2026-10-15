"""Trivial baseline for Problem 1: dictionary lookup of training-set span phrases (greedy longest match,
case-insensitive). The fine-tuned tagger must clearly beat this (plan: Phase 3 check).
Usage: python -m src.p1_extract.baseline_dict --split test"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict

from ..data import skillspan as ss
from ..utils.config import ROOT
from ..utils.io import write_json
from ..utils.text import bio_to_spans
from .data import LABEL_SPACES, records_with_labels


def build_dictionary(column: str) -> dict[int, set[tuple[str, ...]]]:
    by_len: dict[int, set[tuple[str, ...]]] = defaultdict(set)
    for r in records_with_labels("train", column):
        for typ, s, e, _ in bio_to_spans(r["tokens"], r["labels"]):
            by_len[e - s].add(tuple(t.lower() for t in r["tokens"][s:e]))
    return by_len


def tag_with_dictionary(tokens: list[str], by_len: dict[int, set], typ: str, max_len: int = 10) -> list[str]:
    low = [t.lower() for t in tokens]
    tags = ["O"] * len(tokens)
    i = 0
    while i < len(tokens):
        hit = 0
        for n in range(min(max_len, len(tokens) - i), 0, -1):
            if tuple(low[i : i + n]) in by_len.get(n, ()):
                hit = n
                break
        if hit:
            tags[i] = f"B-{typ}"
            for j in range(i + 1, i + hit):
                tags[j] = f"I-{typ}"
            i += hit
        else:
            i += 1
    return tags


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--columns", nargs="*", default=["skill", "knowledge"])
    args = ap.parse_args()
    for column in args.columns:
        typ = LABEL_SPACES[column][1][2:]
        by_len = build_dictionary(column)
        rows = records_with_labels(args.split, column)
        t0 = time.time()
        out = ROOT / "results" / "p1" / "preds" / f"dict_{column}_seed0_{args.split}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            for r in rows:
                pred = tag_with_dictionary(r["tokens"], by_len, typ)
                f.write(json.dumps({"sid": r["sid"], "source": r["source"], "tokens": r["tokens"], "gold": r["labels"], "pred": pred}) + "\n")
        lat = (time.time() - t0) / len(rows) * 1000
        write_json({"approach": "dict", "column": column, "seed": 0, "split": args.split, "n": len(rows), "latency_ms_per_sentence": round(lat, 3),
                    "dictionary_phrases": int(sum(len(v) for v in by_len.values()))}, f"results/p1/preds/dict_{column}_seed0_{args.split}.meta.json")
        print(f"{column}: {sum(len(v) for v in by_len.values())} phrases, wrote {out.name}")


if __name__ == "__main__":
    main()
