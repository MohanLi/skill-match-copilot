"""Text utilities shared by every stage: HTML stripping, sentence splitting, SkillSpan-style
tokenization, and phrase→token alignment (used to score LLM outputs and to build evidence spans)."""
from __future__ import annotations

import html as _html
import re
from typing import Iterable

_BLOCK_TAGS = {"p", "div", "li", "br", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "ul", "ol", "section", "article", "header", "footer", "table"}


def strip_html(raw: str) -> str:
    """HTML → plain text with newlines at block boundaries; safe on plain text input."""
    if not raw:
        return ""
    if "<" not in raw or ">" not in raw:
        return _html.unescape(raw).strip()
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(raw, "lxml")
        for t in soup(["script", "style", "noscript"]):
            t.decompose()
        for tag in soup.find_all(list(_BLOCK_TAGS)):
            tag.insert_before("\n")
            tag.insert_after("\n")
        text = soup.get_text(" ")
    except Exception:  # noqa: BLE001
        text = re.sub(r"<[^>]+>", " ", raw)
        text = _html.unescape(text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


_SENT_END = re.compile(r"(?<=[.!?;])\s+(?=[A-Z0-9(\"'•\-–])")
_BULLET = re.compile(r"^\s*(?:[-–•*·▪●◦]|\d+[.)])\s+")


def split_sentences(text: str, max_tokens: int = 100) -> list[str]:
    """Split by lines/bullets, then by sentence punctuation; long sentences are chunked by tokens.
    Job ads are mostly bullet lists, so newline is the strongest boundary."""
    out: list[str] = []
    for line in text.split("\n"):
        line = _BULLET.sub("", line).strip()
        if not line:
            continue
        for s in _SENT_END.split(line):
            s = s.strip()
            if not s:
                continue
            toks = s.split()
            if len(toks) <= max_tokens:
                out.append(s)
            else:
                for i in range(0, len(toks), max_tokens):
                    out.append(" ".join(toks[i : i + max_tokens]))
    return out


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[+#]+|(?:[-./_'][A-Za-z0-9]+)*)|[^\w\s]", re.UNICODE)


def tokenize(sentence: str) -> list[str]:
    """Word-level tokenization close to SkillSpan's (punctuation split off; keeps C++, C#, node.js, scikit-learn)."""
    return _TOKEN_RE.findall(sentence)


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def find_phrase_spans(tokens: list[str], phrase: str) -> list[tuple[int, int]]:
    """All (start, end_exclusive) token spans whose joined tokens equal `phrase`, case-insensitive,
    ignoring whitespace differences. Used to map LLM phrases back to BIO tags."""
    target = re.sub(r"\s+", "", phrase.lower())
    if not target:
        return []
    ptoks = tokenize(phrase)
    n_p = max(1, len(ptoks))
    spans = []
    low = [t.lower() for t in tokens]
    for i in range(len(tokens)):
        for n in {n_p, n_p - 1, n_p + 1, n_p + 2}:
            if n <= 0 or i + n > len(tokens):
                continue
            if "".join(low[i : i + n]) == target:
                spans.append((i, i + n))
                break
    return spans


def phrases_to_bio(tokens: list[str], phrases: Iterable[str], tag: str, existing: list[str] | None = None) -> tuple[list[str], list[str]]:
    """Mark phrases in a BIO sequence. Returns (tags, unmatched_phrases). Earlier tags win on overlap."""
    tags = list(existing) if existing else ["O"] * len(tokens)
    unmatched = []
    for ph in phrases:
        spans = find_phrase_spans(tokens, ph)
        if not spans:
            unmatched.append(ph)
            continue
        for s, e in spans:
            if all(t == "O" for t in tags[s:e]):
                tags[s] = f"B-{tag}"
                for j in range(s + 1, e):
                    tags[j] = f"I-{tag}"
    return tags, unmatched


def bio_to_spans(tokens: list[str], tags: list[str]) -> list[tuple[str, int, int, str]]:
    """BIO → list of (type, start, end_exclusive, text)."""
    spans = []
    start, typ = None, None
    for i, t in enumerate(list(tags) + ["O"]):
        if t.startswith("B-") or t == "O" or (t.startswith("I-") and (typ is None or t[2:] != typ)):
            if start is not None:
                spans.append((typ, start, i, " ".join(tokens[start:i])))
                start, typ = None, None
            if t.startswith("B-") or t.startswith("I-"):
                start, typ = i, t[2:]
    return spans
