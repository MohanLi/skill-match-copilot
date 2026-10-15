"""PII scrubbing for resumes / CVs before any LLM call (decision #6/#7 in docs/decisions.md).

Replaces e-mail addresses, phone numbers and URLs with typed placeholders and returns counts so the
scrubbing itself can be reported. Deterministic, dependency-free."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", re.I)
URL_RE = re.compile(r"(?:https?://|www\.)[^\s)<>\"']+|(?:linkedin\.com|github\.com|gitlab\.com|behance\.net|kaggle\.com)/[^\s)<>\"']+", re.I)
# international / US / UA style phone numbers: +380 (67) 123-45-67, +1 312-555-0199, 067 123 4567 ...
PHONE_RE = re.compile(
    r"(?<![\w/])(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)\d{2,4}[\s.-]?\d{2,4}(?:[\s.-]?\d{2,4})?(?![\w/])"
)
HANDLE_RE = re.compile(r"(?:^|\s)@[A-Za-z0-9_]{3,}", re.M)


@dataclass
class ScrubResult:
    text: str
    counts: dict[str, int] = field(default_factory=dict)


def _looks_like_phone(s: str) -> bool:
    digits = re.sub(r"\D", "", s)
    return 9 <= len(digits) <= 15


def scrub(text: str) -> ScrubResult:
    counts = {"email": 0, "url": 0, "phone": 0, "handle": 0}
    if not text:
        return ScrubResult(text or "", counts)

    def _email(m):
        counts["email"] += 1
        return "[EMAIL]"

    def _url(m):
        counts["url"] += 1
        return "[URL]"

    def _phone(m):
        s = m.group(0)
        if not _looks_like_phone(s):
            return s
        counts["phone"] += 1
        return "[PHONE]"

    def _handle(m):
        counts["handle"] += 1
        return m.group(0)[: m.group(0).index("@")] + "[HANDLE]"

    out = EMAIL_RE.sub(_email, text)
    out = URL_RE.sub(_url, out)
    out = PHONE_RE.sub(_phone, out)
    out = HANDLE_RE.sub(_handle, out)
    return ScrubResult(out, counts)
