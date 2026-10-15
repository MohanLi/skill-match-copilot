"""Automatic faithfulness checker for agent explanations (Problem 3 metric + the multi-agent Verifier's
deterministic half).

A *matched skill* claim = {"skill", "resume_quote", "job_quote"}. It is SUPPORTED iff
  (a) resume_quote is a verbatim substring of the resume and job_quote of the job text (case- and
      whitespace-insensitive, punctuation at the ends ignored), and
  (b) the claim is about the quoted text: the skill name occurs in one of the two quotes, or the skill is
      among the ESCO skills extracted from BOTH texts (so "PostgreSQL" ↔ "SQL databases" can count when
      both normalize to the same ESCO skill).
A *missing skill* claim = {"skill", "job_quote"}. It is SUPPORTED iff the job_quote is verbatim in the job
text, the skill occurs in the job (quote or text or ESCO set), and the skill does NOT occur in the resume
(neither as text nor as an ESCO skill of the resume).

Hallucination rate = unsupported claims / all claims. Reason codes make failure analysis possible."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Iterable

_WS = re.compile(r"\s+")
_PUNCT_EDGE = re.compile(r"^[\s\"'“”‘’`.,;:!?()\[\]{}<>\-–—*•]+|[\s\"'“”‘’`.,;:!?()\[\]{}<>\-–—*•]+$")


def normalize(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'").replace("–", "-").replace("—", "-")
    s = s.lower()
    s = _WS.sub(" ", s)
    s = _PUNCT_EDGE.sub("", s)
    return s.strip()


def _squash(s: str) -> str:
    """Remove all whitespace so that line-wrapping / bullet differences never break a verbatim check."""
    return re.sub(r"[\s ]+", "", normalize(s))


def quote_in_text(quote: str, text: str, min_chars: int = 3) -> bool:
    q = _squash(quote)
    if len(q) < min_chars:
        return False
    return q in _squash(text)


def _tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9+#.]+", normalize(s)))


def skill_in_text(skill: str, text: str) -> bool:
    """Skill name occurs in the text: squashed substring, or every content token of the skill occurs."""
    sk = normalize(skill)
    if not sk:
        return False
    if _squash(sk) in _squash(text):
        return True
    toks = {t for t in _tokens(sk) if len(t) > 2 and t not in {"and", "the", "with", "for", "skills", "experience"}}
    if not toks:
        return False
    tt = _tokens(text)
    return toks.issubset(tt)


def _in_esco(skill: str, esco: Iterable[str] | None) -> bool:
    if not esco:
        return False
    s = normalize(skill)
    return any(normalize(e) == s for e in esco)


@dataclass
class ClaimCheck:
    kind: str              # 'matched' | 'missing'
    skill: str
    supported: bool
    reasons: list[str] = field(default_factory=list)
    resume_quote: str | None = None
    job_quote: str | None = None


@dataclass
class ClaimReport:
    checks: list[ClaimCheck]

    @property
    def n_claims(self) -> int:
        return len(self.checks)

    @property
    def n_supported(self) -> int:
        return sum(c.supported for c in self.checks)

    @property
    def hallucination_rate(self) -> float:
        return 0.0 if not self.checks else 1 - self.n_supported / self.n_claims

    def summary(self) -> dict[str, Any]:
        reasons: dict[str, int] = {}
        for c in self.checks:
            for r in c.reasons:
                reasons[r] = reasons.get(r, 0) + 1
        return {
            "n_claims": self.n_claims, "n_supported": self.n_supported,
            "n_matched": sum(c.kind == "matched" for c in self.checks),
            "n_matched_supported": sum(c.kind == "matched" and c.supported for c in self.checks),
            "n_missing": sum(c.kind == "missing" for c in self.checks),
            "n_missing_supported": sum(c.kind == "missing" and c.supported for c in self.checks),
            "hallucination_rate": round(self.hallucination_rate, 4), "reasons": reasons,
        }


def check_matched(claim: dict[str, Any], resume: str, job: str, resume_esco: Iterable[str] | None = None, job_esco: Iterable[str] | None = None) -> ClaimCheck:
    skill = str(claim.get("skill", "") or "")
    rq, jq = claim.get("resume_quote") or "", claim.get("job_quote") or ""
    reasons: list[str] = []
    if not rq or not quote_in_text(rq, resume):
        reasons.append("resume_quote_not_found")
    if not jq or not quote_in_text(jq, job):
        reasons.append("job_quote_not_found")
    if not skill:
        reasons.append("empty_skill")
    about_quotes = bool(skill) and (skill_in_text(skill, rq) or skill_in_text(skill, jq))
    esco_both = _in_esco(skill, resume_esco) and _in_esco(skill, job_esco)
    if skill and not (about_quotes or esco_both):
        reasons.append("skill_not_in_quotes")
    return ClaimCheck("matched", skill, supported=not reasons, reasons=reasons, resume_quote=rq, job_quote=jq)


def check_missing(claim: dict[str, Any], resume: str, job: str, resume_esco: Iterable[str] | None = None, job_esco: Iterable[str] | None = None) -> ClaimCheck:
    skill = str(claim.get("skill", "") or "")
    jq = claim.get("job_quote") or ""
    reasons: list[str] = []
    if not jq or not quote_in_text(jq, job):
        reasons.append("job_quote_not_found")
    if not skill:
        reasons.append("empty_skill")
    elif not (skill_in_text(skill, jq) or skill_in_text(skill, job) or _in_esco(skill, job_esco)):
        reasons.append("skill_not_in_job")
    if skill and (skill_in_text(skill, resume) or _in_esco(skill, resume_esco)):
        reasons.append("skill_present_in_resume")
    return ClaimCheck("missing", skill, supported=not reasons, reasons=reasons, job_quote=jq)


def check_explanation(explanation: dict[str, Any], resume: str, job: str, resume_esco: Iterable[str] | None = None, job_esco: Iterable[str] | None = None) -> ClaimReport:
    """explanation = {"matched_skills": [...], "missing_skills": [...]} for ONE job."""
    checks = [check_matched(c, resume, job, resume_esco, job_esco) for c in (explanation.get("matched_skills") or [])]
    checks += [check_missing(c, resume, job, resume_esco, job_esco) for c in (explanation.get("missing_skills") or [])]
    return ClaimReport(checks)
