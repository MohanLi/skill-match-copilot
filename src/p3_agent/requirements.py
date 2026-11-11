"""What the candidate is looking for — role, location, remote — as plain checks on job facts.

No model and no LLM: a posting's title and location text either satisfy a requirement or they do not. The same
rules therefore do two jobs: they restrict the job pool *before* any approach ranks it (so a "remote or Chicago"
search can only return such jobs), and they are shown on each result as a check the user can verify by eye.

Rules
- role: a title matches a role when it contains every word of that role ("AI engineer" ⊂ "Senior AI/ML Engineer").
  A few spellings are unified (machine learning = ML, artificial intelligence = AI, GenAI counts as AI); nothing
  else is inferred — list a second role ("machine learning engineer") to widen the search.
- location: any listed place found as whole words in the posting's location text, or — when remote counts — a
  posting flagged remote (optionally only when its location text also mentions a region such as "United States";
  "United States" / "USA" / "US" and "United Kingdom" / "UK" are the only spellings treated as the same place).
- too few matches: the pool is topped up with postings that miss the role, then the location; those are returned
  after the full matches and flagged, never mixed in silently."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

_FILLER = {"role", "roles", "job", "jobs", "position", "positions", "a", "an", "the", "in", "of", "for", "and", "or", "at", "to", "with", "as"}
_PHRASES = (("machine learning", "ml"), ("artificial intelligence", "ai"), ("generative ai", "genai"), ("gen ai", "genai"),
            ("natural language processing", "nlp"), ("large language models", "llm"), ("large language model", "llm"))
# plurals and abbreviations only; "engineering" is deliberately not "engineer" ("Engineering Manager, ML" is not an ML engineer)
_ALIASES = {"genai": ("genai", "ai"), "llms": ("llm",), "engineers": ("engineer",), "scientists": ("scientist",),
            "analysts": ("analyst",), "developers": ("developer",), "researchers": ("researcher",), "sr": ("senior",), "jr": ("junior",)}


def _clean_list(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = re.split(r"[,;\n]", values)
    out = [" ".join(str(v).split()) for v in values if str(v).strip()]
    return list(dict.fromkeys(out))


@dataclass
class Requirements:
    roles: list[str] = field(default_factory=list)          # any of
    locations: list[str] = field(default_factory=list)      # any of
    remote_ok: bool = False                                 # a remote posting satisfies the location requirement
    remote_within: list[str] = field(default_factory=list)  # optional: …only if its location text mentions one of these
    notes: str = ""                                         # free text, read by the agent only

    @classmethod
    def from_any(cls, value: "Requirements | Mapping[str, Any] | None") -> "Requirements":
        if isinstance(value, Requirements):
            return value
        value = value or {}
        return cls(roles=_clean_list(value.get("roles")), locations=_clean_list(value.get("locations")), remote_ok=bool(value.get("remote_ok")),
                   remote_within=_clean_list(value.get("remote_within")), notes=str(value.get("notes") or "").strip())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def has_role(self) -> bool:
        return bool(self.roles)

    @property
    def has_location(self) -> bool:
        return bool(self.locations) or self.remote_ok

    @property
    def filters(self) -> bool:
        return self.has_role or self.has_location

    def location_text(self) -> str:
        remote = ("remote within " + " / ".join(self.remote_within)) if self.remote_within else "remote"
        if self.locations and self.remote_ok:
            return " or ".join(self.locations) + ", or " + remote
        if self.locations:
            return " or ".join(self.locations)
        return remote if self.remote_ok else ""

    def describe(self) -> str:
        """One line for the results header: 'AI engineer or ML engineer · Chicago, or remote within United States'."""
        parts = [" or ".join(self.roles)] if self.roles else []
        if self.has_location:
            parts.append(self.location_text())
        return " · ".join(parts)

    def to_prompt(self, pool: Mapping[str, Any] | None = None) -> str:
        """The agent's 'Preferences:' line. The pool is already filtered, so the agent is told not to re-check it."""
        parts = []
        if self.roles:
            parts.append("Target role: " + " or ".join(self.roles) + ".")
        if self.has_location:
            parts.append("Location: " + self.location_text() + ".")
        if self.notes:
            parts.append(self.notes if self.notes.endswith((".", "!", "?")) else self.notes + ".")
        if pool and pool.get("relaxed"):
            parts.append("Too few postings meet every requirement, so search_jobs also returns postings that miss some of them; prefer postings that meet all of them.")
            if pool.get("must_include"):
                parts.append("These postings meet every requirement and must come first in the ranking: " + ", ".join(pool["must_include"]) + ".")
        elif pool and pool.get("restricted"):
            parts.append("search_jobs already returns only postings that meet these role and location requirements.")
        return " ".join(parts)


# ----------------------------------------------------------------------------- role
def _words(text: str) -> set[str]:
    s = str(text or "").lower()
    for phrase, short in _PHRASES:
        s = s.replace(phrase, short)
    out: set[str] = set()
    for tok in re.findall(r"[a-z0-9]+[+#]*", s):
        out.update(_ALIASES.get(tok, (tok,)))
    return out


def role_words(role: str) -> set[str]:
    return _words(role) - _FILLER


def match_role(title: str, roles: Iterable[str]) -> str | None:
    """The first requested role whose words all appear in the title, else None."""
    have = _words(title)
    for role in roles:
        need = role_words(role)
        if need and need <= have:
            return role
    return None


# ------------------------------------------------------------------------- location
_US = ("united states", "usa", "us", "u.s.")
_UK = ("united kingdom", "uk", "u.k.", "great britain")
_REGIONS = {**{name: _US for name in _US}, **{name: _UK for name in _UK}}   # the only spellings treated as the same place


def _mentions(text: str, terms: Iterable[str]) -> str | None:
    """The first term that appears in the text as whole words ('US' matches 'Remote - US', not 'Austin')."""
    low = str(text or "").lower()
    for term in terms:
        t = " ".join(str(term).lower().split())
        for spelling in _REGIONS.get(t, (t,)):
            if spelling and re.search(r"(?<![a-z0-9])" + re.escape(spelling) + r"(?![a-z0-9])", low):
                return term
    return None


def location_text(meta: Mapping[str, Any]) -> str:
    return str(meta.get("locations") or meta.get("location") or "")


def is_remote(meta: Mapping[str, Any]) -> bool | None:
    """True when the feed flags the posting remote or its location text says so; False when flagged not remote; else None."""
    flag = meta.get("remote")
    if flag is True or str(flag).lower() == "true" or _mentions(location_text(meta), ["remote"]):
        return True
    if flag is False or str(flag).lower() == "false":
        return False
    return None


def has_location_facts(meta: Mapping[str, Any]) -> bool:
    return bool(location_text(meta)) or meta.get("remote") is not None


def check_location(meta: Mapping[str, Any], req: Requirements) -> dict[str, Any] | None:
    if not req.has_location:
        return None
    if not has_location_facts(meta):
        return {"name": "location", "ok": None, "label": "Location not listed"}
    text = location_text(meta)
    place = _mentions(text, req.locations)
    if place:
        return {"name": "location", "ok": True, "label": place}
    if req.remote_ok and is_remote(meta):
        if not req.remote_within:
            return {"name": "location", "ok": True, "label": "Remote"}
        region = _mentions(text, req.remote_within)
        if region:
            return {"name": "location", "ok": True, "label": f"Remote · {region}"}
        return {"name": "location", "ok": False, "label": "Remote, but not stated as " + " / ".join(req.remote_within)}
    return {"name": "location", "ok": False, "label": "Not " + req.location_text()}


def check_role(meta: Mapping[str, Any], req: Requirements) -> dict[str, Any] | None:
    if not req.has_role:
        return None
    role = match_role(str(meta.get("title") or ""), req.roles)
    return {"name": "role", "ok": bool(role), "label": f"Role: {role}" if role else "Different role"}


def check_job(meta: Mapping[str, Any], req: Requirements) -> list[dict[str, Any]]:
    """The checks to show on a result: [{name, ok (True / False / None = no data), label}] for each stated requirement."""
    return [c for c in (check_role(meta, req), check_location(meta, req)) if c]


def n_failed(checks: Iterable[Mapping[str, Any]]) -> int:
    return sum(1 for c in checks if c.get("ok") is False)


# ----------------------------------------------------------------------------- pool
def select_pool(metas: list[Mapping[str, Any]], req: Requirements, need: int = 5) -> dict[str, Any]:
    """Which postings may be ranked. All of them when nothing is required; otherwise those that meet every requirement,
    topped up — only if fewer than `need` do — with postings that miss the role, then the location, then both."""
    total = len(metas)
    location_data = any(has_location_facts(m) for m in metas)
    use_location = req.has_location and location_data
    tiers: dict[int, list[str]] = {0: [], 1: [], 2: [], 3: []}
    n_location = n_role = 0
    for m in metas:
        role_ok = (not req.has_role) or match_role(str(m.get("title") or ""), req.roles) is not None
        loc_ok = (not use_location) or bool((check_location(m, req) or {}).get("ok"))
        n_location += loc_ok
        n_role += role_ok
        tiers[0 if (role_ok and loc_ok) else 1 if loc_ok else 2 if role_ok else 3].append(str(m["job_id"]))
    ids, relaxed = list(tiers[0]), []
    for tier, missing in ((1, "role"), (2, "location"), (3, "role and location")):
        if len(ids) >= need:
            break
        if tiers[tier]:
            ids += tiers[tier]
            relaxed.append(missing)
    return {"ids": ids, "total": total, "meets_all": len(tiers[0]), "meets_location": n_location if use_location else None, "meets_role": n_role if req.has_role else None,
            "relaxed": relaxed, "must_include": list(tiers[0]) if relaxed else [], "location_data": location_data,
            "restricted": bool((req.has_role or use_location) and len(ids) < total)}


def pool_summary(pool: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The pool without its id list (what a result keeps and the UI shows)."""
    return {k: v for k, v in pool.items() if k != "ids"} if pool else None


def order_by_requirements(ranking: list[dict[str, Any]], checks: Mapping[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Full matches first, in the ranker's own order; postings that miss a requirement after them. Ranks are renumbered."""
    ordered = sorted(ranking, key=lambda e: n_failed(checks.get(e["job_id"], [])))  # sorted() is stable
    for i, e in enumerate(ordered):
        e["rank"] = i + 1
    return ordered
