"""Job facts stored next to each indexed posting: company, locations, remote, employment type, salary.

The live feed carries these fields; the first index build kept only the title, department and date, so a card
could not show where a job is and a "remote or Chicago" requirement had nothing to filter on.

  python -m src.index.job_facts                  # add the facts to the existing live_jobs index

The command changes metadata only — texts and embeddings are untouched, nothing is re-embedded — and rewrites
data/processed/live_jobs.parquet. build_job_index stores the same facts for newly indexed postings."""
from __future__ import annotations

import argparse
import math
import re
from typing import Any, Mapping

FACT_KEYS = ("company", "locations", "remote", "employment_type", "salary_min", "salary_max", "salary_currency", "salary_period")
LOCATION_SEP = " | "


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def company_name(company: Any, board: Any = None) -> str:
    """The feed has no company column, only the employer's job-board name ('openai', 'epidemic-sound',
    'comcast/wd115/comcast_careers', 'Grindr LLC'). Use a real company name when there is one, else a tidied board name."""
    name = _text(company)
    if name:
        return name[:100]
    slug = _text(board).split("/")[0].strip()
    if not slug:
        return ""
    if " " in slug or any(ch.isupper() for ch in slug):
        return slug[:100]                                   # already a display name
    return " ".join(w.capitalize() for w in re.split(r"[-_]+", slug) if w)[:100]


def employment_type(value: Any) -> str:
    """'FullTime', 'Full-time', 'full_time', 'Salaried, full-time' → 'Full time'; board-specific values ('Variable') → ''."""
    s = re.sub(r"[^a-z]+", " ", _text(value).lower())
    for key, label in (("intern", "Internship"), ("part", "Part time"), ("contract", "Contract"), ("temp", "Temporary"), ("full", "Full time")):
        if key in s:
            return label
    return ""


def salary_period(value: Any) -> str:
    s = _text(value).lower()
    for key, label in (("year", "year"), ("annual", "year"), ("month", "month"), ("week", "week"), ("day", "day"), ("hour", "hour")):
        if key in s:
            return label
    return ""


def remote_flag(value: Any) -> bool | None:
    """True / False when the feed says so, None when it does not (about a quarter of the postings)."""
    if isinstance(value, bool) or type(value).__name__ == "bool_":
        return bool(value)
    s = _text(value).lower()
    if s in ("true", "1", "yes", "remote"):
        return True
    if s in ("false", "0", "no", "onsite", "on-site", "hybrid"):
        return False
    return None


def _amount(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) and x > 0 else None


def facts_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Index metadata for one normalised feed row (src/data/live_jobs.py). Unknown values are left out, never guessed."""
    out: dict[str, Any] = {}
    company = company_name(row.get("company"), row.get("board"))
    if company:
        out["company"] = company
    places = row.get("location_list")
    if places is None or isinstance(places, (str, float)):
        places = [p for p in [_text(places)] if p]
    places = list(dict.fromkeys(_text(p) for p in places if _text(p)))
    if places:
        out["locations"] = LOCATION_SEP.join(places)[:3000]
    remote = remote_flag(row.get("remote"))
    if remote is not None:
        out["remote"] = remote
    emp = employment_type(row.get("employment_type"))
    if emp:
        out["employment_type"] = emp
    lo, hi = _amount(row.get("salary_min")), _amount(row.get("salary_max"))
    if lo or hi:
        out["salary_min"], out["salary_max"] = float(lo or hi), float(hi or lo)
        currency = _text(row.get("salary_currency")).upper()[:8]
        if currency:
            out["salary_currency"] = currency
        period = salary_period(row.get("salary_period"))
        if period:
            out["salary_period"] = period
    return out


def backfill(index_name: str = "live_jobs", batch: int = 500) -> dict[str, Any]:
    """Add the facts to every posting already in the index. Safe to run again: it only overwrites the fact fields."""
    from ..data import live_jobs
    from .store import JobIndex, export_parquet

    index = JobIndex(index_name)
    ids = index.all_ids()
    rows = live_jobs.load_rows(ids)
    facts = {r["job_id"]: facts_from_row(r) for r in rows.to_dict("records")}
    facts = {job_id: f for job_id, f in facts.items() if f}
    keys = list(facts)
    for b in range(0, len(keys), batch):
        chunk = keys[b : b + batch]
        index.col.update(ids=chunk, metadatas=[facts[i] for i in chunk])
    table = export_parquet(index, index_name)
    n = len(ids)
    have = lambda key: int(sum(1 for f in facts.values() if key in f))  # noqa: E731
    return {"index": index_name, "indexed": n, "found_in_feed": len(rows), "updated": len(keys), "with_company": have("company"), "with_location": have("locations"),
            "remote_known": have("remote"), "remote_true": int(sum(1 for f in facts.values() if f.get("remote") is True)), "with_salary": have("salary_min"),
            "with_employment_type": have("employment_type"), "table_columns": list(table.columns)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--index", default="live_jobs")
    args = ap.parse_args()
    s = backfill(args.index)
    n = max(1, s["indexed"])
    print(f"{s['index']}: job facts added to {s['updated']:,} of {s['indexed']:,} indexed postings ({s['found_in_feed']:,} found in the feed snapshot)")
    print(f"  company {s['with_company']:,} · location {s['with_location']:,} · remote flag known {s['remote_known']:,} (remote: {s['remote_true']:,})"
          f" · employment type {s['with_employment_type']:,} · salary {s['with_salary']:,}")
    if s["updated"] < 0.9 * n:
        print("  WARNING: fewer than 90% of the indexed postings were found in data/raw/live_jobs — is this the snapshot the index was built from?")


if __name__ == "__main__":
    main()
