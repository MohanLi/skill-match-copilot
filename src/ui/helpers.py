"""Small input and display helpers; no models or API calls."""
from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import math
import re
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from urllib.parse import urlparse


def read_resume(data: bytes, filename: str) -> str:
    if len(data) > 10 * 1024 * 1024:
        raise ValueError("Please use a resume smaller than 10 MB.")
    suffix = Path(filename).suffix.lower()
    try:
        if suffix in {".txt", ".md"}:
            text = data.decode("utf-8-sig")
        elif suffix == ".pdf":
            text = clean_extracted_text(pdf_text(data))
        elif suffix == ".docx":
            from docx import Document
            from docx.table import Table

            parts = []
            for block in Document(io.BytesIO(data)).iter_inner_content():
                if isinstance(block, Table):
                    parts.extend(" | ".join(cell.text for cell in row.cells) for row in block.rows)
                else:
                    parts.append(block.text)
            text = "\n".join(parts)
        else:
            raise ValueError("Choose a PDF, DOCX, TXT or Markdown file.")
    except UnicodeDecodeError as exc:
        raise ValueError("Please save the text file as UTF-8, or upload a PDF or DOCX.") from exc
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("I couldn't read that file. Try a text-based PDF, DOCX or UTF-8 text file.") from exc
    text = text.strip()
    if not text:
        raise ValueError("No text was found. For a scanned resume, paste the text instead.")
    if len(text) > 20_000:
        raise ValueError("The resume is over 20,000 characters. Please shorten it before searching.")
    return text


def pdf_text(data: bytes) -> str:
    """Text of a PDF, page by page. pdfplumber rebuilds lines from glyph positions, which handles PDFs exported
    from Word (each word placed separately on justified lines); pypdf's layout mode is the fallback."""
    try:
        import pdfplumber

        with pdfplumber.open(io.BytesIO(data)) as pdf:
            pages = [page.extract_text(x_tolerance=1.5, y_tolerance=3) or "" for page in pdf.pages]
    except ImportError:
        from pypdf import PdfReader

        pages = [page.extract_text(extraction_mode="layout") or "" for page in PdfReader(io.BytesIO(data)).pages]
    return "\n\n".join(pages)


_TERMINAL = (".", ":", "!", "?")   # ends a sentence or a heading; ';' and ',' continue one
_BULLETS = ("•", "●", "▪", "◦", "-", "–", "—", "*", "·")


def _is_fragment(line: str) -> bool:
    """A word-per-line leftover: one to three words, no closing punctuation, not a heading or a bullet."""
    words = line.split()
    return 0 < len(words) <= 3 and not line.rstrip().endswith(_TERMINAL) and not line.isupper() and not line.lstrip().startswith(_BULLETS)


def clean_extracted_text(text: str) -> str:
    """Tidy PDF text: collapse the wide gaps that tab stops and columns leave, drop stray blank lines, and glue back
    paragraphs that an extractor split into one word per line ("to\n\nautomate\n\ngrocery\n\nrefunds;")."""
    lines = [re.sub(r"[ \t\u00a0]{2,}", "  ", ln.rstrip()) for ln in text.replace("\r", "").split("\n")]
    out: list[str] = []
    blank_pending = False
    for ln in lines:
        if not ln.strip():
            blank_pending = True
            continue
        ln = ln.strip()
        prev = out[-1] if out else None
        joinable = prev is not None and not prev.rstrip().endswith(_TERMINAL) and not prev.isupper() and not ln.lstrip().startswith(_BULLETS)
        if joinable and (_is_fragment(prev) or ln[:1].islower()):
            # a wrapped or word-per-line continuation, whether or not a blank line sat between; keep hyphenated splits tight
            out[-1] = prev + ln if prev.endswith("-") and ln[:1].islower() else prev + " " + ln
        elif prev is None or not blank_pending:
            out.append(ln)
        else:
            out.append("")                     # keep one blank line between real paragraphs
            out.append(ln)
        blank_pending = False
    cleaned = "\n".join(out)
    cleaned = re.sub(r" {2,}", "  ", cleaned)
    return cleaned.strip()


def highlight(text: str, quotes: list[str], color: str = "#dbeafe") -> str:
    # Match the original text once, then escape it. Replacing inside HTML can break tags.
    phrases = sorted({q for q in quotes if isinstance(q, str) and q.strip()}, key=len, reverse=True)
    if not phrases:
        return html.escape(text).replace("\n", "<br>")
    pattern = re.compile("|".join(re.escape(q) for q in phrases), re.I)
    parts, end = [], 0
    for match in pattern.finditer(text):
        parts.append(html.escape(text[end:match.start()]))
        parts.append(f'<mark style="background:{html.escape(color, quote=True)};padding:0 2px;border-radius:3px">'
                     f'{html.escape(match.group())}</mark>')
        end = match.end()
    parts.append(html.escape(text[end:]))
    return "".join(parts).replace("\n", "<br>")


def _cell(value, limit: int = 220) -> str:
    """One Markdown table cell: no pipes or line breaks, trimmed to a readable length."""
    text = " ".join(str(value or "").split())
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return text.replace("|", "\\|")


def evidence_table(claims: list[dict], kind: str) -> str:
    """Matched claims → | Skill | Your resume says | The posting asks for |; missing claims → | Skill | The posting asks for |."""
    rows = [c for c in claims if isinstance(c, dict)]
    if not rows:
        return ""
    if kind == "matched":
        lines = ["| Skill | Your resume says | The posting asks for |", "|---|---|---|"]
        for c in rows:
            lines.append(f"| **{_cell(c.get('skill'), 60)}** | “{_cell(c.get('resume_quote'))}” | “{_cell(c.get('job_quote'))}” |")
    else:
        lines = ["| Skill to check | The posting asks for |", "|---|---|"]
        for c in rows:
            lines.append(f"| **{_cell(c.get('skill'), 60)}** | “{_cell(c.get('job_quote'))}” |")
    return "\n".join(lines)


def mapping_table(profile: dict) -> str:
    """Resume phrase → ESCO skill table from the normaliser's per-phrase output, strongest matches first."""
    mapped = [m for m in (profile.get("mapped") or []) if isinstance(m, dict)]
    if not mapped:
        return ""
    lines = ["| Phrase in your resume | ESCO skill | Match |", "|---|---|---:|"]
    for m in sorted(mapped, key=lambda m: -(m.get("score") or 0)):
        label = m.get("esco_label") if m.get("esco_label") and m.get("esco_label") != "NOT_IN_ESCO" else "— not in ESCO —"
        lines.append(f"| {_cell(m.get('phrase'), 80)} | {_cell(label, 80)} | {float(m.get('score') or 0):.2f} |")
    return "\n".join(lines)


def safe_url(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = urlparse(value.strip())
    except ValueError:
        return None
    return value.strip() if parsed.scheme in {"http", "https"} and parsed.netloc else None


def search_key(resume: str, preferences: str, index: str, mode: str, top_n: int, model: str, requirements: dict | None = None) -> str:
    values = [resume, preferences, index, mode, top_n, model, requirements or {}]
    return hashlib.sha256(json.dumps(values, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def results_csv(result: dict) -> str:
    output = io.StringIO()
    fields = ["rank", "title", "company", "location", "posted", "score", "score_type", "requirements", "matched_skills", "missing_skills", "url"]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    checks = result.get("checks") or {}
    for item in result["run"]["ranking"]:
        job = result["jobs"].get(item["job_id"], {})
        row = {"rank": item["rank"], "title": job.get("title", item["job_id"]),
               "company": job.get("company", ""), "location": str(job.get("locations") or "").replace(" | ", "; "),
               "posted": str(parse_date(job.get("published")) or ""), "score": item.get("fit_score", 0),
               "score_type": "cosine_similarity_x100" if result["mode"] == "baseline" else "agent_fit_score",
               "requirements": "; ".join(("met: " if c.get("ok") else "not met: " if c.get("ok") is False else "unknown: ") + str(c.get("label", "")) for c in checks.get(item["job_id"], [])),
               "matched_skills": "; ".join(c.get("skill", "") for c in item.get("matched_skills", [])),
               "missing_skills": "; ".join(c.get("skill", "") for c in item.get("missing_skills", [])),
               "url": safe_url(job.get("url")) or ""}
        # Keep job titles beginning with formula characters as text in spreadsheets.
        writer.writerow({k: "'" + v if isinstance(v, str) and v.lstrip().startswith(("=", "+", "-", "@")) else v
                         for k, v in row.items()})
    return output.getvalue()


# ----------------------------------------------------------------------------- job facts on a card
def md(value) -> str:
    """Text that is safe inside st.markdown: no accidental emphasis, links or LaTeX (a '$160,000–$210,000' salary
    would otherwise render as a formula)."""
    return re.sub(r"([\\`*_\[\]$|])", r"\\\1", " ".join(str(value or "").split()))


def parse_date(value) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        m = re.match(r"(\d{4})-(\d{2})-(\d{2})", text)
        try:
            return date(int(m[1]), int(m[2]), int(m[3])) if m else None
        except ValueError:
            return None


def posted_label(value, today: date | None = None) -> str:
    """'2026-10-09T22:19:14.701+00:00' → 'Posted Oct 9 · 32 days ago'."""
    d = parse_date(value)
    if not d:
        return ""
    today = today or date.today()
    days = (today - d).days
    when = f"{d.strftime('%b')} {d.day}" + (f", {d.year}" if d.year != today.year else "")
    if days < 0:
        return f"Posted {when}"
    ago = ("today" if days == 0 else "yesterday" if days == 1 else f"{days} days ago" if days < 60
           else f"{days // 30} months ago" if days < 730 else f"{days // 365} years ago")
    return f"Posted {when} · {ago}"


def _number(value) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) and x > 0 else None


_CURRENCY = {"USD": "$", "EUR": "€", "GBP": "£", "CAD": "CA$", "AUD": "A$"}


def salary_label(job: dict) -> str:
    """The salary range as the feed lists it; the period is added only when the feed states one."""
    lo, hi = _number(job.get("salary_min")), _number(job.get("salary_max"))
    if not lo and not hi:
        return ""
    lo, hi = lo or hi, hi or lo
    code = str(job.get("salary_currency") or "").upper()
    sign = _CURRENCY.get(code, code + " " if code and code != "NAN" else "")
    text = f"{sign}{lo:,.0f}" if lo == hi else f"{sign}{lo:,.0f}–{sign}{hi:,.0f}"
    period = str(job.get("salary_period") or "")
    return text + (f" a {period}" if period in {"year", "month", "week", "day", "hour"} else "")


def location_label(job: dict, prefer=(), limit: int = 2) -> str:
    """Up to `limit` of the posting's places, the ones the user asked for first: 'Chicago, IL; Denver (+6 more)'."""
    places = [p.strip() for p in str(job.get("locations") or "").split(" | ") if p.strip()]
    if not places:
        return ""
    wanted = [str(t).lower() for t in prefer if str(t).strip()]
    places.sort(key=lambda place: 0 if any(t in place.lower() for t in wanted) else 1)   # stable: feed order otherwise
    more = len(places) - limit
    return "; ".join(places[:limit]) + (f" (+{more} more)" if more > 0 else "")


def facts_line(job: dict, today: date | None = None, prefer_locations=()) -> str:
    """One Markdown line of job facts for a card: company · place · remote · department · posting age · salary."""
    parts = []
    if str(job.get("company") or "").strip():
        parts.append(f":material/apartment: {md(job['company'])}")
    place = location_label(job, prefer_locations)
    if place:
        parts.append(f":material/location_on: {md(place)}")
    flag = job.get("remote")
    if (flag is True or str(flag).lower() == "true") and "remote" not in place.lower():
        parts.append(":material/home_work: Remote")
    for key in ("category", "employment_type"):
        if str(job.get(key) or "").strip():
            parts.append(md(job[key]))
    years = str(job.get("exp_years") or "").strip().lower()
    if years == "no_exp":
        parts.append("No experience required")
    elif re.fullmatch(r"\d+y", years):
        parts.append(f"{years[:-1]}+ years of experience")
    english = str(job.get("english_level") or "").strip().lower()
    if english and english != "nan":
        parts.append(f"English: {md(english)}")
    posted = posted_label(job.get("published"), today)
    if posted:
        parts.append(f":material/schedule: {posted}")
    salary = salary_label(job)
    if salary:
        parts.append(f":material/payments: {md(salary)}")
    return "  ·  ".join(parts)


# ------------------------------------------------------------------- requirement checks and counts
def _badge(text: str, color: str, icon: str | None = None) -> str:
    return f":{color}-badge[{f':material/{icon}: ' if icon else ''}{md(text)}]"


def requirement_badges(checks: list[dict] | None, n_matched: int | None = None, n_missing: int | None = None) -> str:
    """Green check / red cross per stated requirement, then grey counts of the evidence-backed skills."""
    out = []
    for c in checks or []:
        if c.get("ok") is True:
            out.append(_badge(c.get("label", ""), "green", "check"))
        elif c.get("ok") is False:
            out.append(_badge(c.get("label", ""), "red", "close"))
        else:
            out.append(_badge(c.get("label", ""), "gray", "help"))
    if n_matched is not None:
        out.append(_badge(f"{n_matched} skill{'s' if n_matched != 1 else ''} matched", "gray"))
    if n_missing is not None:
        out.append(_badge(f"{n_missing} to check", "gray"))
    return " ".join(out)


def pool_sentence(pool: dict | None) -> str:
    """'580 of 2,000 jobs meet the location requirement and 587 match the role; 158 do both.'"""
    if not pool:
        return ""
    total, both = pool.get("total", 0), pool.get("meets_all", 0)
    loc, role = pool.get("meets_location"), pool.get("meets_role")
    if loc is not None and role is not None:
        return f"{loc:,} of {total:,} jobs meet the location requirement and {role:,} match the role; {both:,} do both."
    if loc is not None:
        return f"{loc:,} of {total:,} jobs meet the location requirement."
    if role is not None:
        return f"{role:,} of {total:,} jobs match the role."
    return ""


def relaxed_sentence(pool: dict | None) -> str:
    """Says so when too few jobs met every requirement and others were let in (they are flagged on their cards)."""
    relaxed = (pool or {}).get("relaxed") or []
    if not relaxed:
        return ""
    n = pool.get("meets_all", 0)
    if n == 0:
        return "No jobs meet every requirement, so the closest jobs are shown instead and what each one misses is flagged in red."
    others = {("role",): "jobs with a different role", ("location",): "jobs outside the location"}.get(tuple(relaxed), "jobs that miss a requirement")
    return f"Only {n} job{'s meet' if n != 1 else ' meets'} every requirement, so {others} are listed after {'them' if n != 1 else 'it'} and flagged in red."


def _skill_key(name: str) -> str:
    base = re.sub(r"\([^)]*\)", " ", str(name).lower()).split(" / ")[0]      # 'Kubernetes (EKS)' → 'kubernetes'
    return " ".join(re.sub(r"[^a-z0-9+#.]+", " ", base).split())


def skills_summary(ranking: list[dict]) -> dict:
    """Across the returned jobs: which skills the postings ask for that the resume does not show, and which it shows,
    with the number of jobs each one appears in. Built only from the claims already on the cards."""
    from rapidfuzz import fuzz

    def collect(field: str) -> list[tuple[str, int]]:
        groups: list[dict] = []
        for position, entry in enumerate(ranking):
            for claim in entry.get(field) or []:
                name = " ".join(str((claim or {}).get("skill") or "").split())
                key = _skill_key(name)
                if not key:
                    continue
                group = next((g for g in groups if g["key"] == key or fuzz.ratio(g["key"], key) >= 90), None)
                if group is None:
                    group = {"key": key, "names": Counter(), "jobs": set()}
                    groups.append(group)
                group["names"][name] += 1
                group["jobs"].add(position)
        rows = [(min(g["names"], key=lambda n: (-g["names"][n], len(n))), len(g["jobs"])) for g in groups]
        return sorted(rows, key=lambda row: (-row[1], row[0].lower()))

    return {"n_jobs": len(ranking), "missing": collect("missing_skills"), "matched": collect("matched_skills")}


def summary_badges(rows: list[tuple[str, int]], n_jobs: int, color: str, limit: int = 12) -> str:
    """Skills that recur get their count ('Kubernetes · 3 of 5') and their colour; one-off skills are listed plainly."""
    out = [_badge(f"{name} · {count} of {n_jobs}", color) if count > 1 else _badge(name, "gray") for name, count in rows[:limit]]
    if len(rows) > limit:
        out.append(f"+{len(rows) - limit} more")
    return " ".join(out)


def sort_ranking(ranking: list[dict], jobs: dict, how: str = "fit") -> list[dict]:
    if how != "newest":
        return list(ranking)
    return sorted(ranking, key=lambda e: parse_date(jobs.get(e["job_id"], {}).get("published")) or date.min, reverse=True)   # stable


def saved_csv(saved: dict) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["title", "company", "location", "posted", "score", "url"])
    writer.writeheader()
    for item in saved.values():
        row = {"title": item.get("title", ""), "company": item.get("company", ""), "location": str(item.get("locations") or "").replace(" | ", "; "),
               "posted": str(parse_date(item.get("published")) or ""), "score": item.get("score", ""), "url": safe_url(item.get("url")) or ""}
        writer.writerow({k: "'" + v if isinstance(v, str) and v.lstrip().startswith(("=", "+", "-", "@")) else v for k, v in row.items()})
    return output.getvalue()


# ------------------------------------------------------------------------------- "My skills" tab
_ENDS = {"in", "on", "for", "that", "across", "the", "of", "to", "with", "and", "a", "an", "by", "from", "as", "at", "into", "through", "or", "which", "using", "via"}
_STARTS = {"of", "the", "and", "or", "to", "for", "with", "in", "on", "that", "a", "an"}


def is_fragment(phrase: str, kind: str = "knowledge") -> bool:
    """A tagger leftover rather than a skill: cut off mid-phrase ('Bachelor of Science in', 'deployed the backend on'),
    a stray bracket ('IAM )'), or a lone verb from the skill tagger ('taken', 'stores')."""
    words = str(phrase).split()
    if not words:
        return True
    text = str(phrase).strip()
    if text.count("(") != text.count(")") or text.endswith(("(", ",", "/", "-", "&", ":")):
        return True
    if words[-1].lower().strip(",.;:") in _ENDS or words[0].lower() in _STARTS:
        return True
    return kind == "skill" and len(words) == 1


def group_profile(profile: dict) -> dict:
    """The resume's extracted phrases in the order a reader wants them: ESCO skills (every mapped phrase, because the
    matcher uses all of them), tools and technologies ESCO does not have, activities, and tagger leftovers."""
    knowledge = {str(p).lower() for p in profile.get("knowledge") or []}
    groups: dict[str, list] = {"esco": [], "tools": [], "activities": [], "fragments": []}
    seen = set()
    for m in sorted([m for m in (profile.get("mapped") or []) if isinstance(m, dict)], key=lambda m: -(m.get("score") or 0)):
        phrase = " ".join(str(m.get("phrase") or "").split())
        if not phrase or phrase.lower() in seen:
            continue
        seen.add(phrase.lower())
        if m.get("esco_label") and m.get("esco_label") != "NOT_IN_ESCO":
            groups["esco"].append({"phrase": phrase, "esco_label": m["esco_label"], "score": float(m.get("score") or 0)})
            continue
        kind = "knowledge" if phrase.lower() in knowledge else "skill"
        groups["fragments" if is_fragment(phrase, kind) else "tools" if kind == "knowledge" else "activities"].append(phrase)
    return groups


def esco_table(rows: list[dict]) -> str:
    """| Phrase in your resume | ESCO skill | Match | — the strength label makes a shaky mapping visible."""
    if not rows:
        return ""
    lines = ["| Phrase in your resume | ESCO skill | Match |", "|---|---|---|"]
    for r in rows:
        score = float(r.get("score") or 0)
        strength = "strong" if score >= 0.8 else "good" if score >= 0.7 else "weak"
        lines.append(f"| {_cell(r.get('phrase'), 80)} | {_cell(r.get('esco_label'), 80)} | {score:.2f} · {strength} |")
    return "\n".join(lines)
