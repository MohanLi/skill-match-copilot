"""Small input and display helpers; no models or API calls."""
from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import re
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


def search_key(resume: str, preferences: str, index: str, mode: str, top_n: int, model: str) -> str:
    values = [resume, preferences, index, mode, top_n, model]
    return hashlib.sha256(json.dumps(values, ensure_ascii=False).encode()).hexdigest()


def results_csv(result: dict) -> str:
    output = io.StringIO()
    fields = ["rank", "title", "company", "score", "score_type", "matched_skills", "missing_skills", "url"]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for item in result["run"]["ranking"]:
        job = result["jobs"].get(item["job_id"], {})
        row = {"rank": item["rank"], "title": job.get("title", item["job_id"]),
               "company": job.get("company", ""), "score": item.get("fit_score", 0),
               "score_type": "cosine_similarity_x100" if result["mode"] == "baseline" else "agent_fit_score",
               "matched_skills": "; ".join(c.get("skill", "") for c in item.get("matched_skills", [])),
               "missing_skills": "; ".join(c.get("skill", "") for c in item.get("missing_skills", [])),
               "url": safe_url(job.get("url")) or ""}
        # Keep job titles beginning with formula characters as text in spreadsheets.
        writer.writerow({k: "'" + v if isinstance(v, str) and v.lstrip().startswith(("=", "+", "-", "@")) else v
                         for k, v in row.items()})
    return output.getvalue()
