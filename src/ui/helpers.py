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
            from pypdf import PdfReader

            text = "\n\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(data)).pages)
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
