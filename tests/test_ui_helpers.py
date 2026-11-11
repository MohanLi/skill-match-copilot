from __future__ import annotations

import csv
import io

import pytest

from src.ui.helpers import highlight, read_resume, results_csv, safe_url, search_key


def test_text_resume_preserves_unicode_and_removes_bom():
    assert read_resume('\ufeffPython, SQL\n研究'.encode(), 'resume.txt') == 'Python, SQL\n研究'


def test_docx_resume_keeps_paragraph_and_table_order():
    from docx import Document

    doc = Document()
    doc.add_paragraph('Experience')
    table = doc.add_table(rows=1, cols=2)
    table.cell(0, 0).text = 'Python'
    table.cell(0, 1).text = 'Docker'
    doc.add_paragraph('Education')
    data = io.BytesIO()
    doc.save(data)
    assert read_resume(data.getvalue(), 'resume.docx') == 'Experience\nPython | Docker\nEducation'


def test_pdf_resume_reads_selectable_text():
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    page = writer.add_blank_page(612, 792)
    font = DictionaryObject({NameObject('/Type'): NameObject('/Font'), NameObject('/Subtype'): NameObject('/Type1'),
                             NameObject('/BaseFont'): NameObject('/Helvetica')})
    page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): font})})
    stream = DecodedStreamObject()
    stream.set_data(b'BT /F1 12 Tf 30 700 Td (Python and Docker) Tj ET')
    page[NameObject('/Contents')] = writer._add_object(stream)
    data = io.BytesIO()
    writer.write(data)
    assert read_resume(data.getvalue(), 'resume.pdf') == 'Python and Docker'


@pytest.mark.parametrize('data,name,message', [
    (b'', 'empty.txt', 'No text'),
    (b'hello', 'resume.exe', 'PDF, DOCX'),
    (b'\xff', 'resume.txt', 'UTF-8'),
    (b'x' * 20_001, 'resume.txt', '20,000'),
    (b'x' * (10 * 1024 * 1024 + 1), 'resume.pdf', '10 MB'),
    (b'not a docx', 'resume.docx', "couldn't read"),
])
def test_invalid_resume_has_actionable_error(data, name, message):
    with pytest.raises(ValueError, match=message):
        read_resume(data, name)


def test_highlight_escapes_text_and_does_not_nest_overlapping_quotes():
    text = 'Python & Docker <script>alert(1)</script>\npython'
    rendered = highlight(text, ['Python & Docker', 'Docker', 'Python', 'script'])
    assert '<script>' not in rendered
    assert 'Python &amp; Docker</mark>' in rendered
    assert rendered.count('<mark ') == rendered.count('</mark>') == 4
    assert '<br>' in rendered
    assert '<mark' not in highlight('<mark>', [])


@pytest.mark.parametrize('url,expected', [
    ('https://example.com/job', 'https://example.com/job'), (' http://example.com/job ', 'http://example.com/job'),
    ('javascript:alert(1)', None), ('data:text/html,test', None), ('https://[bad', None), ('/jobs/1', None), (None, None),
])
def test_job_links_require_http_or_https(url, expected):
    assert safe_url(url) == expected


def test_search_key_changes_with_resume_and_settings():
    values = ['resume', 'remote', 'live_jobs', 'single', 5, 'agent']
    original = search_key(*values)
    for i, replacement in enumerate(['edited resume', 'Chicago', 'djinni_jobs', 'baseline', 10, 'another-model']):
        edited = values.copy()
        edited[i] = replacement
        assert search_key(*edited) != original


def test_csv_scores_and_formula_like_titles_are_labelled_safely():
    result = {'mode': 'baseline', 'run': {'ranking': [{'job_id': '1', 'rank': 1, 'fit_score': 75}]},
              'jobs': {'1': {'title': '=HYPERLINK("bad")', 'company': 'ACME', 'url': 'javascript:alert(1)'}}}
    row = list(csv.DictReader(io.StringIO(results_csv(result))))[0]
    assert row['title'].startswith("'=")
    assert row['score_type'] == 'cosine_similarity_x100'
    assert row['url'] == ''


def test_clean_extracted_text_rejoins_word_per_line_paragraphs():
    from src.ui.helpers import clean_extracted_text

    raw = ("EXPERIENCE\nGen AI Engineer Intern | TalkMeUp, Inc.            Jul 2026 - Sep 2026\n"
           "● Owned model evaluation for a PyTorch service that scores produce freshness 0-100 from a photo to\n"
           "automate\n\ngrocery\n\nrefunds;\n\nset\n\nacceptance criteria of a 2% false-accept ceiling and shipped the validator.\n"
           "● Built a RAG system over 40k help-center chunks with GPT-4o-\nmini; cut handle time by 18%.\n\nSKILLS\nProgramming: Python, R, SQL\n")
    text = clean_extracted_text(raw)
    lines = text.splitlines()
    assert "from a photo to automate grocery refunds; set acceptance criteria of a 2% false-accept ceiling and shipped the validator." in text
    assert "GPT-4o-mini; cut handle time by 18%." in text
    assert "Gen AI Engineer Intern | TalkMeUp, Inc.  Jul 2026 - Sep 2026" in lines   # tab-stop gap collapsed, not merged away
    assert lines[0] == "EXPERIENCE" and "SKILLS" in lines and "Programming: Python, R, SQL" in lines
    assert "\n\n\n" not in text


def test_evidence_and_mapping_tables_are_valid_markdown():
    from src.ui.helpers import evidence_table, mapping_table

    matched = [{"skill": "Python", "resume_quote": "Python (expert) | SQL", "job_quote": "Strong Python\nskills"}]
    table = evidence_table(matched, "matched")
    assert table.splitlines()[0] == "| Skill | Your resume says | The posting asks for |"
    assert "Python (expert) \\| SQL" in table and "Strong Python skills" in table   # pipes escaped, newlines collapsed
    assert evidence_table([], "matched") == ""
    missing = evidence_table([{"skill": "JAX", "job_quote": "PyTorch / JAX"}], "missing")
    assert missing.splitlines()[0] == "| Skill to check | The posting asks for |" and "**JAX**" in missing
    profile = {"mapped": [{"phrase": "Airflow", "esco_label": "control air flow", "score": 0.77},
                          {"phrase": "SQL", "esco_label": "SQL", "score": 1.0},
                          {"phrase": "dbt", "esco_label": "NOT_IN_ESCO", "score": 0.61}]}
    rows = mapping_table(profile).splitlines()
    assert rows[2].startswith("| SQL | SQL | 1.00") and rows[-1].startswith("| dbt | — not in ESCO — | 0.61")


def test_job_facts_line_is_readable_and_safe_markdown():
    from datetime import date

    from src.ui.helpers import facts_line, location_label, posted_label, salary_label

    today = date(2026, 11, 10)
    assert posted_label("2026-10-09T22:19:14.701+00:00", today) == "Posted Oct 9 · 32 days ago"
    assert posted_label("2026-10-20T11:35:15.557Z", today) == "Posted Oct 20 · 21 days ago"
    assert posted_label("2021-11-01T00:00:00+02:00", today) == "Posted Nov 1, 2021 · 5 years ago"
    assert posted_label("", today) == "" and posted_label("not a date", today) == ""
    assert salary_label({"salary_min": 160000.0, "salary_max": 210000.0, "salary_currency": "USD"}) == "$160,000–$210,000"
    assert salary_label({"salary_min": 45000, "salary_max": 75000, "salary_currency": "EUR", "salary_period": "year"}) == "€45,000–€75,000 a year"
    assert salary_label({"salary_min": float("nan")}) == "" and salary_label({}) == ""
    job = {"company": "Scribd_inc", "locations": "San Francisco | Toronto | Denver | Chicago", "remote": True, "category": "Engineering",
           "published": "2026-10-09", "salary_min": 160000.0, "salary_max": 210000.0, "salary_currency": "USD"}
    assert location_label(job, ["chicago"]) == "Chicago; San Francisco (+2 more)"          # the place I asked for comes first
    line = facts_line(job, today, ["Chicago"])
    assert ":material/location_on: Chicago; San Francisco (+2 more)" in line and ":material/home_work: Remote" in line
    assert "Scribd\\_inc" in line and "\\$160,000–\\$210,000" in line                        # no stray emphasis, no LaTeX
    assert "Posted Oct 9 · 32 days ago" in line
    djinni = facts_line({"company": "GlobalLogic", "category": "DevOps", "exp_years": "3y", "english_level": "upper"}, today)
    assert "3+ years of experience" in djinni and "English: upper" in djinni and "location_on" not in djinni
    assert facts_line({}) == ""


def test_requirement_badges_and_pool_sentences():
    from src.ui.helpers import pool_sentence, relaxed_sentence, requirement_badges

    badges = requirement_badges([{"ok": True, "label": "Role: AI engineer"}, {"ok": False, "label": "Not Chicago [or] remote"},
                                 {"ok": None, "label": "Location not listed"}], 4, 1)
    assert badges.split(" :")[0] == ":green-badge[:material/check: Role: AI engineer]"
    assert ":red-badge[:material/close: Not Chicago \\[or\\] remote]" in badges and ":gray-badge[:material/help: Location not listed]" in badges
    assert badges.endswith(":gray-badge[4 skills matched] :gray-badge[1 to check]")
    assert requirement_badges(None) == "" and requirement_badges([], 1, 0).startswith(":gray-badge[1 skill matched]")
    pool = {"total": 2000, "meets_all": 158, "meets_location": 580, "meets_role": 587, "relaxed": []}
    assert pool_sentence(pool) == "580 of 2,000 jobs meet the location requirement and 587 match the role; 158 do both."
    assert pool_sentence({"total": 2000, "meets_all": 35, "meets_location": 35, "meets_role": None}) == "35 of 2,000 jobs meet the location requirement."
    assert pool_sentence(None) == "" and relaxed_sentence(pool) == ""
    assert relaxed_sentence({"meets_all": 1, "relaxed": ["role"]}).startswith("Only 1 job meets every requirement, so jobs with a different role")
    assert relaxed_sentence({"meets_all": 0, "relaxed": ["role and location"]}).startswith("No jobs meet every requirement")


def test_skills_summary_counts_jobs_not_mentions():
    from src.ui.helpers import skills_summary, sort_ranking, summary_badges

    ranking = [
        {"job_id": "a", "matched_skills": [{"skill": "Python"}, {"skill": "python"}], "missing_skills": [{"skill": "Kubernetes (EKS)"}, {"skill": "active learning / human-in-the-loop evaluation"}]},
        {"job_id": "b", "matched_skills": [{"skill": "Python"}], "missing_skills": [{"skill": "Kubernetes"}, {"skill": "Spark"}]},
        {"job_id": "c", "matched_skills": [], "missing_skills": [{"skill": "kubernetes"}, {"skill": "Active learning"}, {"skill": ""}]},
    ]
    summary = skills_summary(ranking)
    assert summary["n_jobs"] == 3 and summary["matched"] == [("Python", 2)]              # twice on one card is still one job
    assert summary["missing"] == [("Kubernetes", 3), ("Active learning", 2), ("Spark", 1)]
    assert summary_badges(summary["missing"], 3, "orange", limit=2) == ":orange-badge[Kubernetes · 3 of 3] :orange-badge[Active learning · 2 of 3] +1 more"
    assert summary_badges(summary["missing"], 3, "orange").endswith(":gray-badge[Spark]")       # a one-off skill: no count, no colour
    jobs = {"a": {"published": "2026-10-05"}, "b": {"published": "2026-10-20T11:00:00Z"}, "c": {}}
    assert [e["job_id"] for e in sort_ranking(ranking, jobs, "newest")] == ["b", "a", "c"]
    assert [e["job_id"] for e in sort_ranking(ranking, jobs, "fit")] == ["a", "b", "c"]


def test_skill_profile_groups_put_recognised_skills_first_and_fragments_last():
    from src.ui.helpers import esco_table, group_profile, is_fragment

    profile = {"skills": ["taken", "improve detection precision", "deployed the backend on", "evaluated"],
               "knowledge": ["PyTorch", "IAM )", "Bachelor of Science in", "SQL", "Docker"],
               "mapped": [{"phrase": "SQL", "esco_label": "SQL", "score": 1.0}, {"phrase": "Docker", "esco_label": "dock operations", "score": 0.66},
                          {"phrase": "evaluated", "esco_label": "types of evaluation", "score": 0.77}, {"phrase": "PyTorch", "esco_label": "NOT_IN_ESCO", "score": 0.37},
                          {"phrase": "IAM )", "esco_label": "NOT_IN_ESCO", "score": 0.29}, {"phrase": "taken", "esco_label": "NOT_IN_ESCO", "score": 0.5},
                          {"phrase": "improve detection precision", "esco_label": "NOT_IN_ESCO", "score": 0.49},
                          {"phrase": "deployed the backend on", "esco_label": "NOT_IN_ESCO", "score": 0.56},
                          {"phrase": "Bachelor of Science in", "esco_label": "NOT_IN_ESCO", "score": 0.56}, {"phrase": "pytorch", "esco_label": "NOT_IN_ESCO", "score": 0.3}]}
    groups = group_profile(profile)
    assert [r["phrase"] for r in groups["esco"]] == ["SQL", "evaluated", "Docker"]         # every mapped phrase, strongest first
    assert groups["tools"] == ["PyTorch"] and groups["activities"] == ["improve detection precision"]
    assert set(groups["fragments"]) == {"deployed the backend on", "Bachelor of Science in", "taken", "IAM )"}
    table = esco_table(groups["esco"]).splitlines()
    assert table[2] == "| SQL | SQL | 1.00 · strong |" and table[-1] == "| Docker | dock operations | 0.66 · weak |"
    assert is_fragment("of Google") and is_fragment("stores", "skill") and not is_fragment("stores", "knowledge") and not is_fragment("OpenCV homography")
    assert esco_table([]) == ""


def test_search_key_and_csv_include_the_requirements():
    from src.ui.helpers import saved_csv

    base = ['resume', '', 'live_jobs', 'single', 5, 'agent']
    assert search_key(*base) == search_key(*base, requirements={})
    assert search_key(*base, requirements={'roles': ['AI engineer']}) != search_key(*base)
    assert search_key(*base, requirements={'roles': ['AI engineer'], 'remote_ok': True}) == search_key(*base, requirements={'remote_ok': True, 'roles': ['AI engineer']})
    result = {'mode': 'single', 'run': {'ranking': [{'job_id': '1', 'rank': 1, 'fit_score': 75}]},
              'jobs': {'1': {'title': 'AI Engineer', 'company': 'ACME', 'locations': 'Chicago, IL | Remote - US', 'published': '2026-10-09T22:19:14.701+00:00', 'url': 'https://example.com/1'}},
              'checks': {'1': [{'name': 'role', 'ok': True, 'label': 'Role: AI engineer'}, {'name': 'location', 'ok': False, 'label': 'Not Chicago'}]}}
    row = list(csv.DictReader(io.StringIO(results_csv(result))))[0]
    assert row['location'] == 'Chicago, IL; Remote - US' and row['posted'] == '2026-10-09'
    assert row['requirements'] == 'met: Role: AI engineer; not met: Not Chicago'
    saved = list(csv.DictReader(io.StringIO(saved_csv({'1': {'title': '=cmd', 'company': 'ACME', 'locations': 'Chicago', 'score': 75, 'url': 'javascript:x'}}))))[0]
    assert saved['title'] == "'=cmd" and saved['url'] == '' and saved['location'] == 'Chicago'
