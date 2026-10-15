# Labelling guideline — resume–job fit (Problem 3 evaluation set)

Each row pairs one candidate profile (CV) with one job posting. Give **one label** per pair.
The question is: *if this candidate applied to this job, would a reasonable recruiter consider them a fit?*

| Label | Name | Meaning |
|------:|------|---------|
| **2** | Strong fit | Same role family **and** seniority is compatible **and** most core requirements (stack, domain) are met. The recruiter would shortlist. |
| **1** | Partial fit | Could plausibly apply: same or adjacent role family, but clear gaps — missing a core technology, a seniority mismatch of one level, or a different domain. |
| **0** | Not a fit | Different role family (e.g. a designer for a backend job), seniority far off (junior ↔ lead), or the core stack is absent. |

## How to decide (in this order)

1. **Role family.** Backend / frontend / full-stack / mobile / QA / DevOps / data / ML / PM / design / sales…
   Different family → **0** unless the CV shows substantial experience in the job's family.
2. **Core requirements.** List the 3–6 must-have skills in the posting (languages, frameworks, cloud, domain).
   - ≥ 70 % present in the CV → candidate for **2**
   - 30–70 % present → **1**
   - < 30 % present → **0**
3. **Seniority.** Compare the posting's level (junior / middle / senior / lead) with the CV's years and titles.
   One level apart lowers a 2 to a 1; two or more levels apart → **0**.
4. **Transferable evidence.** Equivalent technologies count (Postgres ≈ MySQL for "SQL databases";
   React ≈ Vue only if the posting says "a modern frontend framework").

## Ignore

- Location, remote policy, salary, English level, company size, benefits, and posting date.
- Formatting, typos and CV length (a short CV with the right stack is still a fit).

## Worked examples

- *CV: 6 years Python/Django/PostgreSQL/AWS backend; Job: Senior Python backend (Django, Postgres, Docker)* → **2**
- *CV: 2 years React/TypeScript frontend; Job: Senior full-stack (React + Node.js, 5+ years)* → **1** (family adjacent, seniority gap, Node missing)
- *CV: Manual QA, 3 years; Job: Data scientist (Python, ML, SQL)* → **0**
- *CV: Data analyst (SQL, Tableau, some Python); Job: ML engineer (PyTorch, MLOps)* → **1** if the CV shows ML projects, else **0**

## Labelling protocol used in this project

Two independent raters label every pair without seeing each other's labels (here: `claude-opus-5-5` as
rater A and `claude-sonnet-5-5` as rater B, prompt `prompts/label_fit_v1.yaml`). Weighted Cohen's κ is
reported. Disagreements are adjudicated by rater A with a written rationale (`data/labels/final.csv`).
Human raters can replace any column by filling `data/labels/labels_human.csv` with the same ids.
