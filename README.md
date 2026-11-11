# Skill-Match Copilot

*ADSP 32018 — NLP and Agentic AI (UChicago MS-ADS, Autumn 2026) · final project · Mohan (Kevin) Li*

A career copilot that reads a resume and job postings, **extracts** the skills, **normalizes** them to the
ESCO taxonomy (or says "not in ESCO"), then **ranks jobs and explains** what matches and what is missing —
with every claim backed by a verbatim quote and every step measured on a public benchmark or a labelled set.

| Step | NLP problem | Champion | Challenger(s) | Ground truth | Headline result |
|------|-------------|----------|---------------|--------------|-----------------|
| 1. Extract skill phrases | Span tagging (BIO) | Fine-tuned JobBERT token classifiers (skill / knowledge) | Few-shot `claude-haiku-5-5`; dictionary baseline | SkillSpan test split (3,569 sentences) | strict F1 0.47 / 0.63 vs 0.29 / 0.53 (LLM); LLM paraphrases 7 % of its "extractions" |
| 2. Normalize to ESCO | Retrieval + reject option | ConTeXT bi-encoder + tuned cosine threshold (0.64) | BM25, zero-shot MiniLM, in-house fine-tuned MiniLM, retrieve-then-LLM | Skill-Extraction benchmark via WorkRB (ESCO 1.1.0) | MAP 0.54 / 0.42 / 0.46 (Tech / House / TechWolf); LLM re-ranking +≤0.02 |
| 3. Match & explain | Tool-using agent | **Single ReAct agent + Verifier** (LangGraph) | Planner→Retriever→Matcher→Verifier; embedding baseline | 300 LLM-graded resume–job pairs (κ = 0.83) | nDCG@10 0.96, **0 % unsupported claims** (18 % without the verifier) |

Full write-up: **[REPORT.md](REPORT.md)** (rubric order, every number traceable to `results/`). Presentation plan:
**[SLIDES_OUTLINE.md](SLIDES_OUTLINE.md)**. Build log / how to resume: **[PROGRESS.md](PROGRESS.md)**. Design decisions
and the constraints behind them: **[docs/decisions.md](docs/decisions.md)**. Deployment and maintenance plan:
**[docs/architecture.md](docs/architecture.md)**.

## What is where

```
REPORT.md · SLIDES_OUTLINE.md · PROGRESS.md · REFERENCES.md
configs/      data.yaml (datasets) · models.yaml (training, champion choices, LLM roles, budget)
prompts/      versioned prompts: p1_extract, p2_select, p2_nil, p3_single_agent, p3_planner/matcher/verifier, label_fit, label_adjudicate
scripts/      download_assets.py (runs on a machine that reaches huggingface.co) · import_transfer.py · run_stages.sh (resumable pipeline)
              nb.py (execute a notebook) · results_tables.py (Markdown tables from results/)
data/         raw/ (downloads, git-ignored) · processed/ (index tables, git-ignored) · labels/ (guideline, pairs, LLM labels, κ) · demo/ (synthetic resume)
notebooks/    01 SkillSpan · 02 ESCO benchmark · 03 Djinni · 04 live postings · 05 generated data — executed .ipynb + html/
src/          utils/ (LLM wrapper, PII, text, viz) · data/ · p1_extract/ · p2_normalize/ · index/ · p3_agent/ · api/ · ui/ · monitoring/
results/      eda/ p1/ p2/ p3/ index/ monitoring/ — every table has a .meta.json (git commit, timestamp, config)
reports/      figures/ — every figure as PNG (slides) and HTML (interactive); captions in figures/CAPTIONS.md
models/       base/ (downloaded) · p1_tagger/{skill,knowledge}_seed42 · p2_biencoder (git-ignored, mirrored separately)
tests/        text utils, faithfulness checker, mocked agents, resume input and UI state
```

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
cp .env.example .env            # put ANTHROPIC_API_KEY=... in .env (never committed)
python -m pytest tests           # no models, API keys or network needed
```

Data and base models (one-time, on a machine that can reach huggingface.co):

```bash
pip install --upgrade pip && pip install --only-binary=:all: -r scripts/requirements-download.txt
python scripts/download_assets.py          # ~3 GB; resumable; ends with "ALL DONE"
python scripts/import_transfer.py          # only when the bundle was moved to another machine
```

## Reproduce the results

`scripts/run_stages.sh` runs every stage in order and skips the ones with a marker in `logs/runs/*.done`:

```
p1_eval → p2_val_baselines → p2_train_biencoder → p2_val_ours → p2_val_rtl → p2_test_all → p2_nil
→ p2_val_rtl_context → p2_test_rtl_context → p2_nil_context → eda03 → eda04 → index_djinni → index_live
→ p3_pairs → p3_labels → p3_eval → p3_eval_verified → p3_eval_sonnet → p3_silver → eda04_post → drift → eda05
```

Training the two taggers (`python -m src.p1_extract.train_tagger --column skill|knowledge`) takes about an hour each
on two CPU cores; the Djinni index takes ~3 h, the live index ~2 h. All LLM calls are cached in `.cache/llm`, so a
re-run of any evaluation costs nothing; the budget guard (`llm.budget_usd`) stops new spend at $25. Total spend for
the project: ≈ $14.

## Run the product

```bash
uvicorn src.api.app:app --port 8000             # /health /extract /normalize /search /match
streamlit run src/ui/app.py                     # resume + requirements in, ranked jobs with evidence out
python -m src.index.job_facts                   # once per existing index: add company / location / remote / salary (seconds, no re-embedding)
python -m src.monitoring.daily_ingest           # refresh the live index (cron: 0 6 * * *)
python -m src.monitoring.drift_report           # ops dashboard + emerging-skills review queue
docker build -t skill-match-copilot . && docker run -p 8000:8000 --env-file .env skill-match-copilot
```

`POST /match` takes `{"resume": "...", "mode": "single_verified|single|multi|baseline", "index": "live_jobs|djinni_jobs"}`
and returns the ranking with quotes, the dropped claims, the faithfulness summary and the usage (calls, tokens, cost).
Add `"requirements": {"roles": ["AI engineer"], "locations": ["Chicago"], "remote_ok": true, "remote_within": ["United States"]}`
to rank only postings that meet them; each result then carries its requirement checks.

**Requirements are filters, not hints** (`src/p3_agent/requirements.py`, `src/p3_agent/match.py`). Role, location and
remote are plain rules on the job facts stored with each posting — a title matches a role when it contains every word
of it, a place is matched as whole words in the posting's location, and a remote job can be limited to a region. The
eligible postings are selected *before* any approach ranks, so a "Chicago or remote" search cannot return anything
else; if fewer postings qualify than were asked for, the list is topped up and those results are flagged in red instead
of being mixed in. Every card shows company, location, remote, posting age and salary where the feed has it, a check
per requirement, and the evidence tables; a box above the cards counts the skills the returned jobs ask for that the
resume does not show. Repeat postings of one job (same company and title, near-identical text — 6.5 % of the live
index) share a card. "More like this" lists the nearest postings inside the same requirements, "Save" keeps a
shortlist for the session, and "My skills" groups the resume's phrases into recognised ESCO skills, tools ESCO does
not list, activities, and tagger leftovers. None of this touches the evaluated approaches: without requirements the
code path, the prompts and therefore the reported numbers are unchanged.

The UI accepts a text-based PDF, DOCX, TXT or Markdown resume, or pasted text (PDF text is rebuilt from glyph positions and re-flowed, so Word-exported PDFs do not arrive one word per line). Results stay on screen while you
inspect evidence, download the CSV or change settings; a notice says when they belong to the previous search. The
semantic baseline only needs the sentence encoder and a populated job index — it does not load the skill taggers,
build a skill profile or call an LLM.

An index built before job facts were stored still works — cards show fewer facts and only the role requirement
applies — until `python -m src.index.job_facts` is run on a machine that has the feed snapshot in `data/raw/live_jobs`
(or the updated `chroma/` folder is copied over).

A fresh clone does not include the trained `models/p1_tagger/` checkpoints or the `chroma/` indexes. Restore them from
the team's local bundle or rebuild them with the commands above; `scripts/download_assets.py` downloads base models and
data but does **not** restore the trained taggers or the completed indexes. The UI lists the missing local files before a
search instead of failing with a model-loading traceback, and shows the full error under "Error details" if a search fails.

## Team hand-off

The build was done end to end by one person; the natural split for presenting is **data & EDA** (notebooks 01–05),
**Problem 1** (`src/p1_extract`, REPORT §3), **Problem 2** (`src/p2_normalize`, REPORT §4) and **Problem 3 + ops**
(`src/p3_agent`, `src/api`, `src/monitoring`, REPORT §5–6). Each section's figures are listed in `SLIDES_OUTLINE.md`.

## Privacy, licences, attribution

No real resume is used (the demo resume is synthetic); CV and posting texts never enter git (agent traces and the
labelling sheet are kept locally); `src/utils/pii.py` scrubs e-mails, phones, URLs and handles before any LLM call.
Datasets, models and tools with their licences: [REFERENCES.md](REFERENCES.md). ESCO: *This service uses the ESCO
classification of the European Commission.*
