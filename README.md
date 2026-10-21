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
tests/        33 tests (text utils, faithfulness checker, mocked agents)
```

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
cp .env.example .env            # put ANTHROPIC_API_KEY=... in .env (never committed)
python -m pytest tests           # 33 tests, no network needed
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
streamlit run src/ui/app.py                     # paste a resume, pick an approach, see ranked jobs with evidence
python -m src.monitoring.daily_ingest           # refresh the live index (cron: 0 6 * * *)
python -m src.monitoring.drift_report           # ops dashboard + emerging-skills review queue
docker build -t skill-match-copilot . && docker run -p 8000:8000 --env-file .env skill-match-copilot
```

`POST /match` takes `{"resume": "...", "mode": "single_verified|single|multi|baseline", "index": "live_jobs|djinni_jobs"}`
and returns the ranking with quotes, the dropped claims, the faithfulness summary and the usage (calls, tokens, cost).

## Team hand-off

The build was done end to end by one person; the natural split for presenting is **data & EDA** (notebooks 01–05),
**Problem 1** (`src/p1_extract`, REPORT §3), **Problem 2** (`src/p2_normalize`, REPORT §4) and **Problem 3 + ops**
(`src/p3_agent`, `src/api`, `src/monitoring`, REPORT §5–6). Each section's figures are listed in `SLIDES_OUTLINE.md`.

## Privacy, licences, attribution

No real resume is used (the demo resume is synthetic); CV and posting texts never enter git (agent traces and the
labelling sheet are kept locally); `src/utils/pii.py` scrubs e-mails, phones, URLs and handles before any LLM call.
Datasets, models and tools with their licences: [REFERENCES.md](REFERENCES.md). ESCO: *This service uses the ESCO
classification of the European Commission.*
