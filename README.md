# Skill-Match Copilot

*ADSP 32018 — NLP and Agentic AI (UChicago, Autumn 2026) · final project*

A career copilot that reads a resume and job postings, **extracts** the skills, **normalizes** them to the
ESCO taxonomy, then **ranks jobs and explains** what matches and what is missing — with every claim backed
by quoted text and every step measured on a public benchmark or a labelled test set.

| Step | NLP problem | Champion | Challenger | Ground truth |
|------|-------------|----------|------------|--------------|
| 1. Extract skill phrases | Span tagging (NER-style) | Fine-tuned JobBERT token classifier | Few-shot Claude extractor | SkillSpan (expert BIO tags) |
| 2. Normalize to ESCO | Retrieval / entity linking + reject option | ConTeXT bi-encoder (TechWolf) + tuned cosine threshold; in-house MiniLM fine-tuned on synthetic pairs as the owned alternative | Retrieve-then-LLM re-ranking / LLM reject decision | Skill-Extraction-benchmark via WorkRB |
| 3. Match & explain | Tool-using agent | Single ReAct agent (LangGraph) | Planner → Retriever → Matcher → Verifier | 300 LLM-adjudicated resume–job pairs |

Build status and how to resume an interrupted build: **[PROGRESS.md](PROGRESS.md)**. Design decisions and
the constraints behind them: **[docs/decisions.md](docs/decisions.md)**.

## Quick start (full environment)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
cp .env.example .env   # then put your ANTHROPIC_API_KEY in .env
```

## What Kevin runs (one-time, on the Mac)

The build workspace cannot reach huggingface.co, so the datasets and base models are fetched here:

```bash
cd "$HOME/Desktop/2026 Autumn/NLP/Project/skill-match-copilot"
python3 -m venv .venv-download && source .venv-download/bin/activate
pip install -r scripts/requirements-download.txt
python scripts/download_assets.py          # ~3 GB; resumable; ends with "ALL DONE"
```

Then create the key file (never committed):

```bash
echo "ANTHROPIC_API_KEY=sk-ant-PASTE-YOUR-KEY" > "$HOME/Desktop/2026 Autumn/NLP/Project/skill-match-copilot/.env"
```

## Repository layout

```
configs/      dataset + model + LLM configuration (YAML)
prompts/      versioned prompts (p1_extract_v1.yaml, p2_select_v1.yaml, p3_*.yaml, label_fit_v1.yaml)
scripts/      download_assets.py (runs on the Mac), helper scripts
data/         raw/ (git-ignored downloads) · processed/ · labels/ (guideline, pairs, LLM labels) · demo/
notebooks/    01_eda_skillspan … 05_eda_generated (executed, outputs embedded)
src/          utils/ data/ p1_extract/ p2_normalize/ p3_agent/ index/ api/ ui/ monitoring/
results/      p1/ p2/ p3/ metrics and tables (every number in the report traces here)
reports/      figures/ (PNG + interactive HTML, one-line captions in figures/CAPTIONS.md)
docs/         decisions.md · data_schema.md · architecture.md
```

## Licenses and attribution

See [REFERENCES.md](REFERENCES.md). ESCO: *This service uses the ESCO classification of the European Commission.*
