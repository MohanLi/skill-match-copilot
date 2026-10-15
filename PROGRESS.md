# PROGRESS — Skill-Match Copilot build log

**Last updated:** 2026-10-15 10:30 UTC · **Built by:** Claude in a cloud workspace, mirrored to Kevin's Mac
(`~/Desktop/2026 Autumn/NLP/Project/skill-match-copilot`). **This folder on the Mac is the source of truth.**

## How to resume after an interruption

1. Open the Claude chat for this project (or start a new one with the `Project` folder attached).
2. Say: **"Continue the NLP project from PROGRESS.md."**
3. Claude reads this file, `docs/decisions.md`, and `results/` to pick up at **Next action** below.
   If the cloud workspace was recycled, Claude re-creates it from this folder:
   `python3 -m venv venv && pip install -r requirements.txt` + the spaCy wheel (see README), then
   `python scripts/import_transfer.py` to re-import `data/transfer/` (the bundle made by `scripts/download_assets.py`).

## Status by phase

Legend: ✅ done · 🔄 in progress · ⏳ waiting on Kevin · ⬜ not started

| Phase | What | Status | Evidence |
|------:|------|:------:|----------|
| 0 | Repo skeleton, decision log, configs, download script | ✅ | `docs/decisions.md`, `configs/*.yaml`, `scripts/download_assets.py` |
| 1 | Workspace env + GitHub-sourced data, utils (LLM wrapper, PII, text, viz), prompts v1, faithfulness checker (20/20 tests), labelling guideline, synthetic resume | ✅ | `src/utils/`, `prompts/`, `tests/` (31 tests pass) |
| 2 | Import Kevin's download bundle, verify counts, `docs/data_schema.md` | ⏳ | needs `data/transfer/` + `.env` on the Mac |
| 3 | EDA notebooks 01–04 + figures | 🔄 | 01 SkillSpan ✅, 02 ESCO benchmark ✅ (`notebooks/`, `reports/figures/`, `results/eda/*.json`); 03 Djinni and 04 live jobs are written, waiting for data |
| 4 | Problem 1: tagger vs few-shot LLM | 🔄 | code complete + smoke-tested (`src/p1_extract/`); dictionary baseline scored (`results/p1/metrics_summary_test.csv`); training waits for `jobbert-base-cased`; LLM run waits for the key |
| 5 | Problem 2: WorkRB baselines, bi-encoder, retrieve-then-LLM, NIL | 🔄 | code complete (`src/p2_normalize/`); BM25 validation baseline run with the stand-in ESCO (`results/p2/workrb_val.csv`); models wait for MiniLM/ConTeXT + synthetic pairs |
| 6 | Job index + agent tools | 🔄 | code complete (`src/index/`, `src/p3_agent/tools.py`), mocked tests pass; needs Djinni data + models |
| 7 | Problem 3: gold labels, baseline, single vs multi-agent, EDA 05 | 🔄 | agents, labelling, evaluation code complete and wiring-tested with a scripted LLM; waits for index + key |
| 8 | Model Ops: API, UI, ingestion, drift report, architecture doc, Dockerfile | 🔄 | written (`src/api/app.py`, `src/ui/app.py`, `src/monitoring/`, `docs/architecture.md`, `Dockerfile`); smoke tests after models exist |
| 9 | Draft report, slide outline, references, README, verification pass | ⬜ | |

## Next action

As soon as `data/transfer/` and `.env` exist on the Mac: stage the bundle into the workspace → `python scripts/import_transfer.py`
→ `python -m src.data.verify_assets` → re-run `run_workrb --split val --models bm25` with the official ESCO → train the two taggers
(`train_tagger --column skill`, `--column knowledge`, seed 42) and the bi-encoder → EDA 03/04 → index → P1/P2/P3 evaluations.

## Waiting on Kevin

- [ ] Run `scripts/download_assets.py` on the Mac (README → "What Kevin runs") → creates `data/transfer/`
- [ ] Create `.env` with `ANTHROPIC_API_KEY` in the repo root on the Mac

## Decisions taken while building (beyond docs/decisions.md)

- **Label scheme = SEPARATE taggers** (skill, knowledge): 1.7 % of tagged SkillSpan tokens carry both tags, above the plan's 1 % merge threshold (`results/eda/skillspan_summary.json`).
- **EDA 02 finding:** 43 % of TECH and 37 % of HOUSE skill phrases cannot be mapped to ESCO 1.1.0 — the "not in ESCO" detector is a core requirement, not an edge case.
- **WorkRB tasks load the benchmark from the GitHub CSVs** (`src/p2_normalize/workrb_tasks.py`), identical grouping/filtering to WorkRB's own loader; ESCO stand-in from the benchmark label list until the official CSVs arrive (results flagged `esco_source`).
- **Verifier = deterministic quote check + one LLM repair call per flagged job**; anything still unsupported is dropped (`src/p3_agent/multi_agent.py`).

## Milestone log

- 2026-10-15 09:30 UTC — Decisions 1–15 taken; repo skeleton created (Phase 0).
- 2026-10-15 10:00 UTC — Environment installed; EDA 01–02 executed (16 figures); P1/P2 pipelines smoke-tested (Phase 1).
- 2026-10-15 10:30 UTC — P3 agents, index, labelling, evaluation, API/UI/monitoring written; 31 tests pass.
