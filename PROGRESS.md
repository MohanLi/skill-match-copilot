# PROGRESS — Skill-Match Copilot build log

**Last updated:** 2026-10-15 09:30 UTC · **Built by:** Claude in a cloud workspace, mirrored to Kevin's Mac
(`~/Desktop/2026 Autumn/NLP/Project/skill-match-copilot`). **This folder on the Mac is the source of truth.**

## How to resume after an interruption

1. Open the Claude chat for this project (or start a new one with the `Project` folder attached).
2. Say: **"Continue the NLP project from PROGRESS.md."**
3. Claude reads this file, `docs/decisions.md`, and `results/` to pick up at **Next action** below.
   If the cloud workspace was recycled, Claude re-creates it from this folder: `pip install -r requirements.txt`,
   then re-imports `data/transfer/` (the bundle made by `scripts/download_assets.py`).

## Status by phase

Legend: ✅ done · 🔄 in progress · ⏳ waiting on Kevin · ⬜ not started

| Phase | What | Status | Evidence |
|------:|------|:------:|----------|
| 0 | Repo skeleton, decision log, configs, download script | 🔄 | `docs/decisions.md`, `configs/*.yaml`, `scripts/download_assets.py` |
| 1 | Workspace env + GitHub-sourced data (SkillSpan, ESCO benchmark CSVs), utils, prompts v1 | ⬜ | |
| 2 | Import Kevin's download bundle, verify counts, `docs/data_schema.md` | ⏳ | needs `data/transfer/` + `.env` |
| 3 | EDA notebooks 01–04 + figures | ⬜ | `notebooks/`, `reports/figures/` |
| 4 | Problem 1: tagger vs few-shot LLM | ⬜ | `results/p1/` |
| 5 | Problem 2: WorkRB baselines, bi-encoder, retrieve-then-LLM, NIL | ⬜ | `results/p2/` |
| 6 | Job index + agent tools | ⬜ | `chroma/`, `src/index/` |
| 7 | Problem 3: gold labels, baseline, single vs multi-agent, EDA 05 | ⬜ | `results/p3/`, `data/labels/` |
| 8 | Model Ops: API, UI, ingestion, drift report, architecture doc | ⬜ | `src/api/`, `src/ui/`, `docs/architecture.md` |
| 9 | Draft report, slide outline, references, README, verification pass | ⬜ | `REPORT.md`, `SLIDES_OUTLINE.md` |

## Next action

Phase 1: install the environment, copy the GitHub-sourced data into `data/raw/github/`, write `src/utils`
(LLM wrapper, PII scrubber), prompts v1, the synthetic demo resume, the labelling guideline, and the
faithfulness checker with its 20 unit cases. Then start EDA 01 (SkillSpan) and 02 (ESCO benchmark), which
need no Hugging Face assets.

## Waiting on Kevin

- [ ] Run `scripts/download_assets.py` on the Mac (see README → "What Kevin runs") → creates `data/transfer/`
- [ ] Create `.env` with `ANTHROPIC_API_KEY` in the repo root on the Mac

## Milestone log

- 2026-10-15 09:30 UTC — Decisions 1–15 taken (`docs/decisions.md`); repo skeleton created.
