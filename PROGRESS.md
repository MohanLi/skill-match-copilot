# PROGRESS — Skill-Match Copilot build log

**Last updated:** 2026-10-21 12:50 UTC · **Built by:** Claude in a cloud workspace, mirrored to Kevin's Mac
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
| 2 | Import Kevin's download bundle, verify counts, `docs/data_schema.md` | ✅ | 10/10 asset checks OK (`results/eda/asset_check.json`, `docs/data_schema.md`) |
| 3 | EDA notebooks 01–04 + figures | ✅ | 01 SkillSpan, 02 ESCO benchmark (re-run with the official ESCO CSVs), 03 Djinni (lean loaders), 04 live jobs (description coverage per ATS) — `notebooks/`, `reports/figures/`, `results/eda/*.json` |
| 4 | Problem 1: tagger vs few-shot LLM | ✅ (seed 42) | test strict F1: tagger skill 0.472 / knowledge 0.632 vs Haiku few-shot 0.285 / 0.530 vs dictionary 0.114 / 0.207 (`results/p1/metrics_summary_test.csv`, figures `p1_*`). Extra seeds + longer training = optional follow-up |
| 5 | Problem 2: WorkRB baselines, bi-encoder, retrieve-then-LLM, NIL | ✅ | test MAP Tech/House/TechWolf: BM25 .275/.160/.177 · MiniLM .365/.246/.293 · in-house fine-tuned MiniLM .424/.313/.374 · **ConTeXT (champion) .536/.423/.460** · RTL on ConTeXT k=50 (challenger) .537/.443/.479 (`results/p2/workrb_test.csv`); NIL threshold 0.64 for ConTeXT (`results/p2/nil.csv`) |
| 6 | Job index + agent tools | ✅ | `djinni_jobs`: 6,000 postings (191 min CPU); `live_jobs`: the 2,000 most recent data/AI/ML postings with text, 5–20 Oct 2026, all 7 ATS sources (`results/index/*_build.json`) |
| 7 | Problem 3: gold labels, baseline, single vs multi-agent, EDA 05 | ✅ | 300 gold labels (Opus + Sonnet, linear κ 0.83, $7.4; `data/labels/`); Haiku eval of 5 arms (`results/p3/metrics_agent.csv`): single ReAct nDCG 0.961 / 17.8 % unsupported claims → **single + verifier 0.961 / 0 %** (champion); planner→matcher→verifier 0.887 / 0 %; Sonnet ablation on 12 CVs; silver 1,000-CV check; EDA 05 executed |
| 8 | Model Ops: API, UI, ingestion, drift report, architecture doc, Dockerfile | ✅ | API smoke-tested end to end on the live index (`logs/requests.jsonl`); `results/monitoring/drift_report.json`, `emerging_skills.csv`, figure `ops_dashboard` |
| 9 | Draft report, slide outline, references, README, verification pass | ✅ | `REPORT.md` (all sections, verified against `results/` by an independent pass — 12 corrections applied), `SLIDES_OUTLINE.md`, README, REFERENCES |

## Next action

**The build is complete.** What remains is Kevin's side: read `REPORT.md` and `SLIDES_OUTLINE.md`, build the slides from
`reports/figures/*.png`, push the repo to GitHub (`.gitignore` already excludes data, models, caches, traces and `.env`).

Optional follow-ups if compute allows (each is one command, results are additive):
- `python -m src.p1_extract.train_tagger --column skill --seed 43` (and 44; and `--epochs 8`): error bars + a stronger tagger.
- `python -m src.p2_normalize.train_biencoder --epochs 3` then `scripts/run_stages.sh p2_val_ours` (remove the `.done` marker first).
- A human pass over `data/processed/labeling_sheet.xlsx` (300 pairs) to validate the LLM labels.

## Waiting on Kevin

Nothing — both inputs arrived 2026-10-20 (bundle verified, key works).

## Decisions taken while building (beyond docs/decisions.md)

- **Label scheme = SEPARATE taggers** (skill, knowledge): 1.7 % of tagged SkillSpan tokens carry both tags, above the plan's 1 % merge threshold (`results/eda/skillspan_summary.json`).
- **EDA 02 finding:** 43 % of TECH and 37 % of HOUSE skill phrases cannot be mapped to ESCO 1.1.0 — the "not in ESCO" detector is a core requirement, not an edge case.
- **WorkRB tasks load the benchmark from the GitHub CSVs** (`src/p2_normalize/workrb_tasks.py`), identical grouping/filtering to WorkRB's own loader; ESCO stand-in from the benchmark label list until the official CSVs arrive (results flagged `esco_source`).
- **Verifier = deterministic quote check + one LLM repair call per flagged job**; anything still unsupported is dropped (`src/p3_agent/multi_agent.py`).

## Decisions taken while building (continued)

- **Claude 5.x API**: `temperature` is rejected ("deprecated for this model") and Haiku 5.5 emits *thinking* blocks by default,
  which silently consumed the output budget (240 parse failures). The wrapper now sends `thinking: disabled` and no sampling
  parameters; reproducibility comes from the on-disk cache. Verified prices: Haiku 5.5 $0.10/$0.50, Sonnet 5.5 $2/$10, Opus 5.5 $4/$20 per MTok.
- **Tagger training loop** replaced the HF `Trainer` (transformers 5 removed `group_by_length`): explicit loop with
  length-bucketed batches, linear warm-up, dev evaluation per epoch, best epoch kept.
- **Live index** = the 4,000 most recent data/AI/ML postings *with a description* (CPU budget), configurable in `configs/models.yaml`.
  Only 53 % of the data/AI/ML postings carry a body (Workday and SmartRecruiters feeds are title-only); Greenhouse bodies arrive as
  escaped HTML and are unescaped by `strip_html`.
- **P2 champion = ConTeXT-Skill-Extraction-base**, chosen on WorkRB validation MAP (0.675/0.548 vs 0.538/0.413 for the in-house
  fine-tuned MiniLM). The in-house model is kept as the owned alternative (`p2_normalize.champion_encoder` switches it in; re-run
  `nil_eval` to re-tune the threshold). LLM re-ranking adds ≤ 0.02 MAP on ConTeXT, so the pure encoder is deployed.
- **Job index embeddings use all-MiniLM-L6-v2 at 256 tokens** (`src/index/embedder.py`), not the 64-token skill-phrase encoder.
- **Claude 5.5 thinking controls differ per model** (`src/utils/llm.thinking_kwargs`): Haiku accepts `disabled`, Sonnet needs
  `between_tools`, Opus cannot switch thinking off (adaptive + low effort). The wrapper also retries once with 3× the output
  budget when thinking consumed it, and salvages truncated JSON (keeps the complete entries).
- **Agents need an 8k-token output budget**: a 10-job ranking with quotes is 4–6k tokens; with 3k the single agent's final JSON
  was truncated in most silver-set runs (first results were garbage and were discarded).
- **Verifier = the faithfulness metric**: its deterministic check now applies exactly the checker's rules (verbatim quotes that
  *name* the skill; a missing skill must be in the job and not in the resume), so "verified" output scores 0 % unsupported by
  construction and the interesting numbers are what it repairs vs drops (1.1 vs 4.6 claims per resume).
- **Gold labels are skewed** (205/82/13 for 0/1/2): P@3 and MRR are reported for "strong" and "partial or better"; nDCG@10
  excludes the 2 pools with no relevant job.
- **Added arm `single_verified`** after the first evaluation: the ReAct agent ranks best, the pipeline is faithful; the verifier is a
  post-processor, so the combination is the product champion.
- **Live index reduced to 2,000 postings**: live postings are 3–4× longer than Djinni ones (~8 s each on 2 vCPUs).
- **EDA 03 needed the lean Djinni loaders** (Arrow strings, text fields merged into one column): the first run hit the 7 GB RAM limit.

## Milestone log

- 2026-10-15 09:30 UTC — Decisions 1–15 taken; repo skeleton created (Phase 0).
- 2026-10-15 10:00 UTC — Environment installed; EDA 01–02 executed (16 figures); P1/P2 pipelines smoke-tested (Phase 1).
- 2026-10-15 10:30 UTC — P3 agents, index, labelling, evaluation, API/UI/monitoring written; 31 tests pass.
- 2026-10-21 01:40 UTC — Data bundle imported and verified; both taggers trained (seed 42); P1 evaluated; stage runner started for P2–P3.
- 2026-10-21 04:15 UTC — P2 complete (champion ConTeXT, challenger RTL, NIL for both encoders); EDA 01–04 all executed; trained
  models mirrored to the Mac (`models/p1_tagger`, `models/p2_biencoder`); Djinni index build restarted with the new embedder.
- 2026-10-21 10:10 UTC — Djinni index built; gold labels, Haiku/Sonnet agent evaluations, silver check and EDA 05 done; REPORT.md
  §3–7 written with final numbers. Live index restarted at 2,000 postings (CPU: ~8 s per long posting).
- 2026-10-21 12:50 UTC — Live index rebuilt with the ISO-8601 date fix (2,000 postings, Oct 5–20); drift report and ops
  dashboard produced; API smoke test passed; REPORT.md verified by an independent pass and corrected; README/SLIDES final.
  **Build complete.** LLM spend $14.3.
