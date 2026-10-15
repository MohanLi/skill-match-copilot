# Decision log

Decisions taken on 2026-10-15 (Kevin + Claude), before any code was written. Each one records the
constraint that drove it so a teammate can revisit it with the same facts.

| # | Decision | Why |
|---|----------|-----|
| 1 | **Datasets and base models are downloaded on Kevin's Mac** with `scripts/download_assets.py`, then moved into the build workspace as a chunked tar bundle. | huggingface.co and the ESCO site are blocked from the cloud workspace's network; PyPI and GitHub are reachable. |
| 2 | **LLM provider = Anthropic API**, key in `.env` as `ANTHROPIC_API_KEY`. A thin wrapper (`src/utils/llm.py`) keeps the provider swappable, but LiteLLM is not used. | Only `api.anthropic.com` is reachable from the workspace; OpenAI/Groq/Gemini endpoints are blocked. |
| 3 | **LLM budget ≈ $25.** Haiku for bulk work (P1 extractor, P2 selector, both P3 agent arms); a small Sonnet ablation if budget remains. Every call is cached by (prompt version, input hash) and logged with tokens, latency and cost. | Keeps the full test sets affordable; cost is itself a reported metric. |
| 4 | **Fine-tuning runs on CPU in the workspace** (2 vCPU, 7 GB RAM): seed 42 first for every model, extra seeds afterwards. Scripts auto-select cuda/mps/cpu so they can be rerun on a GPU later. | No GPU available; SkillSpan sentences are short so BERT-base trains in ~30 min/seed with dynamic padding. |
| 5 | **Treated as a personal build** although the course team has 4 members; a handoff section in README lets teammates review/present parts. | Kevin plans to do the technical work himself. |
| 6 | **Gold labels for resume–job fit are LLM-made**: 300 pairs, `claude-opus-5-5` as primary rater and `claude-sonnet-5-5` as independent second rater; weighted Cohen's κ reported; Opus adjudicates disagreements with written rationale. The labelling kit (guideline + sheet) is kept so humans can replace the labels later. | No public resume–job labels exist; human labelling time is not available now. The report states this limitation explicitly. |
| 7 | **Demo resume is synthetic** (`data/demo/synthetic_resume.md`). No real resume is ever committed or sent to an LLM. | Privacy; reproducibility of the demo. |
| 8 | **P1 base model = `jjzha/jobbert-base-cased`** (BERT pretrained on job ads by the SkillSpan authors). The model card lists no license; the report notes this. `microsoft/deberta-v3-base` is an optional extra arm (`--with-deberta`). | Domain-matched and CPU-trainable. |
| 9 | **Live jobs**: one daily `open-apply-jobs` snapshot, filtered to data/AI/ML titles, plus a 20k random sample and whole-snapshot summary stats for EDA. Raw posting text stays out of git. | Enables the "this week's jobs" demo and the drift report at small cost. |
| 10 | **Stack**: Anthropic SDK + own wrapper, **LangGraph** for both agent arms, **Chroma** (persistent, local) for ESCO-skill and job-posting vectors, **FastAPI** service, **Streamlit** demo UI, executed **Jupyter notebooks** for EDA. | Fewest moving parts that still read as an industry-standard stack. |
| 11 | **Deliverables beyond code/results**: every figure as PNG (for slides) *and* interactive HTML (Plotly, to explore), a rubric-ordered `REPORT.md` draft with every number traced to a `results/` file, and `SLIDES_OUTLINE.md`. | Kevin writes the final report/slides from these. |
| 12 | **ESCO version 1.1.0** for all evaluation (what WorkRB and the benchmark labels use); the live ESCO release is newer and is noted as a limitation. | Comparable numbers. |
| 13 | **SkillSpan is read from the GitHub copy** (`kris927b/SkillSpan`, CC BY 4.0), which also provides the tech/house source split; the HF copy is used only as a cross-check. | GitHub is reachable now; the source split is needed for the "F1 by source" metric. |
| 14 | **Resume marker**: `PROGRESS.md` in the repo root is updated at every milestone and mirrored to Kevin's Mac together with the repo, so a new session can continue without the old one. | The build workspace is ephemeral; Kevin's folder is the source of truth. |
| 15 | O*NET is **not used** (optional in the plan; its site is blocked and it adds little for the three problems). | Scope. |
