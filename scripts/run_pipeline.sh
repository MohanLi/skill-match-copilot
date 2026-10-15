#!/usr/bin/env bash
# Full build order (what Claude runs in the workspace; also the reproduction recipe).
# Prerequisites: python env from requirements.txt, data/transfer imported (scripts/import_transfer.py), .env with ANTHROPIC_API_KEY.
set -euo pipefail
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false

step() { echo; echo "===== $(date -u +%H:%M:%S) $* ====="; }

step "0. verify assets"
python -m src.data.verify_assets

step "1. EDA"
python scripts/nb.py notebooks/src/01_eda_skillspan.py
python scripts/nb.py notebooks/src/02_eda_esco_benchmark.py
python scripts/nb.py notebooks/src/03_eda_djinni.py
python scripts/nb.py notebooks/src/04_eda_live_jobs.py

step "2. Problem 1 — taggers (seed 42), baseline, LLM extractor, evaluation"
python -m src.p1_extract.train_tagger --column skill --seed 42
python -m src.p1_extract.train_tagger --column knowledge --seed 42
python -m src.p1_extract.baseline_dict --split test
python -m src.p1_extract.llm_extract --split test
python -m src.p1_extract.evaluate --split test

step "3. Problem 2 — baselines, bi-encoder, retrieve-then-LLM, NIL"
python -m src.p2_normalize.run_workrb --split val --models bm25 minilm context
python -m src.p2_normalize.train_biencoder
python -m src.p2_normalize.run_workrb --split val --models ours
python -m src.p2_normalize.run_workrb --split val --models rtl --k 10 30 50
python -m src.p2_normalize.run_workrb --split test --models bm25 minilm context ours rtl --k 30
python -m src.p2_normalize.nil_eval --llm

step "4. Job indexes"
python -m src.index.build_job_index --pool djinni
python -m src.index.build_job_index --pool live
python scripts/nb.py notebooks/src/04_eda_live_jobs.py   # adds the post-pipeline skills section

step "5. Problem 3 — pairs, gold labels, evaluation"
python -m src.p3_agent.make_pairs
python -m src.p3_agent.label_pairs
python -m src.p3_agent.evaluate --approaches baseline single multi multi_noverify
python -m src.p3_agent.silver_eval

step "6. Extra seeds (optional, CPU time permitting)"
for s in 43 44; do python -m src.p1_extract.train_tagger --column skill --seed $s; python -m src.p1_extract.train_tagger --column knowledge --seed $s; done
python -m src.p1_extract.evaluate --split test

step "7. Model Ops artefacts"
python -m src.monitoring.drift_report
python scripts/nb.py notebooks/src/05_eda_generated.py
echo "done"
