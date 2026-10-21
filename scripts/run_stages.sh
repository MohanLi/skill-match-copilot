#!/usr/bin/env bash
# Resumable stage runner: each stage writes logs/runs/<stage>.log and a marker logs/runs/<stage>.done.
# Re-running skips finished stages. Usage: scripts/run_stages.sh [stage ...]   (default: all stages in order)
set -u
cd "$(dirname "$0")/.."
source /home/claude/env.sh 2>/dev/null || true
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
mkdir -p logs/runs

run_stage() {
  local name="$1"; shift
  if [ -f "logs/runs/$name.done" ]; then echo "skip $name (done)"; return 0; fi
  echo "=== $(date -u +%H:%M:%S) START $name ==="
  "$@" > "logs/runs/$name.log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then touch "logs/runs/$name.done"; echo "=== $(date -u +%H:%M:%S) DONE $name ==="; else echo "=== $(date -u +%H:%M:%S) FAILED $name (rc=$rc) — see logs/runs/$name.log ==="; echo "$name" >> logs/runs/FAILED; fi
  return $rc
}

ALL=(p1_eval p2_val_baselines p2_train_biencoder p2_val_ours p2_val_rtl p2_test_all p2_nil p2_val_rtl_context p2_test_rtl_context p2_nil_context eda03 eda04 index_djinni index_live p3_pairs p3_labels p3_eval p3_eval_sonnet p3_silver eda04_post drift eda05)
STAGES=("$@"); [ ${#STAGES[@]} -eq 0 ] && STAGES=("${ALL[@]}")

for s in "${STAGES[@]}"; do
  case "$s" in
    p1_eval)            run_stage $s python -m src.p1_extract.evaluate --split test ;;
    p2_val_baselines)   run_stage $s python -m src.p2_normalize.run_workrb --split val --models bm25 minilm context --force ;;
    p2_train_biencoder) run_stage $s python -m src.p2_normalize.train_biencoder ;;
    p2_val_ours)        run_stage $s python -m src.p2_normalize.run_workrb --split val --models ours ;;
    p2_val_rtl)         run_stage $s python -m src.p2_normalize.run_workrb --split val --models rtl --k 10 30 50 ;;
    p2_test_all)        run_stage $s python -m src.p2_normalize.run_workrb --split test --models bm25 minilm context ours rtl --k 50 --force ;;
    p2_nil)             run_stage $s python -m src.p2_normalize.nil_eval --llm --encoder models/p2_biencoder --tag p2_biencoder ;;
    p2_val_rtl_context) run_stage $s python -m src.p2_normalize.run_workrb --split val --models rtl_context --k 10 30 50 ;;
    p2_test_rtl_context) run_stage $s python -m src.p2_normalize.run_workrb --split test --models rtl_context --k 50 ;;
    p2_nil_context)     run_stage $s python -m src.p2_normalize.nil_eval --llm ;;
    eda03)              run_stage $s python scripts/nb.py notebooks/src/03_eda_djinni.py ;;
    eda04)              run_stage $s python scripts/nb.py notebooks/src/04_eda_live_jobs.py ;;
    index_djinni)       run_stage $s python -m src.index.build_job_index --pool djinni ;;
    index_live)         run_stage $s python -m src.index.build_job_index --pool live ;;
    p3_pairs)           run_stage $s python -m src.p3_agent.make_pairs ;;
    p3_labels)          run_stage $s python -m src.p3_agent.label_pairs ;;
    p3_eval)            run_stage $s python -m src.p3_agent.evaluate --approaches baseline single multi multi_noverify ;;
    p3_eval_sonnet)     run_stage $s python -m src.p3_agent.evaluate --approaches single multi --model agent_ablation --tag sonnet ;;
    p3_silver)          run_stage $s python -m src.p3_agent.silver_eval --agents-n 60 ;;
    eda04_post)         run_stage $s python scripts/nb.py notebooks/src/04_eda_live_jobs.py ;;
    drift)              run_stage $s python -m src.monitoring.drift_report ;;
    eda05)              run_stage $s python scripts/nb.py notebooks/src/05_eda_generated.py ;;
    *) echo "unknown stage $s" ;;
  esac
done
echo "=== $(date -u +%H:%M:%S) all requested stages processed ==="
