"""Phase 1/2 check: verify every dataset is present with the expected counts and write docs/data_schema.md
with the REAL column names (the plan flagged some as unverified).   python -m src.data.verify_assets"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from ..utils.config import ROOT, data_cfg
from . import djinni, esco, esco_benchmark, live_jobs, skillspan


def main() -> None:
    cfg = data_cfg()["datasets"]
    rows: list[dict] = []
    schema: list[str] = ["# Data schema (verified)", "", f"Generated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} by `python -m src.data.verify_assets`.", ""]

    # SkillSpan
    ss = skillspan.load_all()
    counts = {k: len(v) for k, v in ss.items()}
    exp = cfg["skillspan"]["expected_counts"]
    rows.append({"dataset": "SkillSpan (GitHub json)", "status": "OK" if counts == exp else "MISMATCH", "found": counts, "expected": exp})
    schema += ["## SkillSpan (`data/raw/github/SkillSpan/data/json/*.json`)", "", "One JSON object per line: `idx` (int), `tokens` (list[str]), `tags_skill` (B/I/O per token), `tags_knowledge` (B/I/O per token), `source` ('tech' | 'house').", f"Counts: {counts}.", ""]
    hf_ss = ROOT / cfg["skillspan"]["local_hf"]
    if hf_ss.exists():
        files = [str(p.relative_to(hf_ss)) for p in hf_ss.rglob("*") if p.is_file() and ".cache" not in str(p)]
        schema += [f"Hugging Face copy present (`{cfg['skillspan']['local_hf']}`): {files[:12]}", ""]

    # benchmark
    bm = esco_benchmark.load_all()
    rows.append({"dataset": "Skill-Extraction-benchmark (GitHub csv)", "status": "OK" if len(bm) > 3000 else "MISMATCH", "found": {f"{a}/{b}": int(n) for (a, b), n in bm.groupby(["subset", "split"]).size().items()}, "expected": "tech test 1,164 phrases / 425 sentences"})
    schema += ["## Skill-Extraction-benchmark (`data/raw/github/Skill-Extraction-benchmark/data/*.csv`)", "", "Columns: `sentence`, `span`, `sub_span`, `label` (ESCO 1.1.0 preferred label | 'LABEL NOT PRESENT' | 'UNDERSPECIFIED'); `techwolf_test_annotations.csv` has only `sentence`, `label`.", f"Rows per file: {bm.groupby(['subset', 'split']).size().to_dict()}", ""]

    # ESCO
    sk = esco.load_skills()
    rows.append({"dataset": f"ESCO {cfg['esco']['version']} skills", "status": "OK" if esco.available() else "STAND-IN (benchmark label list)", "found": len(sk), "expected": cfg["esco"]["expected_skills_approx"]})
    if esco.available():
        raw = pd.read_csv(esco.csv_dir() / "skills_en.csv", nrows=5)
        schema += [f"## ESCO v{cfg['esco']['version']} (`{cfg['esco']['local']}`)", "", f"`skills_en.csv` columns: {list(raw.columns)}; {len(sk):,} skills; files: {sorted(p.name for p in esco.csv_dir().glob('*.csv'))}", ""]
    else:
        schema += ["## ESCO", "", "Official CSVs not present — using the benchmark's label list as a stand-in vocabulary (results provisional).", ""]

    # synthetic pairs
    syn_dir = ROOT / cfg["synthetic_esco"]["local"]
    if syn_dir.exists():
        from ..p2_normalize.train_biencoder import detect_columns, load_synthetic_pairs

        syn = load_synthetic_pairs()
        s_col, k_col = detect_columns(syn)
        rows.append({"dataset": "Synthetic ESCO skill sentences", "status": "OK" if abs(len(syn) - cfg["synthetic_esco"]["expected_rows"]) < 2000 else "CHECK", "found": len(syn), "expected": cfg["synthetic_esco"]["expected_rows"]})
        schema += [f"## Synthetic ESCO skill sentences (`{cfg['synthetic_esco']['local']}`)", "", f"Columns: {list(syn.columns)} → sentence column = `{s_col}`, skill column = `{k_col}`; {len(syn):,} rows; {syn[k_col].nunique():,} distinct skills.", f"Example: {syn.iloc[0].to_dict()}", ""]
    else:
        rows.append({"dataset": "Synthetic ESCO skill sentences", "status": "MISSING", "found": 0, "expected": cfg["synthetic_esco"]["expected_rows"]})

    # Djinni
    for key, loader in [("djinni_jobs", djinni.load_jobs), ("djinni_cvs", djinni.load_cvs)]:
        d = ROOT / cfg[key]["local"]
        if d.exists():
            df, info = loader()
            expected = cfg[key].get("expected_rows") or cfg[key].get("expected_rows_approx")
            rows.append({"dataset": key, "status": "OK" if abs(len(df) - expected) < 0.05 * expected else "CHECK", "found": len(df), "expected": expected})
            schema += [f"## {key} (`{cfg[key]['local']}`)", "", f"Source columns: {info['source_columns']}", f"Mapping to our schema: {info['mapping']}", f"Rows: {len(df):,}", ""]
        else:
            rows.append({"dataset": key, "status": "MISSING", "found": 0, "expected": cfg[key].get("expected_rows")})

    # live jobs
    if live_jobs.available():
        st = live_jobs.stats()
        ai, info = live_jobs.load_filtered()
        rows.append({"dataset": f"open-apply-jobs snapshot {live_jobs.snapshot_date()}", "status": "OK", "found": {"snapshot_rows": st.get("n_rows"), "ai_data_rows": len(ai)}, "expected": "one daily snapshot"})
        schema += [f"## open-apply-jobs (`{cfg['live_jobs']['local']}`, snapshot {live_jobs.snapshot_date()})", "", f"Columns: {st.get('columns')}", f"Mapping: {info['mapping']}", f"Snapshot rows: {st.get('n_rows'):,}; data/AI/ML filtered rows: {len(ai):,}", ""]
    else:
        rows.append({"dataset": "open-apply-jobs snapshot", "status": "MISSING", "found": 0, "expected": "one daily snapshot"})

    # base models
    for name in ("jobbert-base-cased", "all-MiniLM-L6-v2", "ConTeXT-Skill-Extraction-base"):
        p = ROOT / "models" / "base" / name
        ok = p.exists() and any(p.glob("*.safetensors")) or (p.exists() and any(p.rglob("*.bin")))
        rows.append({"dataset": f"model {name}", "status": "OK" if ok else "MISSING", "found": sorted(x.name for x in p.glob('*'))[:8] if p.exists() else [], "expected": "config + weights"})

    out = pd.DataFrame(rows)
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "data_schema.md").write_text("\n".join(schema))
    (ROOT / "results" / "eda").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "eda" / "asset_check.json").write_text(json.dumps(rows, indent=2, default=str))
    print(out.to_string())


if __name__ == "__main__":
    main()
