# %% [markdown]
# # EDA 05 — Generated data: synthetic training sentences and LLM outputs
#
# The rubric asks to inspect generated samples for quality, diversity, biases, anomalies and hallucinations.
# Four generated corpora exist in this project:
# 1. **TechWolf's synthetic ESCO sentences** (LLM-generated job-ad sentences used to train the bi-encoder);
# 2. the **few-shot LLM extractor's** outputs on the SkillSpan test set (Problem 1 challenger);
# 3. the **retrieve-then-LLM** selections (Problem 2 challenger);
# 4. the **agents' explanations** and the **LLM gold labels** (Problem 3).

# %%
import json, os, re, collections
import numpy as np, pandas as pd
import plotly.express as px, plotly.graph_objects as go
os.environ["SMC_SOURCE"] = "notebooks/05_eda_generated"
from src.utils import viz; viz.setup()
from src.utils.config import ROOT
from src.utils.io import read_jsonl, write_json
summary = {}

# %% [markdown]
# ## 1. Synthetic ESCO skill sentences (bi-encoder training data)

# %%
syn_dir = ROOT / "data/raw/hf/TechWolf__Synthetic-ESCO-skill-sentences"
if syn_dir.exists():
    from src.p2_normalize.train_biencoder import load_synthetic_pairs, detect_columns
    syn = load_synthetic_pairs(); s_col, k_col = detect_columns(syn)
    syn = syn.rename(columns={s_col: "sentence", k_col: "skill"})[["sentence", "skill"]].dropna()
    syn["n_words"] = syn["sentence"].str.split().str.len()
    per_skill = syn.groupby("skill").size()
    first3 = syn["sentence"].str.lower().str.split().str[:3].str.join(" ").value_counts().head(15)
    ttr = syn["sentence"].str.lower().str.split().explode().nunique() / syn["n_words"].sum()
    dup = syn.duplicated("sentence").mean()
    print(f"{len(syn):,} pairs, {syn['skill'].nunique():,} skills, {per_skill.median():.0f} sentences per skill (median), duplicate sentences {dup:.2%}, type-token ratio {ttr:.3f}")
    summary["synthetic"] = {"pairs": int(len(syn)), "skills": int(syn["skill"].nunique()), "sentences_per_skill_median": float(per_skill.median()), "duplicate_share": float(dup), "ttr": float(ttr), "words_median": float(syn["n_words"].median()), "top_openings": first3.to_dict()}
    fig = px.histogram(syn[syn["n_words"] <= 60], x="n_words", nbins=60, title="Synthetic sentences: words per sentence")
    fig.update_layout(xaxis_title="words", yaxis_title="sentences", showlegend=False, bargap=0.05)
    viz.save_fig(fig, "eda05_synthetic_length", "TechWolf's synthetic ESCO sentences are uniformly short (one or two clauses), unlike real job-ad sentences — a domain gap for the bi-encoder.")
    fig.show()
    fo = first3.rename_axis("opening").reset_index(name="sentences")
    fig = px.bar(fo.iloc[::-1], x="sentences", y="opening", orientation="h", title="Most common three-word openings of synthetic sentences (template bias)")
    fig.update_layout(height=520, yaxis_title="")
    viz.save_fig(fig, "eda05_synthetic_openings", "Template bias in the generated training sentences: a handful of openings ('experience in …', 'ability to …') dominate, which the NO_DUPLICATES batch sampler and real-sentence evaluation guard against.", height=520)
    fig.show()
    print(syn.sample(5, random_state=1).to_string())
else:
    print("synthetic pairs not available")

# %% [markdown]
# ## 2. Few-shot LLM extractor outputs (Problem 1 challenger)

# %%
calls = [p for p in (ROOT / "results/p1").glob("llm_*_test_calls.jsonl")]
if calls:
    llm = pd.DataFrame(read_jsonl(str(calls[0].relative_to(ROOT))))
    llm["n_phrases"] = llm["skills"].str.len() + llm["knowledge"].str.len()
    llm["n_unmatched"] = llm["unmatched_skills"].str.len() + llm["unmatched_knowledge"].str.len()
    from src.data import skillspan as ss
    gold = {r["sid"]: ss.gold_phrases(r) for r in ss.load_split("test")}
    llm["gold_n"] = llm["sid"].map(lambda s: len(gold[s]["skills"]) + len(gold[s]["knowledge"]))
    print(f"{len(llm)} sentences; parse failures {int((~llm['parse_ok']).sum())}; phrases returned {int(llm['n_phrases'].sum())}, not in text (hallucinated) {int(llm['n_unmatched'].sum())}")
    summary["p1_llm"] = {"sentences": int(len(llm)), "parse_failures": int((~llm["parse_ok"]).sum()), "phrases": int(llm["n_phrases"].sum()), "unmatched": int(llm["n_unmatched"].sum()),
                         "mean_phrases_llm": float(llm["n_phrases"].mean()), "mean_phrases_gold": float(llm["gold_n"].mean()), "output_tokens_median": float(llm["output_tokens"].median()), "latency_ms_median": float(llm["latency_ms"].dropna().median()) if llm["latency_ms"].notna().any() else None}
    cmp = pd.concat([pd.DataFrame({"phrases per sentence": llm["n_phrases"].clip(upper=10), "source": "LLM output"}), pd.DataFrame({"phrases per sentence": llm["gold_n"].clip(upper=10), "source": "gold annotation"})])
    fig = px.histogram(cmp, x="phrases per sentence", color="source", barmode="group", title="Phrases per sentence: LLM extractor vs gold annotation (test)")
    fig.update_layout(yaxis_title="sentences", bargap=0.15)
    viz.save_fig(fig, "eda05_llm_phrases_per_sentence", "Over-/under-generation of the few-shot LLM extractor: distribution of phrases per sentence compared with the gold annotation.")
    fig.show()
    un = llm[llm["n_unmatched"] > 0][["sentence", "unmatched_skills", "unmatched_knowledge"]].head(12)
    print("HALLUCINATION EXAMPLES (phrases not present verbatim in the sentence):"); print(un.to_string())
    pl = pd.DataFrame({"len": [len(p.split()) for ph in llm["skills"] for p in ph] + [len(p.split()) for ph in llm["knowledge"] for p in ph], "type": ["skill"] * int(llm["skills"].str.len().sum()) + ["knowledge"] * int(llm["knowledge"].str.len().sum())})
    fig = px.histogram(pl[pl["len"] <= 15], x="len", color="type", barmode="group", title="Length (words) of phrases returned by the LLM")
    fig.update_layout(xaxis_title="words", yaxis_title="phrases", bargap=0.15)
    viz.save_fig(fig, "eda05_llm_phrase_length", "Length of LLM-returned phrases: skill phrases are long, knowledge phrases short — the same shape as the gold spans.")
    fig.show()
else:
    print("P1 LLM outputs not available yet")

# %% [markdown]
# ## 3. Retrieve-then-LLM selections (Problem 2 challenger)

# %%
rtl = sorted((ROOT / "results/p2").glob("rtl_context_calls_test_k*.jsonl")) or sorted((ROOT / "results/p2").glob("rtl_calls_test_k*.jsonl"))  # challenger (ConTeXT retriever) first
if rtl:
    r = pd.DataFrame(read_jsonl(str(rtl[0].relative_to(ROOT))))
    r["n_sel"] = r["selected"].str.len(); r["n_invalid"] = r["invalid"].map(lambda x: len(x) if isinstance(x, list) else 0)
    print(f"{len(r)} sentences; mean selected {r['n_sel'].mean():.2f}; 'none' rate {r['none'].mean():.1%}; invalid labels (not in candidates) {int(r['n_invalid'].sum())}; errors {int(r.get('error', pd.Series()).notna().sum()) if 'error' in r else 0}")
    summary["p2_rtl"] = {"sentences": int(len(r)), "mean_selected": float(r["n_sel"].mean()), "none_rate": float(r["none"].mean()), "invalid_labels": int(r["n_invalid"].sum())}
    fig = px.histogram(r, x="n_sel", nbins=12, title="ESCO skills selected per sentence by the LLM (from the top-k candidates)")
    fig.update_layout(xaxis_title="selected skills", yaxis_title="sentences", showlegend=False, bargap=0.05)
    viz.save_fig(fig, "eda05_rtl_selected", "Retrieve-then-LLM: number of ESCO candidates the LLM accepts per sentence; zero means it judged none of the retrieved skills a true mention.")
    fig.show()
else:
    print("RTL outputs not available yet")

# %% [markdown]
# ## 4. Agent explanations and gold labels (Problem 3)

# %%
tr_dirs = sorted((ROOT / "results/p3/traces").glob("*")) if (ROOT / "results/p3/traces").exists() else []
rows = []
for d in tr_dirs:
    for f in d.glob("*.json"):
        t = json.loads(f.read_text())
        for e in t.get("ranking", []):
            rows.append({"approach": d.name, "cv_id": f.stem, "job_id": e["job_id"], "fit_score": e.get("fit_score"), "n_matched": len(e.get("matched_skills", [])), "n_missing": len(e.get("missing_skills", [])), "n_dropped": len(e.get("dropped", []) or []), "summary_words": len((e.get("summary") or "").split())})
if rows:
    ex = pd.DataFrame(rows)
    pairs = pd.read_csv(ROOT / "data/labels/final.csv")
    ex = ex.merge(pairs[["cv_id", "job_id", "relation", "label"]], on=["cv_id", "job_id"], how="left")
    print(ex.groupby("approach")[["fit_score", "n_matched", "n_missing", "n_dropped", "summary_words"]].mean().round(2))
    summary["p3_explanations"] = ex.groupby("approach")[["fit_score", "n_matched", "n_missing", "n_dropped"]].mean().round(3).to_dict()
    fig = px.box(ex, x="relation", y="fit_score", color="approach", category_orders={"relation": ["same", "related", "random"]}, title="Agent fit scores by pool relation (same / related / random category) — do scores separate the tiers?")
    fig.update_layout(xaxis_title="job relation to the CV's category", yaxis_title="fit score (0-100)")
    viz.save_fig(fig, "eda05_fit_by_relation", "Bias check: distribution of agent fit scores for same-, related- and random-category jobs — scores should fall across tiers; overlap shows where the agents are over-generous.")
    fig.show()
    fig = px.box(ex, x="label", y="fit_score", color="approach", title="Agent fit scores vs gold label (0 / 1 / 2)")
    fig.update_layout(xaxis_title="gold fit label", yaxis_title="fit score (0-100)")
    viz.save_fig(fig, "eda05_fit_by_label", "Calibration check: agent fit scores against the gold 0/1/2 labels.")
    fig.show()
    reasons = collections.Counter()
    for d in tr_dirs:
        for f in d.glob("*.json"):
            t = json.loads(f.read_text())
            for k, v in (t.get("faithfulness", {}).get("reasons") or {}).items():
                reasons[(d.name, k)] += v
    if reasons:
        rd = pd.DataFrame([{"approach": a, "reason": k, "claims": v} for (a, k), v in reasons.items()])
        fig = px.bar(rd, x="reason", y="claims", color="approach", barmode="group", title="Why claims were judged unsupported (automatic checker reason codes)")
        fig.update_layout(xaxis_title="", yaxis_title="claims")
        viz.save_fig(fig, "eda05_unsupported_reasons", "Failure modes of generated explanations: reason codes of unsupported claims (quote not found in resume / job, skill not in the quotes, missing skill actually present).")
        fig.show()
kap = ROOT / "data/labels/kappa.json"
if kap.exists():
    k = json.loads(kap.read_text()); summary["gold_labels"] = {kk: k[kk] for kk in ("n_pairs", "agreement_rate", "cohen_kappa_linear", "cohen_kappa_quadratic", "adjudicated", "final_label_dist") if kk in k}
    cm = np.array(k["confusion_a_rows_b_cols"])
    fig = go.Figure(go.Heatmap(z=cm, x=["B: 0", "B: 1", "B: 2"], y=["A: 0", "A: 1", "A: 2"], text=cm, texttemplate="%{text}", colorscale=[[i/(len(viz.SEQ_BLUE)-1), c] for i, c in enumerate(viz.SEQ_BLUE)], showscale=False))
    fig.update_layout(title=f"Rater agreement on fit labels: {k.get('rater_a')} (A) vs {k.get('rater_b')} (B) — linear κ = {k['cohen_kappa_linear']:.2f}", height=420)
    viz.save_fig(fig, "eda05_rater_agreement", "Agreement between the two LLM raters on the 300 resume–job fit labels (confusion matrix; disagreements were adjudicated).", height=420)
    fig.show()
    lab = pd.read_csv(ROOT / "data/labels/labels_opus.csv")
    lab["rationale_words"] = lab["rationale"].fillna("").str.split().str.len()
    print("rationale length (words):", lab["rationale_words"].describe().round(1).to_dict())
    print("EXAMPLE RATIONALES:"); print(lab.sample(3, random_state=3)[["label", "rationale"]].to_string())

# %% [markdown]
# ## 5. Summary

# %%
write_json(summary, "results/eda/generated_summary.json")
print(json.dumps(summary, indent=1, default=str))
print("""
THREE MOST IMPORTANT FINDINGS
1. The synthetic training sentences are short and template-like (few dominant openings) — useful for teaching the bi-encoder
   skill semantics, but real job-ad sentences are longer and noisier, so evaluation must stay on real benchmark sentences.
2. The few-shot LLM extractor returns phrases that are not verbatim in the sentence (hallucinations measured, not assumed);
   they are counted and dropped, and the remaining errors are mostly boundary disagreements with the SkillSpan conventions.
3. Agent explanations fail in identifiable ways (quote paraphrased, skill not in the quotes, 'missing' skill actually present);
   the verifier removes these before they reach the user, which is visible in the hallucination-rate comparison.
""")
