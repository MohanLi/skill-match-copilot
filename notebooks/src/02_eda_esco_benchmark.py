# %% [markdown]
# # EDA 02 — Skill-Extraction-benchmark and the ESCO taxonomy (Problem 2 data)
#
# **Benchmark:** Decorte et al.'s Skill-Extraction-benchmark — job-ad sentences in which every skill phrase
# (span) was mapped by annotators to an **ESCO 1.1.0** skill, to `LABEL NOT PRESENT` (ESCO has no such skill)
# or to `UNDERSPECIFIED` (too vague to map). Three subsets: **TECH** (tech job ads), **HOUSE** (general job ads)
# and **TECHWOLF** (sentence-level labels). Source: GitHub repo (CSV), also on Hugging Face as
# `TechWolf/skill-extraction-{tech,house,techwolf}` (CC BY 4.0).
#
# **Taxonomy:** ESCO v1.1.0 (English) — ~13.9k skills. The official CSV (with alternative labels and
# descriptions) is used when present; otherwise the preferred-label list shipped with the benchmark.

# %%
import json, os, re, collections
import numpy as np, pandas as pd
import plotly.express as px, plotly.graph_objects as go
os.environ["SMC_SOURCE"] = "notebooks/02_eda_esco_benchmark"
from src.utils import viz; viz.setup()
from src.utils.io import write_json
from src.data import esco_benchmark as bm, esco

df = bm.load_all()
df["subset_split"] = df["subset"] + "/" + df["split"].str.replace("validation", "val")
print("schema:", {c: str(df[c].dtype) for c in df.columns})
df.head(8)

# %% [markdown]
# ## 1. Record counts, missing values, schema

# %%
counts = df.groupby("subset_split").agg(phrases=("span", "size"), sentences=("sentence", "nunique"),
                                        distinct_labels=("label", lambda s: s[~s.isin(bm.NIL_LABELS)].nunique()))
counts["phrases/sentence"] = (counts["phrases"] / counts["sentences"]).round(2)
nulls = {c: int((df[c] == "").sum()) for c in ["sentence", "span", "sub_span", "label"]}
print("empty-string cells:", nulls, "(sub_span is optional: a narrower sub-phrase that justifies the label)")
counts

# %% [markdown]
# ## 2. Label status: how much of a job ad is NOT in ESCO?

# %%
status = df.groupby(["subset_split", "status"]).size().reset_index(name="phrases")
status["share"] = status["phrases"] / status.groupby("subset_split")["phrases"].transform("sum")
order = ["tech/val", "tech/test", "house/val", "house/test", "techwolf/test"]
fig = px.bar(status, x="subset_split", y="share", color="status", text=status["share"].map(lambda x: f"{x:.0%}"),
             category_orders={"subset_split": order, "status": ["ESCO", "LABEL NOT PRESENT", "UNDERSPECIFIED"]},
             title="Share of annotated phrases by label status")
fig.update_traces(textposition="inside")
fig.update_layout(yaxis_tickformat=".0%", yaxis_title="share of phrases", xaxis_title="", bargap=0.35)
viz.save_fig(fig, "eda02_label_status_by_subset", "Label status per benchmark subset: in TECH ads about 43% of annotated skill phrases cannot be mapped to ESCO (mostly LABEL NOT PRESENT) and 37% in HOUSE — the gap the 'not in ESCO' detector targets; TECHWOLF has sentence-level ESCO labels only.")
nil_share = df.groupby("subset")["status"].apply(lambda s: (s != "ESCO").mean()).round(3).to_dict()
print("share of phrases not mappable to ESCO, by subset:", nil_share)
fig

# %% [markdown]
# ## 3. Which ESCO skills are used, and how much of ESCO the benchmark covers

# %%
skills = esco.load_skills()
vocab = set(skills["label"])
esco_source = skills["source"].iloc[0]
print(f"ESCO vocabulary: {len(vocab):,} skills (source: {esco_source})")
lab = df[df["status"] == "ESCO"]
in_vocab = lab["label"].isin(vocab).mean()
print(f"benchmark ESCO labels found in the vocabulary: {in_vocab:.1%}  (labels not found: {lab[~lab['label'].isin(vocab)]['label'].nunique()})")
used = lab["label"].nunique()
print(f"distinct ESCO skills used across the benchmark: {used:,} = {used / len(vocab):.1%} of ESCO")
top = lab["label"].value_counts().head(25).rename_axis("esco_skill").reset_index(name="phrases")
fig = px.bar(top.iloc[::-1], x="phrases", y="esco_skill", orientation="h", title="25 most frequent ESCO skills in the benchmark (all subsets)")
fig.update_layout(height=700, yaxis_title="", xaxis_title="annotated phrases")
viz.save_fig(fig, "eda02_top_esco_labels", "The most frequent ESCO target skills: a long tail — the top 25 skills cover only a small share of phrases, so retrieval must handle thousands of rare labels.", height=700)
cover_top25 = top["phrases"].sum() / len(lab)
print(f"top-25 skills cover {cover_top25:.1%} of ESCO-labelled phrases")
fig

# %%
lab_len = skills["label"].str.split().str.len()
fig = px.histogram(lab_len, nbins=15, title="ESCO skill labels are short multi-word phrases (tokens per label)")
fig.update_layout(xaxis_title="tokens in ESCO preferred label", yaxis_title="skills", showlegend=False, bargap=0.05)
viz.save_fig(fig, "eda02_esco_label_length", "Token length of the ~13.9k ESCO skill labels: most are 2–5-word verb phrases ('manage musical staff'), which is why a bi-encoder (not keyword match) is needed.")
if esco_source == "esco_csv":
    n_alt = skills["alt_labels"].map(len)
    print(f"alternative labels per skill: mean {n_alt.mean():.2f}, max {n_alt.max()}, skills with none: {(n_alt==0).mean():.1%}")
    print("skill types:", skills["skill_type"].value_counts().to_dict())
fig

# %% [markdown]
# ## 4. Sentence and phrase statistics

# %%
sent = df.drop_duplicates(["subset_split", "sentence"]).copy()
sent["n_tokens"] = sent["sentence"].str.split().str.len()
per_sent = df.groupby(["subset_split", "sentence"]).size().reset_index(name="phrases")
print(sent.groupby("subset_split")["n_tokens"].describe().round(1))
fig = px.box(sent, x="subset_split", y="n_tokens", category_orders={"subset_split": order}, points=False,
             title="Tokens per sentence by subset")
fig.update_layout(xaxis_title="", yaxis_title="tokens")
viz.save_fig(fig, "eda02_sentence_length", "Sentence length per subset: TECH sentences are shortest (median ~10 tokens), HOUSE sentences the longest (median 15–18 tokens).")
fig

# %%
fig = px.histogram(per_sent, x="phrases", color="subset_split", barmode="group", category_orders={"subset_split": order},
                   title="Annotated phrases per sentence")
fig.update_layout(xaxis_title="phrases per sentence", yaxis_title="sentences", bargap=0.15)
viz.save_fig(fig, "eda02_spans_per_sentence", "Number of annotated skill phrases per sentence: most sentences carry 1–3 phrases; a few list-style sentences carry 10+.")
fig

# %%
df["span_len"] = df["span"].str.split().str.len()
print(df.groupby("status")["span_len"].describe().round(2))

# %% [markdown]
# ## 5. What kind of phrases have no ESCO skill? (the emerging-skills queue)

# %%
nil = df[df["status"] == "LABEL NOT PRESENT"]
top_nil = nil["span"].str.lower().value_counts().head(30).rename_axis("phrase").reset_index(name="count")
fig = px.bar(top_nil.iloc[::-1], x="count", y="phrase", orientation="h", title="Most frequent phrases with NO ESCO skill (LABEL NOT PRESENT)")
fig.update_traces(marker_color=viz.PALETTE[1])
fig.update_layout(height=760, yaxis_title="", xaxis_title="occurrences")
viz.save_fig(fig, "eda02_unmapped_terms", "Phrases annotators could not map to ESCO 1.1.0: concrete frameworks, tools and platforms (ReactJS, Docker, Django, Spring, Azure, Node.js…) — exactly the vocabulary a job-matching product must track outside the taxonomy.", height=760)
und = df[df["status"] == "UNDERSPECIFIED"]["span"].str.lower().value_counts().head(15)
print("UNDERSPECIFIED examples:", und.index.tolist())
fig

# %%
# Are unmapped phrases "tech" terms? crude check: single-token capitalised/known-tool-like phrases
tool_like = nil["span"].str.match(r"^[A-Za-z0-9.+#-]{2,20}$").mean()
print(f"share of LABEL NOT PRESENT phrases that are single tokens (tool/tech names): {tool_like:.1%}")

# %% [markdown]
# ## 6. Summary

# %%
summary = {
    "counts": counts.reset_index().to_dict("records"),
    "empty_cells": nulls,
    "status_share": {k: {s: round(v, 4) for s, v in g.set_index("status")["share"].items()} for k, g in status.groupby("subset_split")},
    "not_mappable_share_by_subset": nil_share,
    "esco_vocab_size": len(vocab), "esco_source": esco_source,
    "benchmark_labels_in_vocab_share": round(float(in_vocab), 4),
    "distinct_esco_skills_used": int(used), "esco_coverage_share": round(used / len(vocab), 4),
    "top25_coverage_of_phrases": round(float(cover_top25), 4),
    "tokens_per_sentence_median_by_subset": sent.groupby("subset_split")["n_tokens"].median().to_dict(),
    "span_len_median_by_status": df.groupby("status")["span_len"].median().to_dict(),
    "single_token_share_of_unmapped": round(float(tool_like), 4),
    "top_unmapped": top_nil.head(15).to_dict("records"),
}
write_json(summary, "results/eda/esco_benchmark_summary.json")
print(json.dumps({k: v for k, v in summary.items() if k not in ("top_unmapped", "counts")}, indent=1))
print(f"""
THREE MOST IMPORTANT FINDINGS
1. {nil_share.get('tech', 0):.0%} of skill phrases in TECH ads cannot be mapped to any ESCO skill (vs {nil_share.get('house', 0):.0%} in HOUSE):
   ESCO lacks concrete frameworks and tools (ReactJS, Docker, Django, Spring, Azure). "Not in ESCO" detection is a product requirement, not an edge case.
2. The benchmark uses {used:,} distinct ESCO skills ({used/len(vocab):.1%} of the taxonomy) with a long tail — the top 25 skills cover only
   {cover_top25:.0%} of phrases, so Problem 2 is retrieval over ~{len(vocab)//1000}k labels, not classification over a few hundred.
3. ESCO labels are short verb phrases (mostly 2–5 tokens) while job-ad sentences are 10–30 tokens: the mapping is semantic
   ('help our customers' → 'assist customers'), which motivates a fine-tuned bi-encoder and an LLM re-ranker over lexical matching.
""")
