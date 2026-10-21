# %% [markdown]
# # EDA 04 — Live job postings (open-apply-jobs snapshot)
#
# **Dataset:** `edwarddgao/open-apply-jobs` — postings scraped daily from company career pages (Ashby, Greenhouse,
# Lever, …), one full snapshot per day. `scripts/download_assets.py` pulled ONE snapshot and kept (a) the postings
# whose title matches a data/AI/ML pattern (the live demo index), (b) a 20k random sample of the whole snapshot and
# (c) whole-snapshot summary statistics. Raw posting text belongs to the employers and is never committed.

# %%
import json, os, re, collections
import numpy as np, pandas as pd
import plotly.express as px, plotly.graph_objects as go
os.environ["SMC_SOURCE"] = "notebooks/04_eda_live_jobs"
from src.utils import viz; viz.setup()
from src.utils.config import ROOT
from src.utils.io import write_json
from src.data import live_jobs

stats = live_jobs.stats()
ai, ainfo = live_jobs.load_filtered()
smp, sinfo = live_jobs.load_sample()
print("snapshot date:", live_jobs.snapshot_date(), "| columns:", stats.get("columns"))
print(f"whole snapshot: {stats.get('n_rows'):,} postings | data/AI/ML filter: {len(ai):,} | random sample: {len(smp):,}")
print("mapping:", ainfo["mapping"])

# %% [markdown]
# ## 1. Snapshot overview: rows per source, remote share, salary availability

# %%
src_key = next((k for k in stats if k.startswith("rows_by_") and any(t in k.lower() for t in ("source", "ats", "provider", "platform"))), None)
if src_key:
    rs = pd.DataFrame(list(stats[src_key].items()), columns=["source", "postings"]).sort_values("postings", ascending=False).head(12)
    fig = px.bar(rs, x="source", y="postings", text="postings", title=f"Postings per ATS source in the snapshot ({live_jobs.snapshot_date()})")
    fig.update_traces(textposition="outside"); fig.update_layout(xaxis_title="", yaxis_title="postings")
    viz.save_fig(fig, "eda04_postings_by_source", "Live snapshot: number of postings per applicant-tracking-system source (Ashby, Greenhouse, Lever, …).")
    fig.show()
else:
    print("no source/ATS column in this snapshot; keys:", [k for k in stats if k.startswith("rows_by_")])

# %%
rem = smp["remote"].astype(str).str.lower()
remote_share = rem.isin(["true", "1", "remote", "yes"]).mean() if smp["remote"].notna().any() else None
sal_cols = [c for c in smp.columns if "salary" in c.lower()]
print("remote share (sample):", None if remote_share is None else f"{remote_share:.1%}", "| salary-like columns:", sal_cols, {k: v for k, v in stats.items() if k.startswith("nonnull_")})
sal = pd.DataFrame([{"field": k.replace("nonnull_", ""), "share_available": v / stats["n_rows"]} for k, v in stats.items() if k.startswith("nonnull_")])
if len(sal):
    fig = px.bar(sal, x="field", y="share_available", text=sal["share_available"].map(lambda v: f"{v:.0%}"), title="Salary fields: share of postings where the field is filled")
    fig.update_traces(textposition="outside"); fig.update_layout(yaxis_tickformat=".0%", xaxis_title="", yaxis_title="share of postings")
    viz.save_fig(fig, "eda04_salary_missingness", "Salary availability in the live snapshot: most postings leave compensation fields empty.")
    fig.show()

# %% [markdown]
# ## 2. Titles, departments, employment type

# %%
tt = pd.DataFrame(list(stats["top_titles"].items()), columns=["title", "postings"]).head(25)
fig = px.bar(tt.iloc[::-1], x="postings", y="title", orientation="h", title="25 most frequent job titles in the whole snapshot")
fig.update_layout(height=700, yaxis_title="", xaxis_title="postings")
viz.save_fig(fig, "eda04_top_titles", "Most frequent job titles across the whole live snapshot.", height=700)
fig

# %%
for col, name in [("department", "eda04_departments"), ("employment_type", "eda04_employment_type")]:
    if smp[col].notna().any():
        d = smp[col].astype(str).str.strip().replace("", np.nan).dropna().value_counts().head(15).rename_axis(col).reset_index(name="postings")
        fig = px.bar(d.iloc[::-1], x="postings", y=col, orientation="h", title=f"Top {col.replace('_', ' ')}s (20k sample)")
        fig.update_layout(height=520, yaxis_title="", xaxis_title="postings")
        viz.save_fig(fig, name, f"Live snapshot sample: distribution of {col.replace('_', ' ')}.", height=520)
        fig.show()

# %% [markdown]
# ## 3. Description coverage, description length and posting dates
#
# Not every ATS feed returns the posting body: the snapshot stores `description_html` only where the source API
# exposes it. This decides how many postings the skill pipeline can actually read, so we measure it per source.
# (Greenhouse delivers its bodies as *escaped* HTML — `&lt;p&gt;…` — which `strip_html` now unescapes before parsing.)

# %%
for d in (smp, ai):
    d["has_desc"] = d["description"].str.len() > 0
cov = (smp.groupby("source")["has_desc"].agg(share_with_description="mean", postings="size").reset_index()
          .sort_values("postings", ascending=False))
cov["label"] = cov.apply(lambda r: f"{r.share_with_description:.0%} (n={r.postings:,})", axis=1)
fig = px.bar(cov, x="source", y="share_with_description", text="label", title="Share of postings that carry a description, per ATS source (20k sample)")
fig.update_traces(textposition="outside"); fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1.15], xaxis_title="", yaxis_title="postings with description text")
viz.save_fig(fig, "eda04_description_coverage_by_source", "Only the Greenhouse, Ashby, Lever, Rippling and Gem feeds return posting bodies; Workday and SmartRecruiters listings are title-only, so the live index can only cover postings with text.")
print(f"postings with a description: sample {smp['has_desc'].mean():.1%} | data/AI/ML subset {ai['has_desc'].mean():.1%} ({int(ai['has_desc'].sum()):,} of {len(ai):,})")
fig

# %%
smp["n_words"] = smp["description"].str.split().str.len()
wd = smp.loc[smp["has_desc"], "n_words"]
fig = px.histogram(wd[wd <= 2000], x="n_words", nbins=80, title=f"Description length (words) — {len(wd):,} sampled postings that have a description")
fig.update_layout(xaxis_title="words", yaxis_title="postings", bargap=0.05, showlegend=False)
viz.save_fig(fig, "eda04_description_length", f"Length of live postings after HTML stripping (postings with text only): median {wd.median():.0f} words — several times longer than a Djinni posting.")
print(wd.describe().round(1).to_dict())
fig

# %%
smp["posted"] = pd.to_datetime(smp["posted_at"], errors="coerce", utc=True).dt.date
pdd = smp.dropna(subset=["posted"]).groupby("posted").size().reset_index(name="postings").tail(60)
if len(pdd):
    fig = px.bar(pdd, x="posted", y="postings", title="Postings by posting date (last 60 days with data, 20k sample)")
    fig.update_layout(xaxis_title="", yaxis_title="postings")
    viz.save_fig(fig, "eda04_posting_dates", "Posting dates in the sample: the snapshot contains postings that are still open, so dates spread over several months with a recent peak.")
    fig.show()

# %% [markdown]
# ## 4. The data/AI/ML subset used for the live demo

# %%
ai["title_norm"] = ai["title"].str.lower().str.replace(r"[^a-z ]", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
fam = {"machine learning / ML engineer": r"machine learning|ml engineer|mlops|ml scientist", "data scientist / applied scientist": r"data scien|applied scientist|research scientist", "data engineer / platform": r"data engineer|data platform|analytics engineer",
       "AI engineer / LLM / GenAI": r"ai engineer|llm|generative|gen ai|ai platform|ai product|artificial intelligence|deep learning|computer vision|nlp|natural language", "data analyst": r"data analyst"}
ai["family"] = "other"
for name, pat in fam.items():
    ai.loc[ai["family"].eq("other") & ai["title_norm"].str.contains(pat, regex=True), "family"] = name
fc = ai["family"].value_counts().rename_axis("family").reset_index(name="postings")
fig = px.bar(fc, x="family", y="postings", text="postings", title=f"Data/AI/ML postings in the snapshot by role family (n={len(ai):,})")
fig.update_traces(textposition="outside"); fig.update_layout(xaxis_title="", yaxis_title="postings")
viz.save_fig(fig, "eda04_ai_role_families", "The live demo index: data/AI/ML postings grouped into role families by title pattern.")
rem_ai = ai["remote"].astype(str).str.lower().isin(["true", "1", "remote", "yes"]).mean()
print(f"remote share among data/AI/ML postings: {rem_ai:.1%}")
fig

# %% [markdown]
# ## 5. After the pipeline: ESCO skills extracted from the live AI/ML postings (if the index has been built)

# %%
idx_file = ROOT / "data" / "processed" / "live_jobs_jobs.parquet"
if idx_file.exists():
    li = pd.read_parquet(idx_file)
    sk = collections.Counter(s for v in li["esco_skills"].fillna("") for s in v.split("|") if s)
    top_sk = pd.DataFrame(sk.most_common(30), columns=["esco_skill", "postings"])
    fig = px.bar(top_sk.iloc[::-1], x="postings", y="esco_skill", orientation="h", title=f"Top 30 ESCO skills extracted from {len(li):,} live data/AI/ML postings")
    fig.update_layout(height=780, yaxis_title="", xaxis_title="postings mentioning the skill")
    viz.save_fig(fig, "eda04_top_esco_skills_live", "Most frequent ESCO skills extracted from this week's data/AI/ML postings by the champion pipeline (tagger → bi-encoder).", height=780)
    fig.show()
    nil = collections.Counter(s.lower() for v, e in zip(li["raw_skills"].fillna(""), li["n_not_in_esco"]) for s in v.split("|") if s)
    emerging = ROOT / "results" / "monitoring" / "emerging_skills.csv"
    if emerging.exists():
        em = pd.read_csv(emerging).head(20)
        print("emerging-skills queue (not in ESCO):"); display(em)
    print(f"mean ESCO skills per posting: {li['n_esco'].mean():.1f}; mean phrases not in ESCO: {li['n_not_in_esco'].mean():.1f}")
else:
    print("live index not built yet — run: python -m src.index.build_job_index --pool live")

# %% [markdown]
# ## 6. Summary

# %%
summary = {"snapshot_date": live_jobs.snapshot_date(), "n_rows_snapshot": stats.get("n_rows"), "n_ai_data": int(len(ai)), "n_sample": int(len(smp)), "columns": stats.get("columns"),
           "rows_by_source": stats.get(src_key) if src_key else None, "remote_share_sample": remote_share, "remote_share_ai": float(rem_ai),
           "salary_available": sal.set_index("field")["share_available"].round(4).to_dict() if len(sal) else None,
           "description_coverage_sample": float(smp["has_desc"].mean()), "description_coverage_ai": float(ai["has_desc"].mean()), "n_ai_with_description": int(ai["has_desc"].sum()),
           "description_coverage_by_source": cov.set_index("source")["share_with_description"].round(3).to_dict(),
           "description_words_median_with_text": float(wd.median()), "description_words_p25_p75_with_text": [float(wd.quantile(0.25)), float(wd.quantile(0.75))],
           "description_length_quantiles_chars_nonnull_whole_snapshot": stats.get("description_length_quantiles_chars"),
           "ai_role_families": fc.set_index("family")["postings"].to_dict(), "top_titles": dict(list(stats["top_titles"].items())[:15])}
write_json(summary, "results/eda/live_jobs_summary.json")
print(json.dumps({k: v for k, v in summary.items() if k not in ("columns", "top_titles")}, indent=1))
print(f"""
THREE MOST IMPORTANT FINDINGS
1. The snapshot holds {stats.get('n_rows'):,} open postings; {len(ai):,} ({len(ai)/stats.get('n_rows', 1):.1%}) are data/AI/ML roles — enough for a live demo
   that changes every day, and small enough to re-index on CPU.
2. Only {smp['has_desc'].mean():.0%} of postings ({ai['has_desc'].mean():.0%} of the data/AI/ML subset, {int(ai['has_desc'].sum()):,} postings) carry a description: the Greenhouse,
   Ashby, Lever, Rippling and Gem feeds return the body, Workday and SmartRecruiters listings are title-only. The live index therefore
   covers postings with text (>= 200 characters), and coverage of Workday-heavy employers is a known gap for the ops plan.
3. Postings with text are long ({wd.median():.0f} words median) and HTML-formatted (Greenhouse even double-escapes its HTML), unlike SkillSpan's single
   sentences: the pipeline must clean and sentence-split postings, and the tagger faces a domain shift. Compensation and remote
   fields are sparsely filled, so the matcher relies on skills and titles, not on structured metadata.
""")
