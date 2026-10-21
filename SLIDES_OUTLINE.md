# Slide outline — Skill-Match Copilot (15 minutes, ~14 slides)

Each slide lists its figure(s) from `reports/figures/` (PNG for slides, HTML to demo interactively), the
one-sentence message, and speaker notes. Timings add up to 14 min, leaving a minute for the hand-over.
Every number is traceable to `results/` (see `REPORT.md`).

| # | Slide | Figure(s) | Time |
|--:|-------|-----------|-----:|
| 1 | Title + one-line pitch | — | 0:20 |
| 2 | The problem: skills are the common key | — (3-box diagram: resume → skills → ESCO → jobs) | 1:00 |
| 3 | Pipeline and the three NLP problems | `docs/architecture.md` flow (export the mermaid diagram) | 1:00 |
| 4 | Data: four public datasets + one live feed | table from REPORT §2 | 0:40 |
| 5 | EDA I — what the annotations look like | `eda01_span_length_hist`, `eda01_tokens_per_sentence` | 1:00 |
| 6 | EDA II — 40 % of real skills are not in ESCO | `eda02_label_status_by_subset`, `eda02_unmapped_terms` | 1:00 |
| 7 | EDA III — the job market we match against | `eda03_umap_market_map`, `eda04_description_coverage_by_source` | 1:00 |
| 8 | P1 — tagger vs few-shot LLM | `p1_f1_comparison_test`, `p1_false_positives_test` | 1:30 |
| 9 | P2 — normalizing to ESCO: 5 approaches | `p2_workrb_map_test` | 1:30 |
| 10 | P2 — the reject option ("not in ESCO") | `p2_nil_results`, `p2_nil_threshold_sweep` | 1:00 |
| 11 | P3 — agents: single ReAct vs planner/verifier | `p3_ranking_quality_agent`, `p3_efficiency_agent`, one trace | 1:30 |
| 12 | P3 — faithfulness: every claim is a quote | `p3_hallucination_agent`, `eda05_rater_agreement`, `p3_silver_eval` | 1:00 |
| 13 | Model ops: serving, cache, drift, review queue | `ops_dashboard` | 1:00 |
| 14 | Conclusions, limitations, what we would do next | — | 0:40 |

---

## Slide-by-slide notes

### 1. Title
*Skill-Match Copilot — extract, normalize, match, explain.* Name, course, date. One line: "A job-matching
assistant whose every claim is a quote, built on two measured NLP models and one verified agent."

### 2. The problem
Resumes and postings describe the same skill in different words; recruiters and seekers both need a common
key. ESCO (13,891 skills) is that key — but 40 % of real-world skill phrases are not in it, so the system
must also say "I don't know this one". Business value: ranked, auditable matches; emerging-skill tracking.

### 3. Pipeline
Three problems, each with a champion and a challenger, each measured on a public benchmark or a labelled
set: (1) span tagging, (2) retrieval with a reject option, (3) tool-using agent with a verifier. Point at
the shared LLM wrapper (cache, log, budget) and at Chroma as the store for jobs.

### 4. Data
SkillSpan (expert BIO tags), TechWolf benchmark + ESCO 1.1.0 (via WorkRB), 138k synthetic pairs, Djinni
(142k postings, 210k CVs), one live snapshot (1.16M postings, 21k data/AI/ML). Licences on the slide
footer. Say what was cleaned: HTML, PII, duplicates, minimum length.

### 5. EDA I — annotations
Skill spans are 4-token verb phrases, knowledge spans 1–2 tokens → two taggers. 73 % of test sentences
have no span → measure over-extraction. Sentences are short → 128-token window is safe.

### 6. EDA II — ESCO coverage
43 % (TECH) / 37 % (HOUSE) of annotated phrases cannot be mapped; the unmapped ones are tools and
frameworks (Docker, React, Azure). Long-tail target space: top-25 skills cover 24 % of phrases.

### 7. EDA III — the market
Djinni categories form clean clusters and profiles land where postings land (market map). The live feed
only exposes bodies for some ATS sources (31 % overall, 53 % of data/AI/ML) — a coverage limit to state.
Supply/demand imbalance (QA 3.7 profiles per posting, DevOps 0.5) means ranking within categories matters.

### 8. P1 — extraction
Strict F1 skill 0.47 / knowledge 0.63 (tagger) vs 0.29 / 0.53 (Haiku few-shot) vs 0.11 / 0.21 (dictionary).
Tagger makes 5–7 false spans per 100 empty sentences, the LLM 25–30. The LLM paraphrases 7 % of its
"extractions" — unacceptable for quoted evidence. Caveat on the slide: one seed, 4 epochs, dev F1 still rising.

### 9. P2 — normalization
Read the bars left to right: BM25 0.28 → zero-shot MiniLM 0.37 → in-house fine-tuned 0.42 → ConTeXT 0.54
(TECH MAP). LLM re-ranking: +0.09 on the weak retriever, +0.00–0.02 on ConTeXT, at ~1 s and $0.11 per
1,000 phrases. Champion = ConTeXT; model selection was done on validation only.

### 10. P2 — reject option
Threshold 0.64 on cosine (tuned on TECH validation): joint accuracy 0.60 TECH / 0.50 HOUSE; the LLM
selector over the top-10 is 0.58 / 0.52 with higher NIL precision but lower recall. The free threshold is
deployed; the LLM is a fallback for low-confidence phrases.

### 11. P3 — agents
Figure `p3_ranking_quality_agent` (and `p3_efficiency_agent`). Same tools, same Haiku model, 30 CVs × 10
labelled jobs: embedding baseline nDCG 0.89 → planner→retriever→matcher pipeline 0.89 → single ReAct agent
0.96. Step-by-step reading of each posting (get_job + compare_skills) is what buys the +0.07; the one-shot
matcher is no better than cosine similarity. Cost: $0.003–0.010 per resume, 20–30 s. Show one trace from
`results/p3/traces/single_verified_agent/` (tool calls → final JSON).

### 12. P3 — faithfulness: every claim is a quote
Figure `p3_hallucination_agent` + one annotated example. Without a verifier, 17.8 % of matched/missing-skill
claims are unsupported — for both arms, and 23 % with Sonnet: model size does not fix it. The verifier
(deterministic quote check + one repair call per flagged job) flags 5.6 claims per resume, repairs 1.1,
drops 4.6 → **0 % unsupported**, at +$0.002. Because it is a post-processor it composes with the ReAct
agent: **single agent + verifier = nDCG 0.96 and 0 % unsupported claims** (champion). Gold labels: Opus +
Sonnet, 91 % agreement, linear κ = 0.83, 27 adjudicated. Silver check on 1,000 CVs (`p3_silver_eval`):
both agents +0.04–0.05 nDCG over the baseline on the same 60 CVs.

### 13. Model ops
Figure `ops_dashboard`. FastAPI + Streamlit, Docker; one LLM wrapper with cache / JSONL log / budget stop
(every number in the report is reproducible from the cache); daily ingest into Chroma; drift report
(NOT_IN_ESCO share, similarity of new postings to the tagger's training data, unsupported-claim rate with a
2 % alert) and the emerging-skills review queue (`results/monitoring/emerging_skills.csv`). Total LLM spend
for the whole project: ≈ $14 of the $25 budget (gold labels $7.3, Sonnet ablation $2.4).

### 14. Conclusions
Small supervised models win extraction when evidence must be verbatim (tagger +0.19 F1 over few-shot Haiku,
5× fewer false spans, 0 % paraphrase vs 7 %); dense retrieval + a tuned threshold wins normalization
(ConTeXT MAP 0.54, reject option doubles joint accuracy); LLM re-ranking only pays on weak retrievers;
agents need a verifier to be trustworthy (18 % unsupported claims without it, 0 % with it) and step-by-step
tool use beats a one-shot matcher for ranking. Limitations: one seed / 4 epochs, LLM-made labels (κ 0.83),
30 labelled CVs, ESCO 1.1.0, 53 % live coverage. Next: more seeds/epochs, 100 human-labelled pairs, newer
ESCO, prompt caching for latency.
