# Skill-Match Copilot — project report (draft)

*ADSP 32018 NLP and Agentic AI · University of Chicago MS-ADS · Autumn 2026*
*Mohan (Kevin) Li — draft written from the build logs; every number below is traceable to a file under `results/`.*

> **How to read this draft.** Sections follow the grading rubric: problem → data and EDA → the three NLP
> problems (each with two competing approaches, a champion and a challenger) → model ops → conclusion.
> Figures live in `reports/figures/` as PNG (for slides) and HTML (interactive); captions are in
> `reports/figures/CAPTIONS.md`. Where a number is still being produced, the placeholder says so.

---

## 1. Problem and business framing

Job seekers and recruiters both reason about *skills*, but job postings and resumes express them in free
text, with inconsistent vocabulary ("k8s", "Kubernetes", "container orchestration"). Skill-Match Copilot is
an assistant that:

1. **extracts** skill and knowledge phrases from a posting or a resume (Problem 1, span tagging);
2. **normalizes** every phrase to the European Commission's ESCO taxonomy — or says that it is *not in
   ESCO* (Problem 2, retrieval / entity linking with a reject option);
3. **ranks** a candidate's jobs and **explains** every ranking with quoted evidence, through a tool-using
   LLM agent whose claims are verified against the source text (Problem 3, agentic matching).

Why it matters: ESCO-normalized skills are the common key that lets a product compare a resume to thousands
of postings, track emerging skills that the taxonomy does not know yet, and give an explanation a recruiter
can audit. The agent is the user-facing layer; the two upstream models are what make its answers
verifiable and cheap.

**Scope decisions** (full log: `docs/decisions.md`): one person built everything on a CPU-only cloud
workspace with a $25 LLM budget; all LLM calls go through one cached, logged wrapper; no real resume is used
(the demo resume is synthetic); raw job and CV text never enters git.

---

## 2. Data and exploratory analysis

Four public datasets and one live feed (`docs/data_schema.md` has the verified columns; `results/eda/asset_check.json` the row counts).

| Dataset | Used for | Size | Licence |
|---|---|---|---|
| **SkillSpan** (Zhang et al., NAACL 2022; GitHub copy) | Problem 1 train/dev/test (expert BIO tags for *skill* and *knowledge* spans) | 4,800 / 3,174 / 3,569 sentences; tech and house sources | CC BY 4.0 |
| **Skill-Extraction-benchmark** (TechWolf; read through WorkRB) | Problem 2 validation/test (annotated phrases → ESCO labels, incl. "not in ESCO") | TECH 293 val / 1,164 test phrases; HOUSE 231 / 891; TECHWOLF 588 test | MIT |
| **ESCO 1.1.0** skills | the target vocabulary | 13,891 skills | EC licence (attribution) |
| **Synthetic ESCO skill sentences** (TechWolf) | fine-tuning the in-house bi-encoder | 138,260 sentence–skill pairs | MIT |
| **Djinni recruitment dataset** (lang-uk) | job index for the matching evaluation; anonymised CVs as resumes | 141,897 English IT postings; 210,250 CVs | MIT |
| **open-apply-jobs** daily snapshot (2026-10-20) | live demo index, drift report | 1,162,694 postings; 21,157 data/AI/ML titles | — (raw text not redistributed) |

### 2.1 SkillSpan (notebook 01, figures `eda01_*`)

* Sentences are short — median 11 tokens, 99 % under 83 tokens — so a 128-token window loses nothing
  (`eda01_tokens_per_sentence`).
* **67–73 % of sentences contain no span at all** (train 67 %, dev 71 %, test 73 %): the test set is
  dominated by headers, benefits and boilerplate, so *over-extraction on empty sentences* is a metric in
  its own right (Section 3).
* Skill spans are multi-word verb phrases (median 4 tokens, "partner continuously with your many
  stakeholders"); knowledge spans are 1–2 tokens (languages, tools, degrees) (`eda01_span_length_hist`,
  `eda01_pos_patterns`). This asymmetry is why the two span types get **separate taggers**: only 1.7 % of
  tagged tokens carry both labels, above the 1 % threshold set in the plan for merging them
  (`results/eda/skillspan_summary.json`).
* The most frequent skill phrases are soft skills ("communication skills", "team player"); the most
  frequent knowledge phrases are "English", "JavaScript", "Python", "Java" (`eda01_top_*`).
* 19.7 % of sentences contain an anonymisation placeholder (`<ORGANIZATION>`, `<LOCATION>`), confirming the
  entity view (`eda01_entities`): named entities are already scrubbed, dates and numbers ("3+ years") remain.

### 2.2 Skill-Extraction benchmark and ESCO (notebook 02, figures `eda02_*`)

* **43 % of annotated phrases in TECH and 37 % in HOUSE cannot be mapped to ESCO 1.1.0** ("LABEL NOT
  PRESENT" or "UNDERSPECIFIED"; `eda02_label_status_by_subset`). The unmapped phrases are concrete tools
  and frameworks — ReactJS, Docker, Django, Spring, Azure, Node.js (`eda02_unmapped_terms`). A product that
  normalizes to ESCO therefore *needs* a reliable "not in ESCO" decision; it is not an edge case.
* The target space is a long tail: 768 distinct ESCO skills are used by the benchmark, the top 25 cover
  only 24 % of phrases (`eda02_top_esco_labels`). ESCO labels are 2–5-word verb phrases ("manage musical
  staff"), so lexical matching is weak by construction (`eda02_esco_label_length`).
* TECH sentences are shortest (median 10 tokens), HOUSE sentences longest (15–18 tokens); sentences carry
  1–3 phrases, a few list-style sentences 10+ (`eda02_sentence_length`, `eda02_spans_per_sentence`).

### 2.3 Djinni jobs and CVs (notebook 03, figures `eda03_*`)

* **141,897 postings and 210,250 profiles**, both already de-duplicated by the dataset authors (0 exact
  duplicates on title + description, 0 duplicate CV texts). Postings have a median of 234 words
  (IQR 159–330), profiles 227 words with a much longer tail (`eda03_text_length`); 0.9 % of profiles are
  shorter than 300 characters and are excluded from the labelled set.
* Missing values are concentrated in the CV free-text sections: only 34 % of profiles fill all four
  sections (experience, looking-for, highlights, CV) (`eda03_missing_values`).
* **Supply and demand are unbalanced by category** (`eda03_supply_demand`, `eda03_supply_demand_heatmap`):
  JavaScript is the largest category (17,903 postings, 1.9 profiles per posting), QA has 3.7 profiles per
  posting, DevOps only 0.5. A matcher has to rank *within* crowded categories, not just pick the category.
* Postings per month rise through 2021, decline through 2022 and partly recover in 2023
  (`eda03_postings_per_month`) — the index sample is stratified by category, not by time.
* In embedding space (all-MiniLM-L6-v2, UMAP) the ten largest categories form clear clusters and profiles
  occupy the same regions as postings (`eda03_umap_market_map`); adjacent categories (JavaScript ~ Node.js)
  are close (`eda03_category_similarity`), which is what the "related category" tier of the labelled
  pools relies on.
* spaCy's generic NER mislabels technologies as organisations and places ("JavaScript", "SQL", "AWS" as
  ORG; "React", "NoSQL" as GPE) (`eda03_entities`) — one more reason to train a domain tagger instead of
  reusing a general-purpose entity model.
* The PII scrubber removed **14,830** e-mails, phone numbers, URLs and handles from the CV texts before
  any further processing.
* The matching evaluation pool is a stratified sample of **6,000 postings** from the 10 largest categories
  (≥ 300 characters), and resumes are drawn from the same categories.

### 2.4 Live postings (notebook 04, figures `eda04_*`)

* One daily snapshot holds **1,162,694 open postings** from seven applicant-tracking systems; Workday
  (633k) and SmartRecruiters (229k) dominate (`eda04_postings_by_source`). 21,157 postings (1.8 %) have a
  data/AI/ML title; by title pattern they split into AI/LLM engineering (6,283), data engineering (4,993),
  data science (4,545), ML engineering (3,122) and data analysis (1,838) (`eda04_ai_role_families`).
* **Only 31 % of postings carry a description** (53 % of the data/AI/ML subset): the Greenhouse, Ashby,
  Lever, Rippling and Gem feeds return the body, Workday (9 %) and SmartRecruiters (3 %) listings are
  title-only (`eda04_description_coverage_by_source`). The live index therefore covers the 11,288 data/AI/ML
  postings with text, and coverage of Workday-heavy employers is a known gap for the ops plan.
* Postings with text are long — median 668 words, IQR 457–881 (`eda04_description_length`) — and
  HTML-formatted; Greenhouse delivers *escaped* HTML (`&lt;p&gt;`), which the cleaner now unescapes before
  parsing. Compensation fields are filled for 3.9 % of postings and the remote flag for 7 %, so the matcher
  cannot rely on structured metadata (`eda04_salary_missingness`).

**Data-cleaning summary.** HTML → text with block-aware newlines (`src/utils/text.strip_html`), PII
scrubbing (`src/utils/pii.py`), sentence splitting that keeps bullet items, de-duplication on
(title, description), minimum text length (300 characters for Djinni, 200 for live postings), and
Arrow-backed string columns so the 350k-row Djinni tables fit in 7 GB of RAM.

---

## 3. Problem 1 — skill extraction as span tagging

**Task.** Given a sentence from a posting, return the *skill* spans and the *knowledge* spans (SkillSpan
definitions). Evaluation on the 3,569-sentence SkillSpan test split with `seqeval` **strict span F1**
(exact boundaries), plus three diagnostics: *loose* F1 (any token overlap), **false positives on the
2,985–3,057 sentences that contain no span**, and for the LLM a **hallucination rate** (returned phrases
that do not occur verbatim in the sentence).

**Approaches** (`src/p1_extract/`):

| | Champion: fine-tuned token classifier | Challenger: few-shot LLM extractor | Baseline: dictionary |
|---|---|---|---|
| Model | `jjzha/jobbert-base-cased` (BERT pretrained on job ads), one BIO tagger per span type, 4 epochs, lr 3e-5, batch 16, length-bucketed batches, best dev epoch kept | `claude-haiku-5-5`, versioned prompt `p1_extract_v1` with 6 annotated examples, JSON output, thinking disabled | exact match of training-set phrases |
| Training time | 62 min (skill) + 32 min (knowledge) on 2 CPUs | none | none |
| Inference | 39 ms / sentence (CPU) | 754 ms median, $0.13 per 1,000 sentences | 0.02 ms |

**Results** (`results/p1/metrics_summary_test.csv`, figures `p1_f1_comparison_test`, `p1_false_positives_test`, `p1_f1_by_source_test`):

| Approach | Span type | Strict P | Strict R | **Strict F1** | Loose F1 | FP spans / 100 empty sentences | F1 tech | F1 house |
|---|---|---|---|---|---|---|---|---|
| Tagger (champion) | skill | 0.451 | 0.496 | **0.472** | 0.744 | 5.1 | 0.476 | 0.470 |
| Tagger (champion) | knowledge | 0.583 | 0.690 | **0.632** | 0.789 | 6.7 | 0.675 | 0.546 |
| Haiku few-shot (challenger) | skill | 0.230 | 0.377 | 0.285 | 0.535 | 29.8 | 0.274 | 0.295 |
| Haiku few-shot (challenger) | knowledge | 0.435 | 0.678 | 0.530 | 0.624 | 25.2 | 0.612 | 0.391 |
| Dictionary | skill | 0.089 | 0.159 | 0.114 | 0.347 | 34.3 | 0.129 | 0.102 |
| Dictionary | knowledge | 0.128 | 0.541 | 0.207 | 0.281 | 117.1 | 0.300 | 0.096 |

What the numbers say:

* **The fine-tuned tagger wins on every metric**, by +0.19 strict F1 for skills and +0.10 for knowledge
  over the LLM, and it is 5–6× less likely to invent a span on an empty sentence (5–7 vs 25–30 false
  spans per 100 empty sentences). The LLM's errors are mostly *boundary* errors ("CI infrastructure
  pipelines" for the gold "CI") and *spurious* spans in boilerplate (`results/p1/errors_test.csv`).
* The LLM's recall is competitive (0.68 on knowledge) but its precision is not: it returns 1,831 knowledge
  spans against 1,174 gold ones. **7.0 % of the phrases it returned do not occur verbatim in the sentence**
  (9.2 % for skills, 4.7 % for knowledge) — paraphrases, not extractions — which is disqualifying for a
  product that must quote evidence (`results/p1/llm_haiku_test_summary.json`).
* Loose F1 is 0.74–0.79 for the tagger against strict 0.47–0.63: most of the remaining errors are boundary
  disagreements on long verb-phrase skills, consistent with the EDA (median skill span 4 tokens).
* Knowledge is easier than skills for every approach (short noun phrases vs long verb phrases), and the
  house source is harder than tech for knowledge (0.55 vs 0.67), matching its longer, more varied sentences.
* Honest caveat: dev F1 was still rising at the fourth epoch (skills 0.40 → 0.48 → 0.50 → 0.50; knowledge
  0.56 → 0.59 → 0.60 → 0.60, `models/p1_tagger/*/train_summary.json`), and our strict F1 is below the
  roughly 0.6 that the SkillSpan paper reports for JobBERT with longer training. One seed was run; seeds
  43/44 and 8–10 epochs are the first thing to add with more compute.

**Champion for the pipeline:** the two taggers (`models/p1_tagger/{skill,knowledge}_seed42`), wrapped as
`SkillExtractor` with sliding windows and BIO repair. The LLM extractor stays as the challenger and as a
reference for the hallucination metric.

---

## 4. Problem 2 — normalizing phrases to ESCO (with a reject option)

**Task.** For an extracted phrase (in its sentence), retrieve the ESCO 1.1.0 skill it denotes, or decide
that it is not in ESCO. Two evaluations:

* **Ranking** on the TechWolf Skill-Extraction benchmark through **WorkRB** (MAP, R-precision@10, MRR;
  validation = TECH + HOUSE val, test = TECH, HOUSE, TECHWOLF). Model selection used *only* the validation
  split. `results/p2/workrb_{val,test}.csv`, figures `p2_workrb_{map,rp10,mrr}_{val,test}`.
* **"Not in ESCO" detection** on the benchmark's unmappable phrases (490 of 1,164 TECH test phrases, 323 of
  891 HOUSE): joint accuracy (right label *or* right rejection), NIL precision/recall/F1, accuracy on the
  mappable phrases. `results/p2/nil*.csv`, figures `p2_nil_*`.

**Approaches** (`src/p2_normalize/`):

| Key | Approach | What it is |
|---|---|---|
| `bm25` | lexical baseline | BM25 over lower-cased ESCO labels (WorkRB implementation) |
| `minilm` | dense, zero-shot | `all-MiniLM-L6-v2` sentence embeddings, cosine retrieval |
| `ours` | dense, **in-house fine-tuned** | MiniLM fine-tuned on the 138k synthetic sentence–skill pairs with MultipleNegativesRankingLoss (NO_DUPLICATES sampler), 1 epoch, batch 64, lr 2e-5, 37 min on CPU (`models/p2_biencoder/train_summary.json`) |
| `context` | dense, **published** | `TechWolf/ConTeXT-Skill-Extraction-base` (110M, trained on the same synthetic recipe at scale) |
| `rtl`, `rtl_context` | **retrieve-then-LLM** | top-k candidates from a dense encoder, then `claude-haiku-5-5` selects the matching labels (`prompts/p2_select_v1.yaml`); k ∈ {10, 30, 50} tuned on validation |

**Validation (model selection)** — MAP, `results/p2/workrb_val.csv`:

| Approach | TECH val | HOUSE val |
|---|---|---|
| BM25 | 0.424 | 0.264 |
| MiniLM zero-shot | 0.434 | 0.337 |
| Fine-tuned MiniLM (in-house) | 0.538 | 0.413 |
| Retrieve-then-LLM on fine-tuned MiniLM, k=50 | 0.641 | 0.535 |
| **ConTeXT (published)** | **0.675** | 0.548 |
| Retrieve-then-LLM on ConTeXT, k=50 | 0.625 | **0.558** |

Fine-tuning on synthetic pairs is worth +0.10 MAP over the zero-shot encoder after a single CPU epoch — the
validation loss was still falling (0.283 → 0.260), so more epochs would narrow the gap to ConTeXT, which is
the same idea trained at full scale. **Champion = ConTeXT** (best validation MAP, free, 20 ms per phrase on
CPU); **challenger = ConTeXT + Haiku re-ranking with k=50**. The in-house encoder is kept as the
reproducible, fully-owned alternative (`champion_encoder` in `configs/models.yaml` switches it in).

**Test** — `results/p2/workrb_test.csv`:

| Approach | MAP TECH | MAP HOUSE | MAP TECHWOLF | RP@10 TECH | MRR TECH | LLM cost (926 sentences) |
|---|---|---|---|---|---|---|
| BM25 | 0.275 | 0.160 | 0.177 | 0.384 | 0.341 | — |
| MiniLM zero-shot | 0.365 | 0.246 | 0.293 | 0.556 | 0.435 | — |
| Fine-tuned MiniLM (in-house) | 0.424 | 0.313 | 0.374 | 0.596 | 0.508 | — |
| Retrieve-then-LLM on fine-tuned MiniLM, k=50 | 0.516 | 0.421 | 0.452 | 0.699 | 0.600 | $0.10 |
| **ConTeXT (champion)** | 0.536 | 0.423 | 0.460 | **0.739** | **0.631** | — |
| Retrieve-then-LLM on ConTeXT, k=50 (challenger) | **0.537** | **0.443** | **0.479** | 0.721 | 0.622 | $0.10 |

* The ordering found on validation holds on all three test sets. Every dense model beats BM25 by a wide
  margin — ESCO labels are verb phrases, and a phrase like "k8s" shares no token with "Kubernetes".
* **LLM re-ranking helps a weak retriever a lot (+0.09 MAP on the in-house encoder) and a strong one
  barely (+0.00 to +0.02 MAP on ConTeXT, with lower RP@10 and MRR)**. It costs $0.11 per 1,000 sentences
  and about one second of latency per sentence versus 20 ms. For the product the champion stays the pure
  encoder; the LLM selector is the fallback for phrases the encoder is unsure about.
* TECHWOLF is the only test set whose labels were produced with the same synthetic recipe ConTeXT was
  trained on; the ranking is the same on the two human-annotated sets, so the conclusion does not depend on it.

**"Not in ESCO"** (`results/p2/nil.csv` for the champion encoder, `results/p2/nil_p2_biencoder.csv` for
the in-house encoder; figures `p2_nil_threshold_sweep`, `p2_nil_results`):

| Encoder | Approach | TECH joint acc. | TECH NIL-F1 (P / R) | TECH acc. on mapped | HOUSE joint acc. | HOUSE NIL-F1 (P / R) | Cost |
|---|---|---|---|---|---|---|---|
| ConTeXT | top-1, no reject (reference) | 0.338 | — | 0.583 | 0.286 | — | — |
| ConTeXT | **cosine threshold 0.64 (champion)** | **0.604** | 0.679 (0.64 / 0.72) | 0.516 | 0.495 | **0.619** (0.53 / 0.74) | free |
| ConTeXT | top-10 + Haiku decides (challenger) | 0.584 | **0.697** (0.71 / 0.69) | 0.510 | **0.515** | 0.584 (0.63 / 0.55) | $0.13 / 1,000 phrases, 1.1 s |
| in-house MiniLM | cosine threshold 0.70 | 0.578 | 0.660 (0.56 / 0.80) | 0.418 | 0.496 | 0.609 (0.47 / 0.87) | free |
| in-house MiniLM | top-10 + Haiku decides | 0.605 | 0.722 (0.70 / 0.75) | 0.502 | 0.529 | 0.617 (0.62 / 0.61) | $0.13 / 1,000 phrases |

* Without a reject option, a top-1 linker is right on only 34 % (TECH) / 29 % (HOUSE) of phrases, because
  42 % / 36 % of them have no ESCO label at all. A single cosine threshold, tuned on the TECH *validation*
  spans (`p2_nil_threshold_sweep`), nearly doubles joint accuracy (0.60 / 0.50) at zero cost.
* With the stronger ConTeXT encoder, the threshold and the LLM decision are a wash: the LLM is more precise
  about rejecting (0.71 vs 0.64 NIL precision on TECH) but rejects less often, and it costs $0.13 per
  1,000 phrases plus a second of latency. The LLM's advantage is larger on the weaker in-house encoder —
  the same pattern as in the ranking task.
* Accuracy on the mappable phrases drops when a threshold is applied (0.58 → 0.52 on TECH): the price of
  rejecting is rejecting some true mentions. The pipeline keeps the thresholded encoder as champion and
  surfaces the rejected phrases to the review queue (Section 6), where the LLM can be run on demand.

---

## 5. Problem 3 — matching and explaining with a tool-using agent

**Task.** Given a resume, return the top-10 jobs from a pool with, for each job, a fit score, the matched
and missing skills and **verbatim quotes** from the resume and the posting that support every claim.

**Ground truth.** 300 resume–job pairs: 30 Djinni CVs (3 per category × the 10 largest categories) × a
fixed pool of 10 indexed postings each (4 from the same category, 3 from a related one, 3 random), graded
0 / 1 / 2 (not a fit / partial / strong) by two independent LLM raters following a written guideline
(`data/labels/guideline.md`, `prompts/label_fit_v1.yaml`): `claude-opus-5-5` (rater A) and
`claude-sonnet-5-5` (rater B). Results (`data/labels/kappa.json`, `data/labels/final.csv`):

* Agreement 91 %, Cohen's κ = 0.81 (unweighted), **0.83 (linear)**, 0.87 (quadratic) — "almost perfect"
  on the usual scale, and the 27 disagreements are all one level apart (never 0 vs 2). Opus adjudicated
  them with a written rationale (`prompts/label_adjudicate_v1.yaml`).
* Final distribution: **205 not-a-fit / 82 partial / 13 strong**. Labels track the sampling tiers: random
  pairs are 86/90 "0", related pairs 80/90 "0", same-category pairs 39 "0" / 69 "1" / 12 "2". Only 9 of
  the 30 CVs have a strong fit in their pool, and 2 CVs have no relevant job at all — so the ranking task
  is mostly *partial fit above non-fit*; we therefore report Precision@3 and MRR both with "strong" and
  with "partial or better" as the relevance threshold, and nDCG@10 (graded) over the 28 pools that contain
  at least one relevant job.
* Cost of the gold set: $7.28 (Opus $4.52, Sonnet $2.40, adjudication $0.36); ~15 minutes wall time. A
  1,000-CV *silver* evaluation (relevance = same job category, over the whole 6,000-posting index) checks
  that the conclusions hold at scale. Limitation (decision 6): labels are model-made, not human-made; the
  labelling kit (`data/processed/labeling_sheet.xlsx`) exists so that a human pass can replace them.

**Approaches** (`src/p3_agent/`). All arms share the same five tools (`extract_skills`, `normalize_skills`,
`search_jobs`, `get_job`, `compare_skills`), the same model (`claude-haiku-5-5`, thinking off), the same versioned
prompts and the same on-disk cache:

| Key | Approach | How it works |
|---|---|---|
| `baseline` | embedding baseline, no LLM | embed the resume (all-MiniLM-L6-v2), cosine-rank the pool |
| `single` | **single ReAct agent** (LangGraph loop) | one agent calls tools until it answers; up to 12 steps |
| `multi` | **planner → retriever → matcher → verifier** | planner (LLM) writes role family, must-haves and search queries; retriever (code) runs the tools; matcher (LLM) ranks and explains; verifier checks every claim deterministically, sends flagged ones to one LLM repair call per job and drops what remains unsupported |
| `multi_noverify` | ablation | the same pipeline with the verifier removed |
| `single_verified` | **single ReAct agent + the same verifier** | added after the first results showed the two arms win on different axes |

**Metrics.** nDCG@10 (graded gains 0/1/2) over the 28 pools with a relevant job; Precision@3 and MRR with
"strong fit" and with "partial or better" as relevance; *coverage* (share of the 10 pool jobs the agent actually
ranked — unranked jobs are appended in baseline order); **faithfulness** = share of matched/missing-skill claims
whose quotes are verbatim in the resume/posting *and* name the skill (automatic checker,
`src/p3_agent/faithfulness.py`, 20 unit tests); LLM calls, tokens, cost and latency per resume.

**Results on the gold pools** (`results/p3/metrics_agent.csv`, figures `p3_ranking_quality_agent`,
`p3_hallucination_agent`, `p3_efficiency_agent`):

| Approach | nDCG@10 | P@3 ≥ partial | MRR ≥ partial | P@3 strong | MRR strong | Coverage | Claims / CV | **Unsupported claims** | LLM calls | Cost / CV | Latency |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Embedding baseline | 0.886 | 0.678 | 0.920 | 0.122 | 0.704 | 1.00 | — | — | 0 | $0 | 0.7 s |
| Planner→…→Matcher, no verifier | 0.887 | 0.733 | 0.896 | 0.111 | 0.731 | 0.99 | 30.3 | 17.8 % | 2.0 | $0.003 | 12.7 s |
| Planner→…→Matcher→Verifier | 0.887 | 0.733 | 0.896 | 0.111 | 0.731 | 0.99 | 26.0 | **0.0 %** | 5.9 | $0.004 | 20.7 s |
| Single ReAct agent | **0.961** | **0.767** | **0.982** | **0.133** | **0.889** | 0.97 | 31.5 | 17.8 % | 4.0 | $0.008 | 20.3 s |
| **Single ReAct agent + Verifier** | **0.961** | **0.767** | **0.982** | **0.133** | **0.889** | 0.97 | 27.0 | **0.0 %** | 8.2 | $0.010 | 30.2 s |

What the numbers say:

* **Ranking.** The single ReAct agent is the best ranker: nDCG 0.961 vs 0.886 for the embedding baseline, and
  it puts a partial-or-better fit at or next to the top for almost every resume (MRR 0.98). The planner→matcher pipeline, which
  asks one LLM call to order all ten candidates at once, is no better than the embedding baseline (0.887):
  reading each posting with `get_job` + `compare_skills` and scoring it step by step is what buys the +0.07.
  Per category (`results/p3/per_cv_agent.csv`) the single agent wins or ties everywhere; the pipeline loses to
  the baseline on "Other", Node.js and QA, where its one-shot matcher misorders the low-fit tail.
* **Faithfulness.** Without a verifier, **about one claim in six is unsupported** — 17.8 % for both the ReAct
  agent and the matcher. The reasons (`results/p3/failures_agent.csv`) are mostly naming, not fabrication:
  the skill is not named in the quoted text (`skill_not_in_quotes`, 75 of 172 single-agent cases) or a
  "missing" skill is not in the posting at all (`skill_not_in_job`, 68); invented quotes are rare
  (`job_quote_not_found` 16, `resume_quote_not_found` 3). The verifier flags 5.6 claims per resume, repairs
  1.1 with a correct verbatim quote and drops 4.6, which takes the unsupported share to 0 % at a cost of
  ~15 % fewer claims and +$0.002 per resume.
* **Combination.** Because the verifier is a post-processor, it composes with either orchestration: *single
  ReAct agent + verifier* keeps the best ranking **and** reaches 0 % unsupported claims. It is the
  champion for the product; the planner→matcher→verifier pipeline is the cheaper fallback (half the input
  tokens, since it never re-reads the conversation).
* **Cost and latency.** Everything is cheap at Haiku prices — the whole 30-resume evaluation of five arms cost
  under $1 — but latency is 20–30 s per resume for the agentic arms versus 0.7 s for the baseline, which is
  why the baseline remains the first-pass ranker in the API and the agent is invoked for the top-k only.

**Model ablation** — the same evaluation on the first 12 CVs with `claude-sonnet-5-5`
(`results/p3/metrics_sonnet.csv`, same 12 CVs with Haiku in brackets):

| Approach (12 CVs) | nDCG@10 | MRR ≥ partial | Unsupported claims | Cost / CV |
|---|---|---|---|---|
| Single ReAct agent — Sonnet | 0.947 [Haiku 0.957] | 0.955 [0.955] | 22.8 % [20.2 %] | $0.117 [$0.007] |
| Planner→…→Verifier — Sonnet | **0.965** [0.902] | **1.000** [0.955] | 0.0 % [0.0 %] | $0.076 [$0.004] |

The larger model lifts the one-shot matcher (0.902 → 0.965) more than the step-by-step agent, which is already
near the ceiling with Haiku, and it does **not** reduce unsupported claims (22.8 % vs 20.2 %): faithfulness
comes from the verifier, not from model size. At 16× the price, Sonnet is not worth it for this task.

**Silver check at scale** — 1,000 Djinni CVs against the whole 6,000-posting index, relevance = same job
category (`results/p3/silver_metrics.csv`, figure `p3_silver_eval`; agents on the first 60 CVs, baseline
numbers on the same 60 in brackets): baseline nDCG@10 0.777 [0.747], single agent 0.787, pipeline 0.794;
P@10 0.773 [0.742] / 0.787 / 0.792. Both agents beat the embedding ranker by +0.04–0.05 nDCG on this rough
signal over a realistic pool, with no run errors; the two agents are within noise of each other here, as
they are on the gold set's partial-fit metrics.

**Failure analysis** (`results/p3/failures_agent.csv`, traces in `results/p3/traces/`): with the verifier in
place, the remaining failures are ranking errors — a non-strong job at rank 1 while a strong fit exists
(2 of 9 such CVs for the verified single agent, 4 for the pipeline) and one missed strong fit in a pool with
three of them. The dropped claims are almost all of the "skill not named in the quote" kind — e.g. a matched
skill "Unit testing" supported by quotes that talk about test automation in general — which is exactly the
over-generalisation an auditor would reject.

---

## 6. Model ops

See `docs/architecture.md` for the diagram and the maintenance plan. In brief:

* **Serving**: FastAPI (`src/api/app.py`: `/extract`, `/normalize`, `/search`, `/match`, `/health`) and a
  Streamlit demo (`src/ui/app.py`); `Dockerfile` builds the service. Models load once at start-up.
* **LLM layer**: one wrapper (`src/utils/llm.py`) with an on-disk cache keyed by (model, prompt version,
  input), a JSONL log with tokens, latency and cost per call, and a hard budget stop. Prompts are versioned
  files; a prompt change re-runs the fixed evaluation sets before it is merged.
* **Index refresh**: `daily_ingest.py` re-pulls the live snapshot and upserts new postings into Chroma
  (idempotent; `0 6 * * *`).
* **Monitoring**: `drift_report.py` tracks the NOT_IN_ESCO share (new skills), the similarity of new
  postings to the tagger's training distribution, the unsupported-claim rate of agent answers (alert above
  2 %), and feeds frequent unmapped phrases into a human review queue (`results/monitoring/emerging_skills.csv`).
* **Spend**: LLM cost of the whole project ≈ $14.3 of the $25 budget (`logs/llm_calls.jsonl`): gold labels $7.3, Sonnet ablation $2.4, everything else (P1 extractor, P2 re-ranking and reject decisions, Haiku agents, silver set) ≈ $3.5.

---

## 7. Conclusion

* **Extraction (P1).** A 110M-parameter tagger fine-tuned for an hour on CPU beats a frontier LLM used
  few-shot by 0.10–0.19 strict F1, over-extracts 5× less and never paraphrases, whereas 7 % of the LLM's
  "extractions" are not in the text. For span extraction that must quote evidence, the small supervised
  model is the right tool; the LLM is the better *reader* only when recall matters more than precision.
* **Normalization (P2).** Dense retrieval is mandatory (BM25 is 2× worse on verb-phrase labels);
  fine-tuning on synthetic pairs pays off (+0.10 MAP after one CPU epoch) but the published ConTeXT encoder,
  trained on the same recipe at scale, is the champion (MAP 0.54 / 0.42 / 0.46). LLM re-ranking only pays
  on a weak retriever (+0.09 MAP) and adds ≤ 0.02 on a strong one at 50× the latency. The reject option is a
  first-class problem — 37–43 % of real phrases are not in ESCO — and a single tuned cosine threshold
  nearly doubles joint accuracy for free.
* **Matching (P3).** Step-by-step tool use (ReAct) ranks better than a one-shot planner→matcher pipeline
  (nDCG 0.96 vs 0.89, baseline 0.89), but *both* make unsupported claims one time in six. Faithfulness does
  not come from a bigger model (Sonnet: 23 % unsupported) but from a verifier that enforces the metric
  (verbatim quotes that name the skill): it removes every unsupported claim at +$0.002 per resume. The
  product champion is therefore **single ReAct agent + verifier**: nDCG 0.96, 0 % unsupported claims,
  $0.01 and 30 s per resume with Haiku.
* **Across the three problems** the same pattern holds: a measured small model or a deterministic check
  beats "ask a bigger LLM" on precision and cost, and the LLM earns its place where judgement over several
  pieces of evidence is needed (ranking, repairing a claim with the right quote).

**Limitations.** One training seed and four epochs for the tagger (dev F1 still rising); gold fit labels
are model-made (two raters, κ = 0.83, but no human pass yet); 30 labelled CVs make the agent comparison
indicative rather than conclusive (nDCG std 0.07–0.17); ESCO 1.1.0 is not the newest release; the live
index only sees postings whose ATS exposes the body (53 % of data/AI/ML postings); JobBERT has no explicit
licence. **Next steps:** seeds 43/44 and 8–10 epochs for the taggers; more bi-encoder epochs; a 100-pair
human-labelled subset to validate the LLM labels; newer ESCO; prompt-cached agent calls to cut latency.

---

## 8. Reproducibility

```bash
python scripts/download_assets.py        # on a machine that reaches huggingface.co (Kevin's Mac)
python scripts/import_transfer.py        # in the build workspace
scripts/run_stages.sh                    # resumable: every stage writes logs/runs/<stage>.done
python -m pytest tests                   # 31 tests
```

Every results table has a `.meta.json` sibling with the git commit, timestamp and configuration; every
figure has a caption in `reports/figures/CAPTIONS.md`; `PROGRESS.md` is the build log.
