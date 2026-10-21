# Figure captions

Every figure exists as `<name>.png` (for slides/report) and `<name>.html` (interactive).

| # | File | Caption | Source |
|--:|------|---------|--------|
| 1 | `eda01_sentences_by_split_source.png` | SkillSpan: sentences per split (train 4,800 / dev 3,174 / test 3,569), coloured by source (tech vs house). | notebooks/01_eda_skillspan |
| 2 | `eda01_tokens_per_sentence.png` | Tokens per sentence by source; 99% of sentences are shorter than the 128-token training limit. | notebooks/01_eda_skillspan |
| 3 | `eda01_span_length_hist.png` | Span length distribution (tokens): skill spans are multi-word verb phrases (median 4), knowledge spans are mostly 1–2 tokens. | notebooks/01_eda_skillspan |
| 4 | `eda01_top_skill_phrases.png` | The 30 most frequent skill phrases in the training split — dominated by soft skills ('communication skills', 'team player'). | notebooks/01_eda_skillspan |
| 5 | `eda01_top_knowledge_phrases.png` | The 30 most frequent knowledge phrases in the training split — programming languages, frameworks and degrees. | notebooks/01_eda_skillspan |
| 6 | `eda01_pos_patterns.png` | Part-of-speech patterns of spans: skills are VERB+… phrases or NOUN+NOUN ('communication skills'); knowledge is PROPN/NOUN heads. | notebooks/01_eda_skillspan |
| 7 | `eda01_ngrams_tech_vs_house.png` | Top bigrams per source: tech postings are about software development, experience and flexible working; house postings mix application boilerplate ('further information', 'please contact') with business domains ('supply chain', 'project management'). | notebooks/01_eda_skillspan |
| 8 | `eda01_entities.png` | spaCy entity labels in SkillSpan sentences: ORG/GPE are mostly scrubbed to placeholders; DATE and CARDINAL ('3+ years') dominate what remains. | notebooks/01_eda_skillspan |
| 9 | `eda01_tagged_example.png` | A SkillSpan sentence with its merged BIO tags (blue = skill, orange = knowledge). | notebooks/01_eda_skillspan |
| 10 | `eda02_label_status_by_subset.png` | Label status per benchmark subset: in TECH ads about 43% of annotated skill phrases cannot be mapped to ESCO (mostly LABEL NOT PRESENT) and 37% in HOUSE — the gap the 'not in ESCO' detector targets; TECHWOLF has sentence-level ESCO labels only. | notebooks/02_eda_esco_benchmark |
| 11 | `eda02_top_esco_labels.png` | The most frequent ESCO target skills: a long tail — the top 25 skills cover only a small share of phrases, so retrieval must handle thousands of rare labels. | notebooks/02_eda_esco_benchmark |
| 12 | `eda02_esco_label_length.png` | Token length of the ~13.9k ESCO skill labels: most are 2–5-word verb phrases ('manage musical staff'), which is why a bi-encoder (not keyword match) is needed. | notebooks/02_eda_esco_benchmark |
| 13 | `eda02_sentence_length.png` | Sentence length per subset: TECH sentences are shortest (median ~10 tokens), HOUSE sentences the longest (median 15–18 tokens). | notebooks/02_eda_esco_benchmark |
| 14 | `eda02_spans_per_sentence.png` | Number of annotated skill phrases per sentence: most sentences carry 1–3 phrases; a few list-style sentences carry 10+. | notebooks/02_eda_esco_benchmark |
| 15 | `eda02_unmapped_terms.png` | Phrases annotators could not map to ESCO 1.1.0: concrete frameworks, tools and platforms (ReactJS, Docker, Django, Spring, Azure, Node.js…) — exactly the vocabulary a job-matching product must track outside the taxonomy. | notebooks/02_eda_esco_benchmark |
| 16 | `p1_f1_comparison_test.png` | Problem 1: strict and loose span F1 per approach and span type on the SkillSpan test split (identical sentence set for all approaches; tagger = mean over seeds). | src/p1_extract/evaluate.py |
| 17 | `p1_false_positives_test.png` | Problem 1: over-extraction on sentences without any gold span (headers, benefits) — how often each approach invents a span where annotators found none. | src/p1_extract/evaluate.py |
| 18 | `p1_f1_by_source_test.png` | Problem 1: strict span F1 split by posting source (tech vs house) — the domain gap each approach suffers. | src/p1_extract/evaluate.py |
| 19 | `p2_workrb_map_val.png` | Problem 2: WorkRB MAP per task on the val split — lexical and dense baselines, the in-house fine-tuned MiniLM, the published ConTeXT encoder (champion) and retrieve-then-LLM re-ranking (challenger). | src/p2_normalize/run_workrb.py |
| 20 | `p2_workrb_rp10_val.png` | Problem 2: WorkRB RP@10 per task on the val split — lexical and dense baselines, the in-house fine-tuned MiniLM, the published ConTeXT encoder (champion) and retrieve-then-LLM re-ranking (challenger). | src/p2_normalize/run_workrb.py |
| 21 | `p2_workrb_mrr_val.png` | Problem 2: WorkRB MRR per task on the val split — lexical and dense baselines, the in-house fine-tuned MiniLM, the published ConTeXT encoder (champion) and retrieve-then-LLM re-ranking (challenger). | src/p2_normalize/run_workrb.py |
| 22 | `p2_workrb_map_test.png` | Problem 2: WorkRB MAP per task on the test split — lexical and dense baselines, the in-house fine-tuned MiniLM, the published ConTeXT encoder (champion) and retrieve-then-LLM re-ranking (challenger). | src/p2_normalize/run_workrb.py |
| 23 | `p2_workrb_rp10_test.png` | Problem 2: WorkRB RP@10 per task on the test split — lexical and dense baselines, the in-house fine-tuned MiniLM, the published ConTeXT encoder (champion) and retrieve-then-LLM re-ranking (challenger). | src/p2_normalize/run_workrb.py |
| 24 | `p2_workrb_mrr_test.png` | Problem 2: WorkRB MRR per task on the test split — lexical and dense baselines, the in-house fine-tuned MiniLM, the published ConTeXT encoder (champion) and retrieve-then-LLM re-ranking (challenger). | src/p2_normalize/run_workrb.py |
| 25 | `p2_nil_threshold_sweep_p2_biencoder.png` | Tuning the 'not in ESCO' cosine threshold on the TECH validation spans: joint accuracy, NIL-decision F1 and accuracy on mapped phrases vs threshold. | src/p2_normalize/nil_eval.py |
| 26 | `p2_nil_results_p2_biencoder.png` | 'Not in ESCO' detection on TECH and HOUSE test spans: joint accuracy, F1 of the NIL decision and accuracy on mapped phrases for the thresholded in-house fine-tuned MiniLM bi-encoder vs retrieve-then-LLM (challenger). | src/p2_normalize/nil_eval.py |
| 27 | `eda03_missing_values.png` | Djinni: share of missing/empty values per field for job postings and candidate profiles (CV free-text fields such as 'Looking For' and 'Highlights' are often empty). | notebooks/03_eda_djinni |
| 28 | `eda04_postings_by_source.png` | Live snapshot: number of postings per applicant-tracking-system source (Ashby, Greenhouse, Lever, …). | notebooks/04_eda_live_jobs |
| 29 | `eda04_salary_missingness.png` | Salary availability in the live snapshot: most postings leave compensation fields empty. | notebooks/04_eda_live_jobs |
| 30 | `eda04_top_titles.png` | Most frequent job titles across the whole live snapshot. | notebooks/04_eda_live_jobs |
| 31 | `eda04_departments.png` | Live snapshot sample: distribution of department. | notebooks/04_eda_live_jobs |
| 32 | `eda04_employment_type.png` | Live snapshot sample: distribution of employment type. | notebooks/04_eda_live_jobs |
| 33 | `eda04_description_length.png` | Length of live postings after HTML stripping (postings with text only): median 668 words — several times longer than a Djinni posting. | notebooks/04_eda_live_jobs |
| 34 | `eda04_posting_dates.png` | Posting dates in the sample: the snapshot contains postings that are still open, so dates spread over several months with a recent peak. | notebooks/04_eda_live_jobs |
| 35 | `eda04_ai_role_families.png` | The live demo index: data/AI/ML postings grouped into role families by title pattern. | notebooks/04_eda_live_jobs |
| 36 | `eda04_description_coverage_by_source.png` | Only the Greenhouse, Ashby, Lever, Rippling and Gem feeds return posting bodies; Workday and SmartRecruiters listings are title-only, so the live index can only cover postings with text. | notebooks/04_eda_live_jobs |
| 37 | `p2_nil_threshold_sweep.png` | Tuning the 'not in ESCO' cosine threshold on the TECH validation spans: joint accuracy, NIL-decision F1 and accuracy on mapped phrases vs threshold. | src/p2_normalize/nil_eval.py |
| 38 | `p2_nil_results.png` | 'Not in ESCO' detection on TECH and HOUSE test spans: joint accuracy, F1 of the NIL decision and accuracy on mapped phrases for the thresholded bi-encoder (champion) vs retrieve-then-LLM (challenger). | src/p2_normalize/nil_eval.py |
| 39 | `eda03_text_length.png` | Djinni text lengths: job postings are typically 150–400 words; CVs are much more variable, with a long tail of multi-page profiles and a spike of one-liners. | notebooks/03_eda_djinni |
| 40 | `eda03_supply_demand.png` | Supply–demand by category: number of candidate profiles vs job postings for the 20 largest job categories on Djinni. | notebooks/03_eda_djinni |
| 41 | `eda03_supply_demand_heatmap.png` | Supply–demand heatmap: relative job postings, candidate profiles and CVs-per-posting ratio per category — where competition for a job is highest. | notebooks/03_eda_djinni |
| 42 | `eda03_experience_years.png` | Years of experience required by postings vs reported by candidates — postings cluster at 1–3 years, profiles have a longer tail. | notebooks/03_eda_djinni |
| 43 | `eda03_english_level.png` | English level required by postings vs reported by candidates. | notebooks/03_eda_djinni |
| 44 | `eda03_postings_per_month.png` | Monthly job postings per category on Djinni, 2020–2023: growth through 2021, a steady decline through 2022 and a partial recovery in 2023 (the last month of the dataset is incomplete). | notebooks/03_eda_djinni |
| 45 | `eda03_wordclouds.png` | Word clouds of job postings for the six largest categories (800 postings each): the vocabulary differs sharply between stacks. | notebooks/03_eda_djinni |
| 46 | `eda03_umap_market_map.png` | Market map: UMAP projection of all-MiniLM-L6-v2 embeddings for 5,000 job postings and 5,000 candidate profiles (10 largest categories) — categories form clear clusters, and profiles occupy the same regions as postings. | notebooks/03_eda_djinni |
| 47 | `eda03_category_similarity.png` | Cosine similarity between category centroids of posting embeddings — used to define 'related' categories when sampling the labelled job pools. | notebooks/03_eda_djinni |
| 48 | `eda03_entities.png` | spaCy entity labels in Djinni job descriptions: ORG (companies, frameworks mis-typed as organisations), GPE (locations) and DATE dominate. | notebooks/03_eda_djinni |
| 49 | `p3_silver_eval.png` | Problem 3 silver set: ranking quality with 'same job category' as a rough relevance signal, over the whole 6k-posting index (baseline on all sampled CVs, agents on a subset). | src/p3_agent/silver_eval.py |
| 50 | `eda05_synthetic_length.png` | TechWolf's synthetic ESCO sentences are uniformly short (one or two clauses), unlike real job-ad sentences — a domain gap for the bi-encoder. | notebooks/05_eda_generated |
| 51 | `eda05_synthetic_openings.png` | Template bias in the generated training sentences: a handful of openings ('experience in …', 'ability to …') dominate, which the NO_DUPLICATES batch sampler and real-sentence evaluation guard against. | notebooks/05_eda_generated |
| 52 | `eda05_llm_phrases_per_sentence.png` | Over-/under-generation of the few-shot LLM extractor: distribution of phrases per sentence compared with the gold annotation. | notebooks/05_eda_generated |
| 53 | `eda05_llm_phrase_length.png` | Length of LLM-returned phrases: skill phrases are long, knowledge phrases short — the same shape as the gold spans. | notebooks/05_eda_generated |
| 54 | `eda05_rtl_selected.png` | Retrieve-then-LLM: number of ESCO candidates the LLM accepts per sentence; zero means it judged none of the retrieved skills a true mention. | notebooks/05_eda_generated |
| 55 | `eda05_fit_by_relation.png` | Bias check: distribution of agent fit scores for same-, related- and random-category jobs — scores should fall across tiers; overlap shows where the agents are over-generous. | notebooks/05_eda_generated |
| 56 | `eda05_fit_by_label.png` | Calibration check: agent fit scores against the gold 0/1/2 labels. | notebooks/05_eda_generated |
| 57 | `eda05_unsupported_reasons.png` | Failure modes of generated explanations: reason codes of unsupported claims (quote not found in resume / job, skill not in the quotes, missing skill actually present). | notebooks/05_eda_generated |
| 58 | `eda05_rater_agreement.png` | Agreement between the two LLM raters on the 300 resume–job fit labels (confusion matrix; disagreements were adjudicated). | notebooks/05_eda_generated |
