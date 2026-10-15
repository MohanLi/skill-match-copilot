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
| 19 | `p2_workrb_map_val.png` | Problem 2: WorkRB MAP per task on the val split for lexical and dense baselines, the fine-tuned bi-encoder (champion) and retrieve-then-LLM (challenger). | src/p2_normalize/run_workrb.py |
| 20 | `p2_workrb_rp10_val.png` | Problem 2: WorkRB RP@10 per task on the val split for lexical and dense baselines, the fine-tuned bi-encoder (champion) and retrieve-then-LLM (challenger). | src/p2_normalize/run_workrb.py |
| 21 | `p2_workrb_mrr_val.png` | Problem 2: WorkRB MRR per task on the val split for lexical and dense baselines, the fine-tuned bi-encoder (champion) and retrieve-then-LLM (challenger). | src/p2_normalize/run_workrb.py |
