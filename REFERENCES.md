# References, datasets, models, tools and licenses

## Datasets

| Dataset | Used for | License / terms | Citation |
|---------|----------|-----------------|----------|
| **SkillSpan** — https://github.com/kris927b/SkillSpan (data), https://huggingface.co/datasets/jjzha/skillspan | Problem 1 train/dev/test; EDA 01 | CC BY 4.0 | Zhang, M., Jensen, K. N., Sonniks, S., & Plank, B. (2022). *SkillSpan: Hard and Soft Skill Extraction from English Job Postings.* NAACL 2022. https://aclanthology.org/2022.naacl-main.366/ |
| **Skill-Extraction-benchmark** (TECH / HOUSE / TECHWOLF) — https://github.com/jensjorisdecorte/Skill-Extraction-benchmark; HF copies `TechWolf/skill-extraction-{tech,house,techwolf}` | Problem 2 evaluation (via WorkRB), "not in ESCO" evaluation; EDA 02 | CC BY 4.0 (Hugging Face copies); the GitHub repo has no LICENSE file | Decorte, J.-J., Van Hautte, J., Deleu, J., Develder, C., & Demeester, T. (2022). *Design of Negative Sampling Strategies for Distantly Supervised Skill Extraction.* RecSys in HR 2022. https://ceur-ws.org/Vol-3218/RecSysHR2022-paper_4.pdf — Zhang, M., Jensen, K. N., van der Goot, R., & Plank, B. (2022). *Skill Extraction from Job Postings using Weak Supervision.* RecSys in HR 2022 (HOUSE/TECH annotations). |
| **Synthetic ESCO skill sentences** — https://huggingface.co/datasets/TechWolf/Synthetic-ESCO-skill-sentences | Problem 2 bi-encoder training | CC BY 4.0 | Decorte, J.-J., et al. (2023). *Extreme Multi-Label Skill Extraction Training using Large Language Models.* https://arxiv.org/abs/2307.10778 |
| **ESCO v1.1.0** (European Skills, Competences, Qualifications and Occupations) — https://esco.ec.europa.eu/en/use-esco/download | Skill taxonomy for Problem 2/3 | Free to reuse with attribution: *"This service uses the ESCO classification of the European Commission."* | European Commission, ESCO v1.1.0 (2022). |
| **Djinni recruitment dataset** — https://huggingface.co/datasets/lang-uk/recruitment-dataset-job-descriptions-english, https://huggingface.co/datasets/lang-uk/recruitment-dataset-candidate-profiles-english | Problem 3 job pool and resumes; EDA 03 | MIT | Lang-uk (2024). Djinni Recruitment Dataset (English). |
| **open-apply-jobs** — https://huggingface.co/datasets/edwarddgao/open-apply-jobs | Live job index (demo) and drift monitoring; EDA 04 | MIT packaging; posting text belongs to the employers (not redistributed in this repo) | Gao, E. (2025–2026). open-apply-jobs daily snapshots. |

## Models

| Model | Used for | License |
|-------|----------|---------|
| `jjzha/jobbert-base-cased` — https://huggingface.co/jjzha/jobbert-base-cased | Problem 1 champion (fine-tuned token classifier) | No license stated on the model card (research use; noted as a limitation) |
| `sentence-transformers/all-MiniLM-L6-v2` — https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 | Problem 2 zero-shot baseline and base of the in-house fine-tuned bi-encoder; job-text and query embeddings of the Chroma index | Apache-2.0 |
| `TechWolf/ConTeXT-Skill-Extraction-base` — https://huggingface.co/TechWolf/ConTeXT-Skill-Extraction-base | Problem 2 champion encoder (selected on WorkRB validation); evaluated via WorkRB | See model card |
| Anthropic Claude (`claude-haiku-5-5`, `claude-sonnet-5-5`, `claude-opus-5-5`) | Few-shot extractor, retrieve-then-LLM selector, agents, gold-standard labelling | Anthropic API terms |
| spaCy `en_core_web_sm` 3.8 | POS/NER features in EDA | MIT |

## Tools and frameworks

WorkRB (Apache-2.0, https://github.com/techwolf-ai/workrb) · Hugging Face `transformers`, `datasets`, `sentence-transformers` (Apache-2.0) · seqeval (MIT) · LangGraph (MIT) · Chroma (Apache-2.0) · FastAPI, Streamlit (MIT / Apache-2.0) · Plotly (MIT) · spaCy (MIT) · UMAP-learn (BSD) · wordcloud (MIT) · pandas / scikit-learn / NumPy (BSD).

## Reused code, prompts and ideas

- The `RetrieveThenLLM` model follows WorkRB's `ModelInterface` contract (`workrb/models/base.py`); our local-CSV task loaders mirror `workrb/tasks/ranking/skill_extraction.py` (Apache-2.0).
- Skill/knowledge definitions and span conventions in `prompts/p1_extract_v1.yaml` follow the SkillSpan annotation guidelines (Zhang et al., 2022).
- The LLM-vs-supervised hypothesis for Problem 1 follows Nguyen, K. C., et al. (2024). *Rethinking Skill Extraction in the Job Market Domain using Large Language Models.* NLP4HR 2024. https://aclanthology.org/2024.nlp4hr-1.3
- Chart palette and rules follow a validated, brand-neutral data-visualization reference (categorical hues in fixed order, one-hue sequential ramps).
- All prompts are our own and live in `prompts/` with version ids.
