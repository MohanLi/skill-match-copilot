# Jordan Reyes — Machine Learning Engineer

Chicago, IL · open to remote · [EMAIL] · [PHONE] · [URL]

## Summary
Machine learning engineer with 3 years of experience building NLP and recommendation systems in production.
Comfortable owning a feature end to end: data pipelines, model training, evaluation, deployment and monitoring.
Looking for an ML / applied-scientist role on a product team that ships weekly.

## Skills
- **Languages:** Python (expert), SQL, Bash, some Scala
- **ML / NLP:** PyTorch, Hugging Face Transformers, sentence-transformers, scikit-learn, XGBoost, spaCy,
  retrieval-augmented generation (RAG), prompt engineering with the OpenAI and Anthropic APIs, LangGraph
- **Data:** pandas, PySpark, Airflow, dbt, PostgreSQL, BigQuery, Snowflake
- **MLOps / infra:** Docker, Kubernetes (EKS), MLflow, FastAPI, GitHub Actions, AWS (S3, SageMaker, Lambda), Terraform basics
- **Other:** A/B testing, experiment design, vector databases (FAISS, Chroma, pgvector), Streamlit

## Experience
**Machine Learning Engineer — Brightline Commerce (Chicago, IL)** · Jun 2023 – present
- Built a semantic product-search service with a fine-tuned bi-encoder (sentence-transformers) and FAISS; lifted
  search conversion by 7.4 % in an A/B test across 2.1 M monthly users.
- Designed and shipped a RAG assistant for customer support on top of Claude, with retrieval over 40k help-center
  chunks; cut median handle time by 18 %. Added an automatic faithfulness check that flags unsupported answers.
- Own the training and serving stack: PyTorch models tracked in MLflow, packaged with Docker, served through
  FastAPI on Kubernetes, with drift dashboards in Grafana.
- Wrote the team's evaluation harness (offline metrics + LLM-as-judge) and the on-call runbook for model incidents.

**Data Scientist — Northwind Analytics (remote)** · Jul 2021 – May 2023
- Built demand-forecasting models (XGBoost, Prophet) for 300+ retail SKUs; reduced stock-outs by 11 %.
- Created an Airflow + dbt pipeline on BigQuery that replaced a nightly spreadsheet process; 6 analysts use it daily.
- Developed a text-classification model (DistilBERT) to route 50k support tickets a month with 93 % accuracy.

## Projects
- **skill-match-copilot** — resume–job matching agent (skill extraction with a fine-tuned JobBERT tagger,
  ESCO normalization, LangGraph multi-agent ranking with a verifier). Python, PyTorch, Chroma, FastAPI, Streamlit.
- **Papers-to-podcast** — summarizes arXiv papers into short audio briefs with speaker diarization.

## Education
- **M.S. Computer Science**, University of Illinois Urbana-Champaign, 2021
- **B.S. Statistics**, Purdue University, 2019

## Certifications
AWS Certified Machine Learning – Specialty (2024)
