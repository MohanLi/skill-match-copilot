"""20 hand-made cases (10 supported, 10 unsupported) for the faithfulness checker — the Phase 6 gate
in the project plan ('gets all 20 right')."""
from __future__ import annotations

import pytest

from src.p3_agent.faithfulness import check_explanation, check_matched, check_missing

RESUME = """Jordan Reyes — Machine Learning Engineer
Skills: Python (expert), SQL, PyTorch, Hugging Face Transformers, scikit-learn, Docker, Kubernetes (EKS), AWS (S3, SageMaker).
Built a semantic product-search service with a fine-tuned bi-encoder and FAISS.
Designed and shipped a RAG assistant for customer support on top of Claude.
Created an Airflow + dbt pipeline on BigQuery that replaced a nightly spreadsheet process."""

JOB = """Senior Machine Learning Engineer — Remote
Requirements:
- 4+ years of experience with Python and PyTorch
- Experience deploying models with Docker and Kubernetes
- Strong SQL and data-pipeline skills (Airflow or similar)
- Experience with Java is required for our legacy services
- Nice to have: Rust, Kafka streaming, GCP Vertex AI"""

RESUME_ESCO = ["Python (computer programming)", "SQL", "Docker", "Kubernetes", "use machine learning"]
JOB_ESCO = ["Python (computer programming)", "SQL", "Docker", "Kubernetes", "Java (computer programming)", "Rust"]

SUPPORTED = [
    # matched: verbatim quotes in both texts, skill in the quotes
    ("matched", {"skill": "Python", "resume_quote": "Python (expert)", "job_quote": "4+ years of experience with Python and PyTorch"}),
    ("matched", {"skill": "PyTorch", "resume_quote": "PyTorch, Hugging Face Transformers", "job_quote": "Python and PyTorch"}),
    ("matched", {"skill": "Docker", "resume_quote": "Docker, Kubernetes (EKS)", "job_quote": "deploying models with Docker and Kubernetes"}),
    ("matched", {"skill": "SQL", "resume_quote": "SQL, PyTorch", "job_quote": "Strong SQL and data-pipeline skills"}),
    # whitespace / line-wrap / case differences in quotes are tolerated
    ("matched", {"skill": "Airflow", "resume_quote": "created an airflow + dbt pipeline", "job_quote": "(Airflow or similar)"}),
    ("matched", {"skill": "Kubernetes", "resume_quote": "Kubernetes   (EKS)", "job_quote": "Docker and Kubernetes"}),
    # ESCO-level match: quotes verbatim, skill name not literally in quotes but in both ESCO sets
    ("matched", {"skill": "Python (computer programming)", "resume_quote": "Python (expert)", "job_quote": "experience with Python and PyTorch"}),
    # missing: in job, verbatim, absent from resume
    ("missing", {"skill": "Java", "job_quote": "Experience with Java is required"}),
    ("missing", {"skill": "Rust", "job_quote": "Nice to have: Rust, Kafka streaming"}),
    ("missing", {"skill": "Kafka", "job_quote": "Kafka streaming"}),
]

UNSUPPORTED = [
    # paraphrased resume quote (not verbatim)
    ("matched", {"skill": "Python", "resume_quote": "expert Python developer", "job_quote": "experience with Python and PyTorch"}),
    # invented job quote
    ("matched", {"skill": "FAISS", "resume_quote": "bi-encoder and FAISS", "job_quote": "experience with FAISS vector search"}),
    # verbatim quotes but the skill is not what the quotes are about and not in both ESCO sets
    ("matched", {"skill": "Spark", "resume_quote": "Python (expert)", "job_quote": "4+ years of experience with Python"}),
    # skill only in the resume (job never mentions it) — quote fabricated for the job
    ("matched", {"skill": "SageMaker", "resume_quote": "AWS (S3, SageMaker)", "job_quote": "AWS SageMaker experience"}),
    # empty quotes
    ("matched", {"skill": "Docker", "resume_quote": "", "job_quote": "Docker and Kubernetes"}),
    # too-short quote
    ("matched", {"skill": "SQL", "resume_quote": "S", "job_quote": "Strong SQL"}),
    # missing skill that the resume DOES have
    ("missing", {"skill": "Docker", "job_quote": "deploying models with Docker and Kubernetes"}),
    # missing skill with a quote not in the job
    ("missing", {"skill": "Java", "job_quote": "must know Java and Spring Boot"}),
    # missing skill not present in job at all (quote verbatim but unrelated)
    ("missing", {"skill": "Scala", "job_quote": "Nice to have: Rust, Kafka streaming"}),
    # missing skill present in resume via ESCO set ("Kubernetes" in RESUME_ESCO)
    ("missing", {"skill": "Kubernetes", "job_quote": "Docker and Kubernetes"}),
]


@pytest.mark.parametrize("kind,claim", SUPPORTED)
def test_supported(kind, claim):
    fn = check_matched if kind == "matched" else check_missing
    res = fn(claim, RESUME, JOB, RESUME_ESCO, JOB_ESCO)
    assert res.supported, res.reasons


@pytest.mark.parametrize("kind,claim", UNSUPPORTED)
def test_unsupported(kind, claim):
    fn = check_matched if kind == "matched" else check_missing
    res = fn(claim, RESUME, JOB, RESUME_ESCO, JOB_ESCO)
    assert not res.supported


def test_report_rates():
    expl = {"matched_skills": [c for k, c in SUPPORTED if k == "matched"] + [c for k, c in UNSUPPORTED if k == "matched"],
            "missing_skills": [c for k, c in SUPPORTED if k == "missing"] + [c for k, c in UNSUPPORTED if k == "missing"]}
    rep = check_explanation(expl, RESUME, JOB, RESUME_ESCO, JOB_ESCO)
    assert rep.n_claims == 20 and rep.n_supported == 10
    assert abs(rep.hallucination_rate - 0.5) < 1e-9
