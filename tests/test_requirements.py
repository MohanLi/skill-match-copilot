"""Role / location / remote requirements: plain rules on job facts, no model involved."""
from __future__ import annotations

import pytest

from src.p3_agent.requirements import Requirements, check_job, check_location, match_role, order_by_requirements, pool_summary, select_pool

JOBS = [
    {"job_id": "chi-ml", "title": "Machine Learning Engineer", "locations": "Chicago, IL (Hybrid)", "remote": False},
    {"job_id": "us-ai", "title": "Senior AI/ML Engineer", "locations": "Remote - US", "remote": True},
    {"job_id": "kr-ai", "title": "Applied AI Engineer", "locations": "Seoul, Korea", "remote": True},
    {"job_id": "ny-ds", "title": "Data Scientist", "locations": "New York, NY", "remote": False},
    {"job_id": "multi-de", "title": "Data Engineer", "locations": "San Francisco | Toronto | Chicago"},
    {"job_id": "austin-ml", "title": "ML Engineer, Ranking", "locations": "Austin", "remote": True},
]


def test_role_matches_when_the_title_contains_every_word_of_it():
    assert match_role("Senior AI/ML Engineer", ["AI engineer"]) == "AI engineer"
    assert match_role("Machine Learning Engineer, Ranking", ["ML engineer"]) == "ML engineer"
    assert match_role("Generative AI Engineer", ["AI engineer"]) and match_role("GenAI Engineers", ["AI engineer"])
    assert match_role("Staff Machine Learning Engineer", ["AI engineer"]) is None                # nothing is inferred: ML is not AI
    assert match_role("Staff Machine Learning Engineer", ["AI engineer", "machine learning engineer"]) == "machine learning engineer"
    assert match_role("Engineering Manager, Machine Learning", ["machine learning engineer"]) is None
    assert match_role("AI Engineer", ["AI Engineer roles"]) and match_role("Data Engineer", ["AI Engineer roles"]) is None
    assert match_role("AI Engineer", []) is None and match_role("AI Engineer", ["roles"]) is None  # a filler-only role matches nothing


@pytest.mark.parametrize("locations,remote,ok,label", [
    ("Chicago, IL (Hybrid)", None, True, "Chicago"),
    ("San Francisco | Toronto | Chicago", None, True, "Chicago"),
    ("Remote - US", True, True, "Remote · United States"),
    ("United States - Remote", None, True, "Remote · United States"),       # 'remote' in the text counts without the flag
    ("Austin", True, False, "Remote, but not stated as United States"),     # 'US' is matched as a word, not inside 'Austin'
    ("Seoul, Korea", True, False, "Remote, but not stated as United States"),
    ("New York, NY", False, False, "Not Chicago, or remote within United States"),
])
def test_location_is_a_listed_place_or_a_remote_job_in_the_stated_region(locations, remote, ok, label):
    req = Requirements(locations=["Chicago"], remote_ok=True, remote_within=["United States"])
    meta = {"locations": locations, **({} if remote is None else {"remote": remote})}
    assert check_location(meta, req) == {"name": "location", "ok": ok, "label": label}


def test_remote_anywhere_and_missing_location_data():
    anywhere = Requirements(remote_ok=True)
    assert check_location({"locations": "Seoul, Korea", "remote": True}, anywhere)["ok"] is True
    assert check_location({"locations": "Berlin"}, anywhere)["ok"] is False
    assert check_location({"title": "no facts stored"}, anywhere) == {"name": "location", "ok": None, "label": "Location not listed"}
    assert check_location({"locations": "Berlin"}, Requirements(roles=["AI engineer"])) is None     # nothing asked, nothing checked
    assert check_job({"title": "AI Engineer", "locations": "Berlin"}, Requirements()) == []


def test_pool_holds_only_jobs_that_meet_every_requirement():
    req = Requirements(roles=["AI engineer", "machine learning engineer"], locations=["Chicago"], remote_ok=True, remote_within=["United States"])
    pool = select_pool(JOBS, req, need=2)
    assert pool["ids"] == ["chi-ml", "us-ai"] and pool["relaxed"] == [] and pool["restricted"]
    assert (pool["total"], pool["meets_all"], pool["meets_location"], pool["meets_role"]) == (6, 2, 3, 4)
    assert "ids" not in pool_summary(pool) and pool_summary(None) is None
    assert select_pool(JOBS, Requirements(), need=5)["restricted"] is False                         # nothing asked: the whole index


def test_pool_is_topped_up_when_too_few_qualify_and_full_matches_stay_first():
    req = Requirements(roles=["AI engineer"], locations=["Chicago"])
    pool = select_pool(JOBS, req, need=3)                 # nobody is an AI engineer in Chicago
    assert pool["meets_all"] == 0 and pool["relaxed"] == ["role", "location"]
    assert pool["ids"][:2] == ["chi-ml", "multi-de"]     # in Chicago with another role first, then AI engineers elsewhere
    assert set(pool["ids"][2:]) == {"us-ai", "kr-ai"}
    checks = {m["job_id"]: check_job(m, req) for m in JOBS}
    ranking = [{"job_id": "ny-ds", "rank": 1}, {"job_id": "us-ai", "rank": 2}, {"job_id": "chi-ml", "rank": 3}]
    ordered = order_by_requirements(ranking, checks)
    assert [e["job_id"] for e in ordered] == ["us-ai", "chi-ml", "ny-ds"] and [e["rank"] for e in ordered] == [1, 2, 3]


def test_an_index_without_location_facts_applies_only_the_role():
    jobs = [{"job_id": "a", "title": "AI Engineer"}, {"job_id": "b", "title": "QA Engineer"}]
    pool = select_pool(jobs, Requirements(roles=["AI engineer"], locations=["Chicago"]), need=1)
    assert pool["ids"] == ["a"] and pool["location_data"] is False and pool["meets_location"] is None


def test_requirements_are_described_for_the_reader_and_for_the_agent():
    req = Requirements.from_any({"roles": ["AI engineer", " ML engineer "], "locations": ["Chicago"], "remote_ok": True,
                                 "remote_within": "United States, Canada", "notes": "no on-call"})
    assert req.remote_within == ["United States", "Canada"] and req.roles == ["AI engineer", "ML engineer"]
    assert req.describe() == "AI engineer or ML engineer · Chicago, or remote within United States / Canada"
    prompt = req.to_prompt({"restricted": True, "relaxed": []})
    assert prompt.startswith("Target role: AI engineer or ML engineer. Location: Chicago, or remote within United States / Canada. no on-call.")
    assert "already returns only postings that meet" in prompt
    relaxed = req.to_prompt({"restricted": False, "relaxed": ["role"], "must_include": ["live-1", "live-2"]})
    assert "prefer postings that meet all of them" in relaxed and relaxed.endswith("must come first in the ranking: live-1, live-2.")
    assert Requirements.from_any(None).describe() == "" and not Requirements.from_any({"notes": "x"}).filters
