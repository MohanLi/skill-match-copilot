from __future__ import annotations

from src.ui.service import run_search, setup_issues
from tests.test_agents_mock import FakeIndex


def test_semantic_baseline_never_loads_skill_models_or_profile(monkeypatch):
    from src.p3_agent import common, tools

    class BaselineToolBox:
        pool_ids = None

        def __init__(self, name):
            self.index = FakeIndex()

        def search_jobs(self, query, k):
            return self.index.search(query, k=k)

    def should_not_run(*args, **kwargs):
        raise AssertionError('The semantic baseline must not load skill models or call resume_profile.')

    monkeypatch.setattr(tools, 'ToolBox', BaselineToolBox)
    monkeypatch.setattr(common, 'resume_profile', should_not_run)
    result = run_search('Python and PyTorch experience. Email: me@example.com', 'remote', 'djinni_jobs',
                        'baseline', 3, model_loader=should_not_run)
    assert len(result['run']['ranking']) == 3
    assert result['profile'] is None and result['faithfulness'] is None
    assert result['run']['llm_calls'] == 0


def test_setup_distinguishes_base_models_and_trained_checkpoints(tmp_path, monkeypatch):
    cfg = {'index': {'text_embedding_model': 'models/base/encoder'}}
    encoder = tmp_path / 'models/base/encoder'
    encoder.mkdir(parents=True)
    (encoder / 'config.json').write_text('{}')
    monkeypatch.setattr('src.ui.service.load_env', lambda: None)
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    assert setup_issues('baseline', tmp_path, cfg) == []
    issues = setup_issues('single_verified', tmp_path, cfg)
    assert any('trained skill taggers' in message for message in issues)
    assert any('ANTHROPIC_API_KEY' in message for message in issues)
    for kind in ['skill', 'knowledge']:
        tagger = tmp_path / f'models/p1_tagger/{kind}_seed42'
        tagger.mkdir(parents=True)
        (tagger / 'config.json').write_text('{}')
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'test-only-key')
    assert setup_issues('single_verified', tmp_path, cfg) == []


PLACES = {"j1": {"locations": "Chicago, IL (Hybrid)", "remote": False}, "j2": {"locations": "Remote - US", "remote": True},
          "j3": {"locations": "Kyiv", "remote": False}, "j4": {"locations": "London", "remote": True}}


class FactsIndex(FakeIndex):
    """The fake index plus what the live index now stores: a name, job facts and all_metadata()."""
    name = "live_jobs"

    def __init__(self):
        super().__init__()
        for job_id, facts in PLACES.items():
            self.jobs[job_id].update(facts)

    def all_metadata(self):
        return [{k: v for k, v in job.items() if k != "text"} for job in self.jobs.values()]


def facts_toolbox():
    from src.p3_agent.tools import ToolBox

    class FactsToolBox(ToolBox):
        def __init__(self, name, extractor=None, normalizer=None):
            self.index, self.pool_ids = FactsIndex(), None
            self._extractor, self._normalizer = extractor, normalizer
            self.snippet_chars, self.job_text_chars = 300, 3500
            self.calls, self._pool_cache, self.allowed_ids, self._allowed = [], None, None, None

    return FactsToolBox


def test_toolbox_restriction_limits_search_and_get_job():
    tb = facts_toolbox()("live_jobs")
    tb.restrict_to(["j2", "j4"])
    assert {h["job_id"] for h in tb.search_jobs("python engineer", k=10)} == {"j2", "j4"}
    assert "error" in tb.get_job("j1")
    job = tb.get_job("j2")
    assert job["title"].startswith("Machine") and job["location"] == "Remote - US" and job["remote"] is True   # the agent sees where the job is
    tb.restrict_to(None)
    assert len(tb.search_jobs("python engineer", k=10)) == 4 and "error" not in tb.get_job("j1")


def test_requirements_filter_the_pool_before_ranking(monkeypatch):
    from src.p3_agent import tools
    from src.ui import service

    monkeypatch.setattr(tools, "ToolBox", facts_toolbox())
    service._FACTS.clear()
    result = run_search("Python, PyTorch and Docker engineer with Kubernetes experience in production.", "", "live_jobs", "baseline", 3,
                        requirements={"locations": ["Chicago"], "remote_ok": True})
    ranked = [e["job_id"] for e in result["run"]["ranking"]]
    assert set(ranked) == {"j1", "j2", "j4"}                                   # Kyiv is neither Chicago nor remote
    assert all(c["ok"] for job_id in ranked for c in result["checks"][job_id])
    assert result["pool"]["meets_all"] == 3 and result["pool"]["restricted"] and "ids" not in result["pool"]
    assert result["requirements"]["locations"] == ["Chicago"] and result["requirements"]["remote_ok"] is True


def test_jobs_that_miss_a_requirement_are_listed_last_and_flagged(monkeypatch):
    from src.p3_agent import tools
    from src.p3_agent.requirements import n_failed
    from src.ui import service

    monkeypatch.setattr(tools, "ToolBox", facts_toolbox())
    service._FACTS.clear()
    result = run_search("Python, PyTorch and Docker engineer with Kubernetes experience in production.", "", "live_jobs", "baseline", 4,
                        requirements={"roles": ["Machine learning engineer"], "locations": ["Chicago"]})
    assert result["pool"]["meets_all"] == 0 and result["pool"]["relaxed"] == ["role", "location", "role and location"]
    ranked = [e["job_id"] for e in result["run"]["ranking"]]
    failures = [n_failed(result["checks"][job_id]) for job_id in ranked]
    assert failures == sorted(failures) and failures[0] == 1 and failures[-1] == 2
    assert [e["rank"] for e in result["run"]["ranking"]] == [1, 2, 3, 4]
    assert {"j1", "j2"} == set(ranked[:2])                                     # Chicago with another role, the ML engineer elsewhere


def test_repeat_postings_of_one_job_share_a_card():
    from src.p3_agent.common import merge_repeats

    jobs = {"a": {"title": "Senior ML Platform Engineer", "company": "Preply", "locations": "Kyiv", "url": "https://x/a"},
            "b": {"title": "Data Scientist", "company": "Preply", "locations": "London"},
            "c": {"title": "senior ml platform engineer ", "company": "preply", "locations": "London", "url": "https://x/c"},
            "d": {"title": "Senior ML Platform Engineer", "company": "Preply", "locations": "Lisbon"}}
    vectors = {"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [0.999, 0.04], "d": [0.8, 0.6]}          # d: same title, a different text
    ranking = [{"job_id": i, "rank": n + 1} for n, i in enumerate("abcd")]
    merged = merge_repeats(ranking, jobs, vectors)
    assert [e["job_id"] for e in merged] == ["a", "b", "d"] and [e["rank"] for e in merged] == [1, 2, 3]
    assert merged[0]["also"] == [{"job_id": "c", "locations": "London", "url": "https://x/c"}]
    assert "also" not in merged[1] and "also" not in merged[2]
    assert merge_repeats([{"job_id": "a", "rank": 1}, {"job_id": "c", "rank": 2}], jobs, {}) == [{"job_id": "a", "rank": 1}, {"job_id": "c", "rank": 2}]   # no vectors, no merging


def test_the_few_full_matches_are_never_crowded_out():
    from src.p3_agent.common import AgentRun
    from src.p3_agent.match import match_jobs
    from src.p3_agent.requirements import Requirements

    seen = {}

    class Stub:                                   # a ranker that returned two closer-sounding jobs and left the QA job out
        def run(self, resume, preferences):
            seen["preferences"] = preferences
            return AgentRun(approach="stub", ranking=[{"job_id": "j1", "rank": 1, "fit_score": 70}, {"job_id": "j2", "rank": 2, "fit_score": 60}])

    tb = facts_toolbox()("live_jobs")
    req = Requirements(roles=["QA engineer"])
    run, jobs, checks, pool = match_jobs(Stub(), tb, "Python, PyTorch and Docker engineer.", req, top_n=2, baseline=True)
    assert pool["must_include"] == ["j3"] and pool["relaxed"] == ["role"]
    assert "must come first in the ranking: j3" in seen["preferences"]            # agents are told which postings fully qualify
    assert [e["job_id"] for e in run.ranking] == ["j3", "j1"] and [e["rank"] for e in run.ranking] == [1, 2]
    assert checks["j3"][0]["ok"] is True and checks["j1"][0]["ok"] is False and set(jobs) == {"j1", "j2", "j3"}
    assert tb.allowed_ids is None                                                   # the temporary restriction is undone
