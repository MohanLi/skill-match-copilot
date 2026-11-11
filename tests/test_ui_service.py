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
