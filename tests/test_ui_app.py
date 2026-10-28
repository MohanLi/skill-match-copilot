from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.ui import service

APP = Path(__file__).resolve().parents[1] / 'src/ui/app.py'


def button(app, label):
    return next(item for item in app.button if item.label == label)


@pytest.fixture
def ui(monkeypatch):
    st.cache_data.clear()
    monkeypatch.setattr(service, 'index_status', lambda name: (4, None))
    monkeypatch.setattr(service, 'setup_issues', lambda mode: [])
    calls = []

    def search(resume, preferences, index, mode, top_n, model, model_loader=None):
        calls.append((resume, preferences, index, mode, top_n, model))
        return {'mode': mode, 'index': index, 'model': model, 'top_n': top_n,
                'preferences': preferences, 'profile': None, 'faithfulness': None, 'elapsed': .1,
                'run': {'ranking': [{'job_id': 'j1', 'rank': 1, 'fit_score': 70, 'matched_skills': [], 'missing_skills': []}],
                        'llm_calls': 0, 'cost_usd': 0, 'tool_calls': 1, 'trace': [], 'error': None},
                'jobs': {'j1': {'title': 'Python Engineer', 'company': 'Example', 'text': 'Python and Docker'}}}

    monkeypatch.setattr(service, 'run_search', search)
    app = AppTest.from_file(str(APP), default_timeout=10).run()
    assert not app.exception
    app.radio[0].set_value('baseline').run()
    button(app, 'Find matching jobs').click().run()
    assert not app.exception
    return app, calls


def test_results_survive_widget_reruns_and_show_original_search(ui):
    app, calls = ui
    before = deepcopy(app.session_state['search_result'])
    app.slider[0].set_value(8).run()
    assert not app.exception
    assert app.session_state['search_result'] == before
    assert len(calls) == 1
    assert any('previous search' in message.value for message in app.info)
    assert any('Python Engineer' in title.value for title in app.subheader)
    assert any(item.label == 'Similarity' for item in app.metric)


def test_failed_new_search_keeps_previous_results(ui, monkeypatch):
    app, calls = ui
    before = deepcopy(app.session_state['search_result'])

    def unavailable(*args, **kwargs):
        raise RuntimeError('provider unavailable')

    monkeypatch.setattr(service, 'run_search', unavailable)
    button(app, 'Find matching jobs').click().run()
    assert not app.exception
    assert app.error and 'previous results' in app.error[0].value
    assert app.session_state['search_result'] == before


def test_clear_results_without_rerunning_the_models(ui):
    app, calls = ui
    button(app, 'Clear results').click().run()
    assert not app.exception
    assert 'search_result' not in app.session_state
    assert len(calls) == 1


def test_missing_local_assets_show_setup_instead_of_crashing(monkeypatch):
    st.cache_data.clear()
    monkeypatch.setattr(service, 'index_status', lambda name: (0, 'Restore the job index.'))
    monkeypatch.setattr(service, 'setup_issues', lambda mode: ['Restore the trained checkpoints.'])
    app = AppTest.from_file(str(APP), default_timeout=10).run()
    assert not app.exception
    assert button(app, 'Find matching jobs').disabled
    assert any('local files' in item.value for item in app.warning)
