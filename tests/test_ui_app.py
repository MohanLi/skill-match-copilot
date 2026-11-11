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
    monkeypatch.setattr(service, 'requirement_options', lambda name: {'locations': ['Chicago', 'London'], 'location_data': True, 'jobs': 4})
    calls = []

    def search(resume, preferences, index, mode, top_n, model, model_loader=None, requirements=None):
        calls.append((resume, preferences, index, mode, top_n, model, requirements))
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
    assert any(item.label == 'Error details' for item in app.expander)
    assert any('provider unavailable' in item.value for item in app.code)
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
    monkeypatch.setattr(service, 'requirement_options', lambda name: {'locations': [], 'location_data': True, 'jobs': 0})
    app = AppTest.from_file(str(APP), default_timeout=10).run()
    assert not app.exception
    assert button(app, 'Find matching jobs').disabled
    assert any('local files' in item.value for item in app.warning)


def test_form_sends_structured_requirements(ui):
    app, calls = ui
    requirements = calls[0][6]
    assert requirements == {'roles': ['Machine learning engineer'], 'locations': ['Chicago'], 'remote_ok': True, 'remote_within': ['United States']}
    assert calls[0][1] == ''                                   # the free-text note is for agents only
    next(item for item in app.text_input if item.label == 'Remote within (optional)').set_value('United States, Canada')
    next(item for item in app.multiselect if item.label == 'Role, any of').set_value(['AI engineer', 'Machine learning engineer'])
    button(app, 'Find matching jobs').click().run()
    assert not app.exception
    assert calls[1][6]['remote_within'] == ['United States', 'Canada'] and calls[1][6]['roles'] == ['AI engineer', 'Machine learning engineer']


def test_cards_show_job_facts_requirement_checks_and_can_be_saved(monkeypatch):
    st.cache_data.clear()
    monkeypatch.setattr(service, 'index_status', lambda name: (4, None))
    monkeypatch.setattr(service, 'setup_issues', lambda mode: [])
    monkeypatch.setattr(service, 'requirement_options', lambda name: {'locations': ['Chicago'], 'location_data': True, 'jobs': 4})
    monkeypatch.setattr(service, 'similar_jobs', lambda index, job_id, requirements=None, exclude=(), k=5: [
        {'job_id': 'j9', 'title': 'Applied AI Engineer', 'company': 'Other', 'locations': 'Remote - US', 'score': 0.71}])

    def search(resume, preferences, index, mode, top_n, model, model_loader=None, requirements=None):
        claim = {'skill': 'Python', 'resume_quote': 'Python (expert)', 'job_quote': 'Strong Python'}
        return {'mode': mode, 'index': index, 'model': model, 'top_n': top_n, 'preferences': preferences, 'elapsed': .1,
                'profile': {'skills': ['taken'], 'knowledge': ['SQL', 'PyTorch'], 'esco_skills': ['SQL'], 'not_in_esco': ['PyTorch', 'taken'],
                            'mapped': [{'phrase': 'SQL', 'esco_label': 'SQL', 'score': 1.0}, {'phrase': 'PyTorch', 'esco_label': 'NOT_IN_ESCO', 'score': .4},
                                       {'phrase': 'taken', 'esco_label': 'NOT_IN_ESCO', 'score': .5}]},
                'faithfulness': {'hallucination_rate': 0.0, 'n_claims': 3},
                'requirements': {**requirements, 'notes': preferences},
                'pool': {'total': 2000, 'meets_all': 1, 'meets_location': 580, 'meets_role': 587, 'relaxed': ['role'], 'location_data': True, 'restricted': True},
                'checks': {'j1': [{'name': 'role', 'ok': True, 'label': 'Role: Machine learning engineer'}, {'name': 'location', 'ok': True, 'label': 'Remote'}],
                           'j2': [{'name': 'role', 'ok': False, 'label': 'Different role'}, {'name': 'location', 'ok': True, 'label': 'Chicago'}]},
                'run': {'ranking': [{'job_id': 'j1', 'rank': 1, 'fit_score': 82, 'matched_skills': [claim], 'missing_skills': [{'skill': 'Kubernetes', 'job_quote': 'Kubernetes'}], 'summary': 'Close fit.'},
                                    {'job_id': 'j2', 'rank': 2, 'fit_score': 55, 'matched_skills': [claim], 'missing_skills': [{'skill': 'kubernetes', 'job_quote': 'K8s'}], 'summary': 'Other role.'}],
                        'llm_calls': 5, 'cost_usd': .01, 'tool_calls': 9, 'trace': [], 'error': None},
                'jobs': {'j1': {'title': 'ML Engineer', 'company': 'Example', 'locations': 'Remote - US', 'remote': True, 'published': '2026-10-09T22:19:14.701+00:00',
                                'salary_min': 160000.0, 'salary_max': 210000.0, 'salary_currency': 'USD', 'url': 'https://example.com/j1', 'text': 'Strong Python and Kubernetes'},
                         'j2': {'title': 'Data Analyst', 'company': 'Other', 'locations': 'Chicago, IL', 'text': 'Strong Python, K8s'}}}

    monkeypatch.setattr(service, 'run_search', search)
    app = AppTest.from_file(str(APP), default_timeout=10).run()
    button(app, 'Find matching jobs').click().run()
    assert not app.exception
    markdown = ' '.join(item.value for item in app.markdown)
    captions = ' '.join(item.value for item in app.caption)
    assert ':green-badge[:material/check: Role: Machine learning engineer]' in markdown and ':red-badge[:material/close: Different role]' in markdown
    assert ':orange-badge[Kubernetes · 2 of 2]' in markdown and ':green-badge[Python · 2 of 2]' in markdown     # the across-jobs summary
    assert 'Looking for:' in markdown and '580 of 2,000 jobs meet the location requirement' in captions
    assert any('Only 1 job meets every requirement' in item.value for item in app.warning)
    assert ':material/location_on: Remote - US' in captions and '\\$160,000–\\$210,000' in captions and 'Posted Oct 9' in captions
    assert ':blue-badge[PyTorch]' in markdown and '| SQL | SQL | 1.00 · strong |' in markdown                    # the My skills tab
    next(item for item in app.button if item.key == 'save-j1').click().run()
    assert not app.exception
    assert list(app.session_state['saved_jobs']) == ['j1'] and app.session_state['saved_jobs']['j1']['score'] == 82
    assert any(item.label == 'Download saved jobs (CSV)' for item in app.get('download_button'))
    next(item for item in app.button if item.key == 'more-j2').click().run()
    assert not app.exception
    assert any('Applied AI Engineer' in item.value for item in app.markdown)
    next(item for item in app.button if item.key == 'unsave-j1').click().run()
    assert not app.exception and app.session_state['saved_jobs'] == {}


def test_an_index_without_job_facts_says_how_to_add_them(monkeypatch):
    st.cache_data.clear()
    monkeypatch.setattr(service, 'index_status', lambda name: (4, None))
    monkeypatch.setattr(service, 'setup_issues', lambda mode: [])
    monkeypatch.setattr(service, 'requirement_options', lambda name: {'locations': [], 'location_data': False, 'jobs': 4})
    calls = []
    monkeypatch.setattr(service, 'run_search', lambda *args, **kwargs: calls.append(kwargs) or (_ for _ in ()).throw(RuntimeError('stop here')))
    app = AppTest.from_file(str(APP), default_timeout=10).run()
    assert not app.exception
    assert any('python -m src.index.job_facts' in item.value for item in app.info)
    assert next(item for item in app.multiselect if item.label == 'Location, any of').disabled
    assert not button(app, 'Find matching jobs').disabled                      # searching still works, by role only
    button(app, 'Find matching jobs').click().run()
    assert calls[0]['requirements'] == {'roles': ['Machine learning engineer'], 'locations': [], 'remote_ok': False, 'remote_within': []}
