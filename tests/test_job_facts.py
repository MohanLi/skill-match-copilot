"""Job facts taken from the live feed into the index metadata (no index, no models needed)."""
from __future__ import annotations

import numpy as np

from src.index.job_facts import company_name, employment_type, facts_from_row, remote_flag, salary_period


def test_company_comes_from_the_job_board_name_when_the_feed_has_no_company():
    assert company_name("Acme Corp", "acme") == "Acme Corp"
    assert company_name(None, "epidemic-sound") == "Epidemic Sound"
    assert company_name(float("nan"), "comcast/wd115/comcast_careers") == "Comcast"
    assert company_name("", "Grindr LLC") == "Grindr LLC"                 # already a display name: left alone
    assert company_name(None, None) == ""


def test_feed_spellings_are_normalised():
    assert {employment_type(v) for v in ("FullTime", "Full-time", "full_time", "Salaried, full-time")} == {"Full time"}
    assert employment_type("Part time") == "Part time" and employment_type("Variable") == "" and employment_type(None) == ""
    assert salary_period("PER-YEAR-SALARY") == "year" and salary_period("PER-MONTH-SALARY") == "month" and salary_period(None) == ""
    assert remote_flag(True) is True and remote_flag(np.bool_(False)) is False and remote_flag("True") is True
    assert remote_flag(None) is None and remote_flag(float("nan")) is None


def test_unknown_facts_are_left_out_not_guessed():
    full = facts_from_row({"board": "openai", "location_list": np.array(["San Francisco", "Remote - US", "San Francisco"]), "remote": True,
                           "employment_type": "FullTime", "salary_min": 160000.0, "salary_max": 210000.0, "salary_currency": "usd", "salary_period": None})
    assert full == {"company": "Openai", "locations": "San Francisco | Remote - US", "remote": True, "employment_type": "Full time",
                    "salary_min": 160000.0, "salary_max": 210000.0, "salary_currency": "USD"}
    sparse = facts_from_row({"board": None, "location_list": [], "remote": None, "salary_min": float("nan"), "salary_max": None})
    assert sparse == {}
