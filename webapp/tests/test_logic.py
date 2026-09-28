"""Unit tests for the release board logic."""
from datetime import datetime

import pytest

from webapp.logic import (classify, is_newer, parse_version, sort_releases,
                          summarise)


def test_parse_version_handles_v_prefix():
    assert parse_version("v2.4.1") == (2, 4, 1)
    assert parse_version("2.4.1") == (2, 4, 1)


def test_parse_version_rejects_bad_input():
    with pytest.raises(ValueError):
        parse_version("2.4")
    with pytest.raises(ValueError):
        parse_version("v2.4.x")


def test_is_newer_compares_numerically():
    assert is_newer("v2.4.1", "v2.4.0") is True
    assert is_newer("v2.10.0", "v2.9.9") is True     # not string comparison
    assert is_newer("v2.4.0", "v2.4.1") is False


def test_sort_releases_puts_newest_first():
    releases = [
        {"version": "old", "deployed_at": datetime(2026, 1, 1)},
        {"version": "new", "deployed_at": datetime(2026, 6, 1)},
    ]
    assert sort_releases(releases)[0]["version"] == "new"


def test_classify_uses_the_error_budget():
    assert classify(0.0) == "healthy"
    assert classify(0.4) == "watch"
    assert classify(1.0) == "degraded"
    assert classify(5.2) == "degraded"


def test_classify_rejects_negative_rates():
    with pytest.raises(ValueError):
        classify(-1.0)


def test_summarise_reports_the_live_release():
    releases = [
        {"version": "v1.0.0", "deployed_at": datetime(2026, 1, 1), "error_rate": 0.0},
        {"version": "v1.1.0", "deployed_at": datetime(2026, 2, 1), "error_rate": 0.0},
    ]
    summary = summarise(releases)
    assert summary["live"]["version"] == "v1.1.0"
    assert summary["total"] == 2
    assert summary["healthy_streak"] == 2


def test_summarise_handles_an_empty_board():
    assert summarise([])["live"] is None
