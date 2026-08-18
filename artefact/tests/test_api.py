"""
Test suite for the CI/CD failure prediction service.

Run:  pytest tests/ -v

The most important tests here are the history ones: they verify that recorded
outcomes actually change the risk score. If those fail, the history store is
disconnected and the model has silently lost ~86% of its predictive power
while still returning plausible-looking numbers.
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("DB_PATH", "data/test_suite.db")

from fastapi.testclient import TestClient  # noqa: E402

from app.features import FEATURE_ORDER, build_feature_vector, to_row  # noqa: E402
from app.main import app, store  # noqa: E402

BASE_PAYLOAD = {
    "project": "test/repo",
    "git_diff_src_churn": 150,
    "git_diff_test_churn": 20,
    "gh_diff_files_modified": 5,
    "gh_diff_src_files": 4,
    "gh_sloc": 20000,
    "gh_test_lines_per_kloc": 150,
    "gh_team_size": 5,
    "is_pr": False,
    "branch": "main",
    "language": "ruby",
}


@pytest.fixture
def client():
    store.reset()
    with TestClient(app) as c:
        yield c
    store.reset()


# ------------------------------------------------------------------ health
def test_health_reports_model_loaded(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True
    assert r.json()["n_features"] == 46


def test_model_info_exposes_threshold(client):
    body = client.get("/model-info").json()
    assert 0 < body["threshold"] <= 1
    assert body["n_features"] == len(FEATURE_ORDER)


# ------------------------------------------------------------- predictions
def test_predict_returns_valid_probability(client):
    body = client.post("/predict", json=BASE_PAYLOAD).json()
    assert 0.0 <= body["risk_score"] <= 1.0
    assert body["decision"] in {"LOW_RISK", "HIGH_RISK"}
    assert body["latency_ms"] > 0


def test_cold_start_flagged(client):
    body = client.post("/predict", json=BASE_PAYLOAD).json()
    assert body["history_available"] is False
    assert body["builds_in_history"] == 0


def test_decision_matches_threshold(client):
    body = client.post("/predict", json=BASE_PAYLOAD).json()
    expected = "HIGH_RISK" if body["risk_score"] >= body["threshold"] else "LOW_RISK"
    assert body["decision"] == expected


# ---------------------------------------------------------------- history
def test_failing_history_raises_risk(client):
    """The core mechanism. If this fails, history is not reaching the model."""
    for _ in range(5):
        client.post("/outcome", json={"project": "test/repo", "failed": False})
    healthy = client.post("/predict", json=BASE_PAYLOAD).json()["risk_score"]

    for _ in range(5):
        client.post("/outcome", json={"project": "test/repo", "failed": True})
    broken = client.post("/predict", json=BASE_PAYLOAD).json()["risk_score"]

    assert broken > healthy, (
        f"Failing history did not raise risk ({healthy:.3f} -> {broken:.3f}). "
        "The history store is likely disconnected from feature assembly."
    )


def test_outcome_increments_history(client):
    client.post("/outcome", json={"project": "test/repo", "failed": True})
    body = client.post("/outcome", json={"project": "test/repo", "failed": False}).json()
    assert body["builds_in_history"] == 2
    assert body["recorded"] is True


def test_history_is_per_project(client):
    for _ in range(5):
        client.post("/outcome", json={"project": "project/a", "failed": True})

    a = client.post("/predict", json={**BASE_PAYLOAD, "project": "project/a"}).json()
    b = client.post("/predict", json={**BASE_PAYLOAD, "project": "project/b"}).json()

    assert a["builds_in_history"] == 5
    assert b["builds_in_history"] == 0
    assert a["risk_score"] != b["risk_score"]


# ---------------------------------------------------------------- policies
def test_shadow_and_warn_never_block(client, monkeypatch):
    """Only 'block' policy may set should_block. Shadow mode must be inert."""
    import app.main as main
    from app.schemas import PolicyMode

    for _ in range(10):
        client.post("/outcome", json={"project": "test/repo", "failed": True})

    for policy in (PolicyMode.SHADOW, PolicyMode.WARN):
        monkeypatch.setattr(main, "POLICY", policy)
        body = client.post("/predict", json=BASE_PAYLOAD).json()
        assert body["should_block"] is False, f"{policy} must never block"


# -------------------------------------------------------------- validation
def test_rejects_negative_churn(client):
    bad = {**BASE_PAYLOAD, "git_diff_src_churn": -10}
    assert client.post("/predict", json=bad).status_code == 422


def test_rejects_missing_project(client):
    assert client.post("/predict", json={"git_diff_src_churn": 10}).status_code == 422


# ---------------------------------------------------------- feature module
def test_feature_vector_length_and_order():
    features = build_feature_vector(BASE_PAYLOAD, {
        "prev_build_failed": None, "failure_rate_last_5": 0.0,
        "failure_rate_last_20": 0.0, "project_cum_failure_rate": 0.0,
        "builds_so_far_in_project": 0, "consecutive_prior_failures": 0,
        "hours_since_last_build": None, "builds_in_last_24h": 0,
    })
    row = to_row(features)
    assert len(row) == len(FEATURE_ORDER) == 46
    assert all(isinstance(v, float) for v in row)


def test_missing_indicators_set_on_cold_start():
    features = build_feature_vector(BASE_PAYLOAD, {
        "prev_build_failed": None, "failure_rate_last_5": 0.0,
        "failure_rate_last_20": 0.0, "project_cum_failure_rate": 0.0,
        "builds_so_far_in_project": 0, "consecutive_prior_failures": 0,
        "hours_since_last_build": None, "builds_in_last_24h": 0,
    })
    assert features["prev_build_failed_was_missing"] == 1.0
    assert features["hours_since_last_build_was_missing"] == 1.0
    assert features["is_first_build_in_project"] == 1.0


def test_derived_ratios_match_notebook():
    """These formulas must match Notebook 01 exactly — training/serving skew check."""
    features = build_feature_vector(
        {**BASE_PAYLOAD, "git_diff_src_churn": 100, "git_diff_test_churn": 50,
         "gh_diff_files_modified": 4, "gh_sloc": 1000},
        {"builds_so_far_in_project": 0})
    assert features["total_churn"] == 150
    assert features["total_files_changed"] == 4
    assert features["test_to_src_churn_ratio"] == pytest.approx(50 / 101)
    assert features["churn_per_file"] == pytest.approx(150 / 5)
    assert features["churn_relative_to_sloc"] == pytest.approx(150 / 1001)
