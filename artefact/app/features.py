"""
Feature assembly — the SINGLE SOURCE OF TRUTH for how raw build data becomes
a model input vector.

CRITICAL: this module must be used by BOTH training and serving. If the API
computes a feature even slightly differently from the notebook, predictions
degrade silently — no error is raised, the numbers just get worse. This is
called training/serving skew and it is one of the most common production ML
bugs.

The 46 features are grouped exactly as in Notebook 01:
  G1  code change (raw counts from the diff)
  G2  derived ratios
  G3  project health snapshot
  G4  pipeline history   <-- requires the history store, ~86% of model gain
  G5  temporal / context
  G6  missing-value indicators
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

# Order matters. Must match models/feature_names.json exactly.
FEATURE_ORDER: list[str] = [
    # G1 — code change
    "git_diff_src_churn", "git_diff_test_churn",
    "gh_diff_files_added", "gh_diff_files_deleted", "gh_diff_files_modified",
    "gh_diff_tests_added", "gh_diff_tests_deleted",
    "gh_diff_src_files", "gh_diff_doc_files", "gh_diff_other_files",
    "gh_num_commits_on_files_touched", "git_num_all_built_commits",
    "gh_num_commits_in_push",
    # G2 — derived ratios
    "total_churn", "total_files_changed", "test_to_src_churn_ratio",
    "churn_per_file", "churn_relative_to_sloc",
    "is_test_only_change", "is_doc_only_change", "tests_touched",
    # G3 — project health
    "gh_sloc", "gh_test_lines_per_kloc", "gh_test_cases_per_kloc",
    "gh_asserts_cases_per_kloc", "gh_team_size",
    # G4 — pipeline history
    "prev_build_failed", "failure_rate_last_5", "failure_rate_last_20",
    "project_cum_failure_rate", "builds_so_far_in_project",
    "consecutive_prior_failures", "hours_since_last_build", "builds_in_last_24h",
    # G5 — temporal / context
    "gh_is_pr", "gh_by_core_team_member", "hour_of_day", "day_of_week",
    "is_weekend", "month", "is_default_branch", "lang_java",
    "is_first_build_in_project",
    # G6 — missing indicators
    "gh_num_commits_in_push_was_missing", "prev_build_failed_was_missing",
    "hours_since_last_build_was_missing",
]

DEFAULT_BRANCHES = {"master", "main", "develop", "development"}

# Median hours-since-last-build from the TRAINING set. Used when a project has
# no history. Update this if you retrain on different data.
MEDIAN_HOURS_SINCE_LAST_BUILD = 3.0


def _num(value: Any, default: float = 0.0) -> float:
    """Coerce to float, treating None/blank/non-numeric as the default."""
    if value is None:
        return default
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return default if f != f else f  # NaN check


def build_feature_vector(payload: dict, history: dict) -> dict[str, float]:
    """
    Assemble the full feature dict from a prediction request plus the project's
    history summary.

    payload : the raw request fields (code diff stats, project health, context)
    history : output of HistoryStore.summarise() for this project

    Returns a dict keyed by feature name. Use `to_row()` to get the ordered list.
    """
    f: dict[str, float] = {}

    # ---- G1: code change ---------------------------------------------------
    src_churn = _num(payload.get("git_diff_src_churn"))
    test_churn = _num(payload.get("git_diff_test_churn"))
    files_added = _num(payload.get("gh_diff_files_added"))
    files_deleted = _num(payload.get("gh_diff_files_deleted"))
    files_modified = _num(payload.get("gh_diff_files_modified"))
    tests_added = _num(payload.get("gh_diff_tests_added"))
    tests_deleted = _num(payload.get("gh_diff_tests_deleted"))
    src_files = _num(payload.get("gh_diff_src_files"))
    doc_files = _num(payload.get("gh_diff_doc_files"))
    other_files = _num(payload.get("gh_diff_other_files"))

    f["git_diff_src_churn"] = src_churn
    f["git_diff_test_churn"] = test_churn
    f["gh_diff_files_added"] = files_added
    f["gh_diff_files_deleted"] = files_deleted
    f["gh_diff_files_modified"] = files_modified
    f["gh_diff_tests_added"] = tests_added
    f["gh_diff_tests_deleted"] = tests_deleted
    f["gh_diff_src_files"] = src_files
    f["gh_diff_doc_files"] = doc_files
    f["gh_diff_other_files"] = other_files
    f["gh_num_commits_on_files_touched"] = _num(
        payload.get("gh_num_commits_on_files_touched"))
    f["git_num_all_built_commits"] = _num(
        payload.get("git_num_all_built_commits"), 1.0)

    # commits_in_push is NA for PR builds in TravisTorrent -> impute + flag
    raw_push = payload.get("gh_num_commits_in_push")
    push_missing = raw_push is None
    f["gh_num_commits_in_push"] = (
        f["git_num_all_built_commits"] if push_missing else _num(raw_push))
    f["gh_num_commits_in_push_was_missing"] = 1.0 if push_missing else 0.0

    # ---- G2: derived ratios ------------------------------------------------
    sloc = _num(payload.get("gh_sloc"))
    total_churn = src_churn + test_churn
    total_files = files_added + files_deleted + files_modified

    f["total_churn"] = total_churn
    f["total_files_changed"] = total_files
    f["test_to_src_churn_ratio"] = test_churn / (src_churn + 1.0)
    f["churn_per_file"] = total_churn / (total_files + 1.0)
    f["churn_relative_to_sloc"] = total_churn / (sloc + 1.0)
    f["is_test_only_change"] = 1.0 if (src_files == 0 and test_churn > 0) else 0.0
    f["is_doc_only_change"] = (
        1.0 if (src_files == 0 and doc_files > 0 and test_churn == 0) else 0.0)
    f["tests_touched"] = 1.0 if (tests_added + tests_deleted) > 0 else 0.0

    # ---- G3: project health ------------------------------------------------
    f["gh_sloc"] = sloc
    f["gh_test_lines_per_kloc"] = _num(payload.get("gh_test_lines_per_kloc"))
    f["gh_test_cases_per_kloc"] = _num(payload.get("gh_test_cases_per_kloc"))
    f["gh_asserts_cases_per_kloc"] = _num(payload.get("gh_asserts_cases_per_kloc"))
    f["gh_team_size"] = _num(payload.get("gh_team_size"), 1.0)

    # ---- G4: pipeline history (from the store) -----------------------------
    prev_missing = history.get("prev_build_failed") is None
    f["prev_build_failed"] = 0.0 if prev_missing else float(history["prev_build_failed"])
    f["prev_build_failed_was_missing"] = 1.0 if prev_missing else 0.0

    f["failure_rate_last_5"] = _num(history.get("failure_rate_last_5"))
    f["failure_rate_last_20"] = _num(history.get("failure_rate_last_20"))
    f["project_cum_failure_rate"] = _num(history.get("project_cum_failure_rate"))
    f["builds_so_far_in_project"] = _num(history.get("builds_so_far_in_project"))
    f["consecutive_prior_failures"] = _num(history.get("consecutive_prior_failures"))
    f["builds_in_last_24h"] = _num(history.get("builds_in_last_24h"))

    hours = history.get("hours_since_last_build")
    hours_missing = hours is None
    f["hours_since_last_build"] = (
        MEDIAN_HOURS_SINCE_LAST_BUILD if hours_missing else _num(hours))
    f["hours_since_last_build_was_missing"] = 1.0 if hours_missing else 0.0

    f["is_first_build_in_project"] = (
        1.0 if _num(history.get("builds_so_far_in_project")) == 0 else 0.0)

    # ---- G5: temporal / context --------------------------------------------
    ts_raw = payload.get("build_started_at")
    ts = _parse_timestamp(ts_raw)

    f["hour_of_day"] = float(ts.hour)
    f["day_of_week"] = float(ts.weekday())
    f["is_weekend"] = 1.0 if ts.weekday() >= 5 else 0.0
    f["month"] = float(ts.month)

    f["gh_is_pr"] = 1.0 if payload.get("is_pr") else 0.0
    f["gh_by_core_team_member"] = 1.0 if payload.get("by_core_team_member") else 0.0

    branch = str(payload.get("branch") or "").lower()
    f["is_default_branch"] = 1.0 if branch in DEFAULT_BRANCHES else 0.0

    lang = str(payload.get("language") or "").lower()
    f["lang_java"] = 1.0 if lang == "java" else 0.0

    return f


def _parse_timestamp(value: Any) -> datetime:
    """Parse an ISO timestamp; fall back to now() if absent or malformed."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            pass
    return datetime.now(timezone.utc)


def to_row(features: dict[str, float]) -> list[float]:
    """Ordered feature list matching FEATURE_ORDER. Order errors are silent — always use this."""
    return [float(features.get(name, 0.0)) for name in FEATURE_ORDER]


def validate_feature_order(model_feature_names: list[str]) -> None:
    """Fail loudly at startup if the model expects a different feature order."""
    if list(model_feature_names) != FEATURE_ORDER:
        missing = set(model_feature_names) - set(FEATURE_ORDER)
        extra = set(FEATURE_ORDER) - set(model_feature_names)
        raise RuntimeError(
            "Feature order mismatch between model and features.py.\n"
            f"  In model but not features.py: {sorted(missing) or 'none'}\n"
            f"  In features.py but not model: {sorted(extra) or 'none'}\n"
            "Fix FEATURE_ORDER to match models/feature_names.json exactly."
        )
