"""
Build history store.

WHY THIS EXISTS
---------------
The model's four strongest features (~86% of gain) describe a project's RECENT
BUILD HISTORY: previous outcome, consecutive failures, failure rate over the
last 5 and 20 builds. A CI runner invoking /predict knows about the current
commit but has no idea how the last twenty builds went.

So the service is stateful. It keeps a rolling window of outcomes per project
and derives the history features from it. The CI pipeline reports each build's
result back via POST /outcome once the build finishes.

This mirrors the notebook exactly: features are computed from builds STRICTLY
BEFORE the current one, which is the serving-time equivalent of the .shift(1)
in training. Never record an outcome before predicting on it.

SQLite is used for durability across restarts. It is single-writer, which is
fine at CI volumes (a busy monorepo does maybe a few builds a minute). If you
outgrow it, swap in Postgres or Redis behind the same interface.
"""
from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

WINDOW = 20  # longest rolling window the features need


class HistoryStore:
    def __init__(self, db_path: str = "history.db") -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS builds (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    project     TEXT    NOT NULL,
                    build_ref   TEXT,
                    failed      INTEGER NOT NULL,
                    finished_at TEXT    NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_project_time "
                "ON builds (project, finished_at DESC)")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    project      TEXT    NOT NULL,
                    build_ref    TEXT,
                    risk_score   REAL    NOT NULL,
                    decision     TEXT    NOT NULL,
                    threshold    REAL    NOT NULL,
                    predicted_at TEXT    NOT NULL
                )
            """)

    # ---------------------------------------------------------------- writes
    def record_outcome(self, project: str, failed: bool,
                       build_ref: str | None = None,
                       finished_at: datetime | None = None) -> None:
        """Record a completed build. Call this AFTER the build finishes."""
        ts = (finished_at or datetime.now(timezone.utc)).isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO builds (project, build_ref, failed, finished_at) "
                "VALUES (?, ?, ?, ?)",
                (project, build_ref, int(bool(failed)), ts))

    def record_prediction(self, project: str, build_ref: str | None,
                          risk_score: float, decision: str,
                          threshold: float) -> None:
        """Log a prediction so you can measure live accuracy later."""
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO predictions "
                "(project, build_ref, risk_score, decision, threshold, predicted_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (project, build_ref, float(risk_score), decision, float(threshold),
                 datetime.now(timezone.utc).isoformat()))

    # ---------------------------------------------------------------- reads
    def summarise(self, project: str, now: datetime | None = None) -> dict:
        """
        Derive the G4 history features for a project from builds recorded so far.

        Returns None for prev_build_failed and hours_since_last_build when the
        project has no history — features.py turns those into missing-indicator
        flags rather than silently imputing zero.
        """
        now = now or datetime.now(timezone.utc)

        with self._lock, self._connect() as conn:
            recent = conn.execute(
                "SELECT failed, finished_at FROM builds WHERE project = ? "
                "ORDER BY finished_at DESC, id DESC LIMIT ?",
                (project, WINDOW)).fetchall()

            totals = conn.execute(
                "SELECT COUNT(*) AS n, COALESCE(SUM(failed), 0) AS failures "
                "FROM builds WHERE project = ?", (project,)).fetchone()

            cutoff = (now - timedelta(hours=24)).isoformat()
            last_24h = conn.execute(
                "SELECT COUNT(*) AS n FROM builds "
                "WHERE project = ? AND finished_at >= ?",
                (project, cutoff)).fetchone()

        total_builds = int(totals["n"])
        total_failures = int(totals["failures"])

        if total_builds == 0:
            return {
                "prev_build_failed": None,
                "failure_rate_last_5": 0.0,
                "failure_rate_last_20": 0.0,
                "project_cum_failure_rate": 0.0,
                "builds_so_far_in_project": 0,
                "consecutive_prior_failures": 0,
                "hours_since_last_build": None,
                "builds_in_last_24h": 0,
            }

        # recent[0] is the most recent build
        outcomes = [int(r["failed"]) for r in recent]

        last5 = outcomes[:5]
        last20 = outcomes[:20]

        # run length of failures ending at the most recent build
        consecutive = 0
        for o in outcomes:
            if o == 1:
                consecutive += 1
            else:
                break

        try:
            last_finished = datetime.fromisoformat(recent[0]["finished_at"])
            if last_finished.tzinfo is None:
                last_finished = last_finished.replace(tzinfo=timezone.utc)
            hours_since = (now - last_finished).total_seconds() / 3600.0
        except (ValueError, TypeError):
            hours_since = None

        return {
            "prev_build_failed": outcomes[0],
            "failure_rate_last_5": sum(last5) / len(last5),
            "failure_rate_last_20": sum(last20) / len(last20),
            "project_cum_failure_rate": total_failures / total_builds,
            "builds_so_far_in_project": total_builds,
            "consecutive_prior_failures": consecutive,
            "hours_since_last_build": hours_since,
            "builds_in_last_24h": int(last_24h["n"]),
        }

    def project_stats(self, project: str) -> dict:
        """Summary for the dashboard / debugging."""
        with self._lock, self._connect() as conn:
            builds = conn.execute(
                "SELECT COUNT(*) AS n, COALESCE(SUM(failed), 0) AS f "
                "FROM builds WHERE project = ?", (project,)).fetchone()
            preds = conn.execute(
                "SELECT COUNT(*) AS n FROM predictions WHERE project = ?",
                (project,)).fetchone()
        return {
            "project": project,
            "builds_recorded": int(builds["n"]),
            "failures_recorded": int(builds["f"]),
            "predictions_made": int(preds["n"]),
        }

    def list_projects(self) -> list[str]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT project FROM builds ORDER BY project").fetchall()
        return [r["project"] for r in rows]

    def reset(self) -> None:
        """Wipe all data. Tests only."""
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM builds")
            conn.execute("DELETE FROM predictions")
