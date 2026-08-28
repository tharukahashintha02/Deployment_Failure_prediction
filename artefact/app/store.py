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

TWO BACKENDS
------------
SQLite (default)  - zero setup, used for local development and the test suite.
PostgreSQL        - used in deployment, selected automatically when the
                    DATABASE_URL environment variable is set.

The Postgres path exists because free-tier container filesystems are ephemeral.
Without an external database the history table is wiped on every restart, every
project silently reverts to cold start, and the service keeps returning
plausible-looking scores while running on code features alone (~0.64 AUC
instead of ~0.88).
"""
from __future__ import annotations

import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

WINDOW = 20  # longest rolling window the features need


def _is_postgres(url: str | None) -> bool:
    if not url:
        return False
    return urlparse(url).scheme in ("postgres", "postgresql", "postgresql+psycopg2")


class HistoryStore:
    """Per-project build history. Backend selected by DATABASE_URL."""

    def __init__(self, db_path: str = "history.db", database_url: str | None = None) -> None:
        self.database_url = database_url if database_url is not None else os.getenv("DATABASE_URL")
        self.is_pg = _is_postgres(self.database_url)
        self.db_path = db_path
        self._lock = threading.Lock()

        if self.is_pg:
            import psycopg2                 # imported lazily: SQLite users need no driver
            import psycopg2.extras
            self._psycopg2 = psycopg2
            self._extras = psycopg2.extras
            # SQLAlchemy-style prefixes are common in hosting dashboards but
            # psycopg2 does not understand them
            dsn = self.database_url.replace("postgresql+psycopg2://", "postgresql://")
            self._dsn = dsn.replace("postgres://", "postgresql://")
            self.backend = "postgresql"
        else:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            self.backend = "sqlite"

        self._init_schema()

    # ------------------------------------------------------------- plumbing
    def _connect(self):
        if self.is_pg:
            conn = self._psycopg2.connect(self._dsn, connect_timeout=10,
                                          cursor_factory=self._extras.RealDictCursor)
            conn.autocommit = True
            return conn
        import sqlite3
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _q(self, sql: str) -> str:
        """SQLite uses ? placeholders; psycopg2 uses %s."""
        return sql.replace("?", "%s") if self.is_pg else sql

    def _execute(self, sql: str, params: tuple = ()) -> None:
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.cursor()
                cur.execute(self._q(sql), params)
                if not self.is_pg:
                    conn.commit()
            finally:
                conn.close()

    def _fetch(self, sql: str, params: tuple = (), one: bool = False):
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.cursor()
                cur.execute(self._q(sql), params)
                rows = [dict(r) for r in cur.fetchall()]
            finally:
                conn.close()
        if one:
            return rows[0] if rows else None
        return rows

    def _init_schema(self) -> None:
        serial = "SERIAL PRIMARY KEY" if self.is_pg else "INTEGER PRIMARY KEY AUTOINCREMENT"
        self._execute(f"""
            CREATE TABLE IF NOT EXISTS builds (
                id          {serial},
                project     TEXT    NOT NULL,
                build_ref   TEXT,
                failed      INTEGER NOT NULL,
                finished_at TEXT    NOT NULL
            )
        """)
        self._execute(
            "CREATE INDEX IF NOT EXISTS idx_project_time ON builds (project, finished_at DESC)")
        self._execute(f"""
            CREATE TABLE IF NOT EXISTS predictions (
                id           {serial},
                project      TEXT    NOT NULL,
                build_ref    TEXT,
                risk_score   REAL    NOT NULL,
                decision     TEXT    NOT NULL,
                threshold    REAL    NOT NULL,
                predicted_at TEXT    NOT NULL
            )
        """)

    # --------------------------------------------------------------- writes
    def record_outcome(self, project: str, failed: bool,
                       build_ref: str | None = None,
                       finished_at: datetime | None = None) -> None:
        """Record a completed build. Call this AFTER the build finishes."""
        ts = (finished_at or datetime.now(timezone.utc)).isoformat()
        self._execute(
            "INSERT INTO builds (project, build_ref, failed, finished_at) VALUES (?, ?, ?, ?)",
            (project, build_ref, int(bool(failed)), ts))

    def record_prediction(self, project: str, build_ref: str | None,
                          risk_score: float, decision: str, threshold: float) -> None:
        """Log a prediction so live accuracy can be measured against outcomes later."""
        self._execute(
            "INSERT INTO predictions "
            "(project, build_ref, risk_score, decision, threshold, predicted_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (project, build_ref, float(risk_score), decision, float(threshold),
             datetime.now(timezone.utc).isoformat()))

    # ---------------------------------------------------------------- reads
    def summarise(self, project: str, now: datetime | None = None) -> dict:
        """
        Derive the Group 4 history features from builds recorded so far.

        Returns None for prev_build_failed and hours_since_last_build when the
        project has no history; features.py converts those into missing-value
        indicator flags rather than silently imputing zero.
        """
        now = now or datetime.now(timezone.utc)

        recent = self._fetch(
            "SELECT failed, finished_at FROM builds WHERE project = ? "
            "ORDER BY finished_at DESC, id DESC LIMIT ?", (project, WINDOW))

        totals = self._fetch(
            "SELECT COUNT(*) AS n, COALESCE(SUM(failed), 0) AS failures "
            "FROM builds WHERE project = ?", (project,), one=True)

        cutoff = (now - timedelta(hours=24)).isoformat()
        last_24h = self._fetch(
            "SELECT COUNT(*) AS n FROM builds WHERE project = ? AND finished_at >= ?",
            (project, cutoff), one=True)

        total_builds = int(totals["n"]) if totals else 0
        total_failures = int(totals["failures"]) if totals else 0

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

        outcomes = [int(r["failed"]) for r in recent]   # recent[0] is most recent
        last5, last20 = outcomes[:5], outcomes[:20]

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
        except (ValueError, TypeError, KeyError, IndexError):
            hours_since = None

        return {
            "prev_build_failed": outcomes[0],
            "failure_rate_last_5": sum(last5) / len(last5),
            "failure_rate_last_20": sum(last20) / len(last20),
            "project_cum_failure_rate": total_failures / total_builds,
            "builds_so_far_in_project": total_builds,
            "consecutive_prior_failures": consecutive,
            "hours_since_last_build": hours_since,
            "builds_in_last_24h": int(last_24h["n"]) if last_24h else 0,
        }

    def project_stats(self, project: str) -> dict:
        builds = self._fetch(
            "SELECT COUNT(*) AS n, COALESCE(SUM(failed), 0) AS f FROM builds WHERE project = ?",
            (project,), one=True)
        preds = self._fetch(
            "SELECT COUNT(*) AS n FROM predictions WHERE project = ?", (project,), one=True)
        return {
            "project": project,
            "builds_recorded": int(builds["n"]) if builds else 0,
            "failures_recorded": int(builds["f"]) if builds else 0,
            "predictions_made": int(preds["n"]) if preds else 0,
        }

    def list_projects(self) -> list[str]:
        rows = self._fetch("SELECT DISTINCT project FROM builds ORDER BY project")
        return [r["project"] for r in rows]

    def health(self) -> dict:
        """Backend status, surfaced by /health so a broken database is visible."""
        try:
            self._fetch("SELECT 1 AS ok", one=True)
            return {"backend": self.backend, "connected": True, "persistent": self.is_pg}
        except Exception as exc:  # noqa: BLE001
            return {"backend": self.backend, "connected": False,
                    "persistent": self.is_pg, "error": str(exc)[:200]}

    def reset(self) -> None:
        """Wipe all data. Tests only."""
        self._execute("DELETE FROM builds")
        self._execute("DELETE FROM predictions")
