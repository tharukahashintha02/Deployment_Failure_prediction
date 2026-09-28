"""Release board logic.

Kept separate from the Flask routes so it can be unit tested without starting a
server. These are the functions most likely to be broken by a careless change,
which is what makes them useful in a gating demonstration.
"""
from __future__ import annotations

from datetime import datetime

# Anything at or above this error rate is considered unhealthy.
ERROR_BUDGET = 1.0


def parse_version(tag: str) -> tuple[int, int, int]:
    """Turn a tag such as 'v2.4.1' or '2.4.1' into (2, 4, 1)."""
    cleaned = tag.strip().lstrip("vV")
    parts = cleaned.split(".")
    if len(parts) != 3:
        raise ValueError(f"Not a three-part version: {tag!r}")
    try:
        return tuple(int(p) for p in parts)  # type: ignore[return-value]
    except ValueError as exc:
        raise ValueError(f"Version parts must be numeric: {tag!r}") from exc


def is_newer(candidate: str, current: str) -> bool:
    """True when candidate is a later version than current."""
    return parse_version(candidate) > parse_version(current)


def sort_releases(releases: list[dict]) -> list[dict]:
    """Newest release first, by deployment time."""
    return sorted(releases, key=lambda r: r["deployed_at"], reverse=True)


def classify(error_rate: float) -> str:
    """Health of a release from its observed error rate."""
    if error_rate < 0:
        raise ValueError("Error rate cannot be negative")
    if error_rate >= ERROR_BUDGET:
        return "degraded"
    if error_rate > 0:
        return "watch"
    return "healthy"


def summarise(releases: list[dict]) -> dict:
    """Overall board state: what is live, and how the recent record looks."""
    if not releases:
        return {"live": None, "total": 0, "degraded": 0, "healthy_streak": 0}

    ordered = sort_releases(releases)
    live = ordered[0]

    streak = 0
    for release in ordered:
        if classify(release["error_rate"]) == "degraded":
            break
        streak += 1

    return {
        "live": live,
        "total": len(ordered),
        "degraded": sum(1 for r in ordered if classify(r["error_rate"]) == "degraded"),
        "healthy_streak": streak,
    }


def sample_releases() -> list[dict]:
    """Stand-in data. A real board would read this from a deployment record."""
    return [
        {"version": "v2.4.1", "commit": "0b7e1be", "author": "tharuka",
         "deployed_at": datetime(2026, 9, 28, 14, 12), "error_rate": 0.0,
         "note": "Cache warm-up on boot"},
        {"version": "v2.4.0", "commit": "5eafa4c", "author": "tharuka",
         "deployed_at": datetime(2026, 9, 27, 9, 40), "error_rate": 0.4,
         "note": "Batch export endpoint"},
        {"version": "v2.3.7", "commit": "12c24d0", "author": "tharuka",
         "deployed_at": datetime(2026, 9, 25, 16, 5), "error_rate": 2.6,
         "note": "Session handling rewrite"},
        {"version": "v2.3.6", "commit": "4e5d226", "author": "tharuka",
         "deployed_at": datetime(2026, 9, 24, 11, 22), "error_rate": 0.0,
         "note": "Dependency bump"},
    ]
