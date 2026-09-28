"""Release board web application.

Two surfaces: an HTML page for people, and a small JSON API for machines.
Run it with:

    python -m webapp.app

then open http://localhost:5000
"""
from __future__ import annotations

from flask import Flask, jsonify, render_template

from webapp.logic import classify, sample_releases, sort_releases, summarise

app = Flask(__name__)


@app.route("/")
def board():
    releases = sort_releases(sample_releases())
    for release in releases:
        release["health"] = classify(release["error_rate"])
    return render_template("index.html",
                           releases=releases,
                           summary=summarise(sample_releases()))


@app.route("/health")
def health():
    return jsonify({"status": "ok", "service": "release-board"})


@app.route("/api/releases")
def api_releases():
    releases = sort_releases(sample_releases())
    return jsonify([
        {
            "version": r["version"],
            "commit": r["commit"],
            "deployed_at": r["deployed_at"].isoformat(),
            "error_rate": r["error_rate"],
            "health": classify(r["error_rate"]),
        }
        for r in releases
    ])


if __name__ == "__main__":
    app.run(debug=True)
