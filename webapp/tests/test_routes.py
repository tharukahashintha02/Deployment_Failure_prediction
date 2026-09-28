"""Tests for the HTTP surface of the release board."""
import pytest

from webapp.app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_board_page_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "Currently deployed" in body
    assert "Recent releases" in body


def test_board_shows_the_live_version(client):
    body = client.get("/").get_data(as_text=True)
    assert "v2.5.0" in body
    assert 'id="live-release"' in body


def test_api_returns_releases_newest_first(client):
    response = client.get("/api/releases")
    assert response.status_code == 200
    data = response.get_json()
    assert len(data) == 5
    assert data[0]["version"] == "v2.5.0"
    assert data[0]["health"] == "healthy"


def test_api_flags_a_degraded_release(client):
    data = client.get("/api/releases").get_json()
    degraded = [r for r in data if r["health"] == "degraded"]
    assert degraded and degraded[0]["version"] == "v2.3.7"
