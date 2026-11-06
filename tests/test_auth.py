"""Login / sign-up / protected-endpoint tests (offline, temporary database)."""

import pytest
from fastapi.testclient import TestClient

import backend.rag as rag

rag.knowledge_base.build = lambda *args, **kwargs: None  # type: ignore[assignment]

from backend import auth, config  # noqa: E402
from backend.main import app  # noqa: E402

PASSWORD = "correct-horse-battery"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(auth, "DB_PATH", tmp_path / "users.db")
    monkeypatch.setattr(config, "AUTH_REQUIRED", True)
    auth._FAILED.clear()
    return TestClient(app)


def signup(client, email="student@example.com", password=PASSWORD):
    return client.post("/api/auth/signup", json={"email": email, "password": password})


def test_chat_requires_login(client):
    assert client.post("/api/chat", json={"query": "hello"}).status_code == 401
    assert client.post("/api/verify", json={"answer": "x"}).status_code == 401


def test_signup_login_me_logout(client):
    assert signup(client).status_code == 201
    assert client.get("/api/auth/me").status_code == 401
    assert client.post("/api/auth/login", json={"email": "student@example.com", "password": PASSWORD}).status_code == 200
    assert client.get("/api/auth/me").json()["email"] == "student@example.com"
    assert client.post("/api/auth/logout").status_code == 200
    assert client.get("/api/auth/me").status_code == 401


def test_wrong_password_and_duplicate_email(client):
    signup(client)
    assert signup(client).status_code == 409
    bad = client.post("/api/auth/login", json={"email": "student@example.com", "password": "wrong-password"})
    assert bad.status_code == 401


def test_short_password_rejected(client):
    assert signup(client, password="short").status_code == 422


def test_login_is_throttled_after_repeated_failures(client):
    signup(client)
    for _ in range(5):
        client.post("/api/auth/login", json={"email": "student@example.com", "password": "nope-nope-nope"})
    r = client.post("/api/auth/login", json={"email": "student@example.com", "password": PASSWORD})
    assert r.status_code == 429


def test_logged_in_user_passes_the_gate(client):
    signup(client)
    client.post("/api/auth/login", json={"email": "student@example.com", "password": PASSWORD})
    # an empty query is a validation error (422), not 401 -> the login gate was passed
    assert client.post("/api/chat", json={"query": "  "}).status_code == 422
