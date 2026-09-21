"""API tests. They run offline: Ollama and the NLI model are not required."""

import pytest
from fastapi.testclient import TestClient

import backend.rag as rag

# The knowledge base must not be rebuilt (and no model downloaded) during tests
rag.knowledge_base.build = lambda *args, **kwargs: None  # type: ignore[assignment]

from backend.main import app  # noqa: E402

client = TestClient(app)


def test_health_endpoint_always_answers():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in {"ok", "degraded"}
    assert "ollama_model" in data
    assert isinstance(data["message"], str)


def test_config_endpoint_exposes_weights_and_no_secrets():
    response = client.get("/api/config")
    assert response.status_code == 200
    data = response.json()
    assert data["risk_weights"]["contradiction"] == pytest.approx(0.40)
    assert data["max_iterations"] >= 1
    # Nothing that looks like a credential may ever be exposed
    text = str(data).lower()
    assert "api_key" not in text and "secret" not in text and "token" not in text


def test_empty_query_is_rejected_with_422():
    response = client.post("/api/chat", json={"query": "   "})
    assert response.status_code == 422


def test_missing_answer_is_rejected():
    response = client.post("/api/verify", json={"query": "hello"})
    assert response.status_code == 422


def test_verify_works_without_ollama_using_fallbacks(fake_kb, monkeypatch):
    """Ollama is down in CI, so claim extraction must fall back to sentence splitting."""
    monkeypatch.setattr(rag, "knowledge_base", fake_kb)
    import backend.main as main

    monkeypatch.setattr(main, "knowledge_base", fake_kb)

    response = client.post(
        "/api/verify",
        json={"query": "What is the capital of France?",
              "answer": "Paris is the capital of France."},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["claims"], "at least one claim should be extracted by the fallback"
    assert 0 <= data["risk_score"] <= 100
    assert data["confidence_score"] == 100 - data["risk_score"]
