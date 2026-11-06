"""Student level mode and attachment handling (no Ollama needed)."""

import base64

import pytest

from backend import config
from backend.attachments import (
    AttachmentError, EvidenceOverlay, build_context, describe_images, parse_attachments,
)
from backend.generator import generate_answer
from backend.levels import level_instruction, normalise_level


def b64(text: str) -> str:
    return base64.b64encode(text.encode()).decode()


class FakeClient:
    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return "ok"


def test_levels_have_distinct_instructions_and_default_to_college():
    texts = {level_instruction(n) for n in (1, 2, 3)}
    assert len(texts) == 3
    assert "school" in level_instruction(1).lower()
    assert "college" in level_instruction(2).lower()
    assert "postgraduate" in level_instruction(3).lower()
    assert normalise_level(None) == 2 and normalise_level(9) == 2


def test_generator_puts_the_level_into_the_prompt():
    client = FakeClient()
    generate_answer(client, "What is DNA?", level=1)
    assert "school student" in client.calls[0]["system"]
    generate_answer(client, "What is DNA?", level=3)
    assert "postgraduate" in client.calls[1]["system"]


def test_document_text_is_decoded_and_used_as_context_and_evidence(fake_kb):
    parsed = parse_attachments([{"name": "notes.txt", "data": b64("Mitochondria produce ATP in the cell.")}])
    assert "notes.txt" in parsed.documents
    assert "Mitochondria" in build_context(parsed.documents, {}, "What makes ATP?")

    overlay = EvidenceOverlay(fake_kb, parsed.documents)
    hits = overlay.search("Mitochondria produce ATP")
    assert any(h["source"] == "attachment: notes.txt" for h in hits)
    assert overlay.backend == "keyword"


def test_bad_attachments_are_rejected():
    with pytest.raises(AttachmentError):
        parse_attachments([{"name": "virus.exe", "data": b64("x")}])
    with pytest.raises(AttachmentError):
        parse_attachments([{"name": "a.txt", "data": ""}])
    too_many = [{"name": f"{i}.txt", "data": b64("hi")} for i in range(config.MAX_ATTACHMENTS + 1)]
    with pytest.raises(AttachmentError):
        parse_attachments(too_many)


def test_photo_without_vision_model_is_skipped_with_a_note(monkeypatch):
    monkeypatch.setattr(config, "OLLAMA_VISION_MODEL", "")
    parsed = parse_attachments([{"name": "pic.png", "data": b64("not-really-a-png")}])
    notes = []
    assert describe_images(FakeClient(), parsed.images, notes) == {}
    assert "no vision model" in notes[0]


def test_photo_with_vision_model_is_described(monkeypatch):
    monkeypatch.setattr(config, "OLLAMA_VISION_MODEL", "llava")
    parsed = parse_attachments([{"name": "pic.png", "data": b64("not-really-a-png")}])
    client = FakeClient()
    out = describe_images(client, parsed.images, [])
    assert out == {"pic.png": "ok"}
    assert client.calls[0]["model"] == "llava" and client.calls[0]["images"]


def test_chat_endpoint_end_to_end_with_level_and_attachment(fake_kb, monkeypatch):
    """Whole pipeline with a fake LLM: level reaches the prompt, attachment becomes evidence."""
    from fastapi.testclient import TestClient

    import backend.main as main
    from backend.ollama_client import OllamaClient

    prompts = []

    def fake_generate(self, prompt, system=None, **kwargs):
        prompts.append((system or "", prompt))
        if "factual claims" in prompt:                       # claim extraction call
            return '{"claims": ["Mitochondria produce ATP in the cell."]}'
        return "Mitochondria produce ATP in the cell."

    monkeypatch.setattr(OllamaClient, "generate", fake_generate)
    monkeypatch.setattr(main, "knowledge_base", fake_kb)

    r = TestClient(main.app).post("/api/chat", json={
        "query": "What do mitochondria do?",
        "level": 1,
        "attachments": [{"name": "bio.txt", "data": b64("Mitochondria produce ATP in the cell.")}],
    })
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["level"] == 1 and data["level_name"] == "School"
    assert any("school student" in s for s, _ in prompts)
    assert any("bio.txt" in p for _, p in prompts)             # attachment text reached the model
    assert any(e["best_evidence_source"] == "attachment: bio.txt" for e in data["evidence"])
    assert data["highlights"] and data["highlights"][0]["level"] == "LOW"


def test_chat_rejects_unsupported_attachment_with_400(fake_kb, monkeypatch):
    from fastapi.testclient import TestClient

    import backend.main as main
    monkeypatch.setattr(main, "knowledge_base", fake_kb)
    r = TestClient(main.app).post("/api/chat", json={
        "query": "hi there", "attachments": [{"name": "run.exe", "data": b64("x")}]})
    assert r.status_code == 400


def test_reasoning_text_is_removed_from_model_output():
    from backend.generator import _drop_unfinished_sentence
    from backend.ollama_client import strip_reasoning

    leaked = "Hmm, the user asks about DNA. I should be brief.\n</think>\n\nDNA stores genetic information."
    assert strip_reasoning(leaked) == "DNA stores genetic information."
    assert strip_reasoning("<think>plan</think>Answer.") == "Answer."
    assert strip_reasoning("<think>never finished") == ""
    assert _drop_unfinished_sentence("Plants make food. They use light and") == "Plants make food."
    assert _drop_unfinished_sentence("Complete sentence.") == "Complete sentence."


def test_thinking_models_get_think_false(monkeypatch):
    import backend.ollama_client as oc

    sent = {}

    class Resp:
        status_code = 200
        text = ""
        def json(self):
            return {"response": "Hidden plan\n</think>\nFinal answer."}

    def fake_post(url, json=None, timeout=None):
        sent.update(json)
        return Resp()

    monkeypatch.setattr(oc.requests, "post", fake_post)
    client = oc.OllamaClient()
    monkeypatch.setattr(client, "model", "qwen3:4b", raising=False)
    assert client.generate("Q?") == "Final answer."
    assert sent["think"] is False and sent["prompt"].endswith("/no_think")


def test_correction_must_be_tagged_and_not_reasoning():
    from backend.corrector import CorrectionRejected, extract_answer, looks_like_reasoning

    assert extract_answer("<answer>Plants make sugar from light.</answer>") == "Plants make sugar from light."
    with pytest.raises(CorrectionRejected):
        extract_answer("We are given an answer to the question ...")
    leaked = ('We are given an answer to the question "what is photosynthesis?" and a list of claims. '
              'The trusted evidence from educationq.txt includes: ...')
    assert looks_like_reasoning(leaked, "Photosynthesis is how plants make food.")
    assert not looks_like_reasoning("Plants turn light, water and CO2 into sugar and oxygen.",
                                    "Photosynthesis is how plants make food.")


def test_pipeline_keeps_answer_when_model_rambles(fake_kb, monkeypatch):
    """The exact failure from the screenshot: the 'correction' is the model's reasoning."""
    from backend.ollama_client import OllamaClient
    from backend.pipeline import run_pipeline

    def fake_generate(self, prompt, system=None, **kwargs):
        if "factual claims" in prompt:
            return '{"claims": ["Plants make glucose using light."]}'
        if "fact-grounded editor" in (system or ""):
            return 'We are given an answer to the question "x" and a list of claims. Let\'s check the evidence.'
        return "Plants make glucose using light."

    monkeypatch.setattr(OllamaClient, "generate", fake_generate)
    out = run_pipeline("What is photosynthesis?", fake_kb, OllamaClient(), level=2)
    assert out["final_answer"] == "Plants make glucose using light."
    assert "We are given" not in out["final_answer"]
