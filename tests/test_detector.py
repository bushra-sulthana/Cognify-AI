"""Unit tests for claim extraction and claim verification (no LLM required)."""

from backend.claim_extractor import parse_claims_json, split_sentences
from backend.generator import estimate_model_uncertainty, is_refusal
from backend.verifier import CONTRADICTED, INSUFFICIENT, SUPPORTED, lexical_scores, verify_claim


# ----------------------------- claim extraction ------------------------- #
def test_parse_clean_json():
    raw = '{"claims": ["Paris is the capital of France.", "France is in Europe."]}'
    assert parse_claims_json(raw) == [
        "Paris is the capital of France.",
        "France is in Europe.",
    ]


def test_parse_json_wrapped_in_markdown_and_prose():
    raw = 'Sure! Here is the JSON:\n```json\n{"claims": ["Water boils at 100 C."]}\n```'
    assert parse_claims_json(raw) == ["Water boils at 100 C."]


def test_invalid_json_returns_empty_list_instead_of_crashing():
    assert parse_claims_json("this is not json at all") == []


def test_sentence_fallback_skips_questions_and_chatter():
    text = "Paris is the capital of France. Would you like more detail? I hope this helps."
    sentences = split_sentences(text)
    assert sentences == ["Paris is the capital of France."]


# ----------------------------- verification ----------------------------- #
def test_supported_claim(fake_kb):
    result = verify_claim("Paris is the capital of France.", fake_kb)
    assert result["status"] == SUPPORTED
    assert "Paris" in result["best_evidence"]


def test_contradicted_claim_detected_by_number_conflict(fake_kb):
    result = verify_claim("Water boils at 90 degrees Celsius at sea level.", fake_kb)
    assert result["status"] in {CONTRADICTED, INSUFFICIENT}


def test_claim_without_evidence_is_insufficient(fake_kb):
    result = verify_claim("Neptune has fourteen confirmed moons orbiting it.", fake_kb)
    assert result["status"] == INSUFFICIENT
    assert result["best_evidence"] == ""


def test_lexical_scores_sum_to_one():
    scores = lexical_scores(
        "Paris is the capital and largest city of France.", "Paris is the capital of France."
    )
    assert abs(sum(scores.values()) - 1.0) < 1e-6
    assert scores["entailment"] > 0.5


# ----------------------------- uncertainty ------------------------------ #
def test_hedging_raises_model_uncertainty():
    confident, _ = estimate_model_uncertainty(
        "Canberra is the capital of Australia and it was founded in 1913 as a planned city."
    )
    hedged, _ = estimate_model_uncertainty(
        "I think it might be Sydney, but I am not sure; possibly Canberra as well."
    )
    assert hedged > confident


def test_refusal_is_recognised():
    assert is_refusal("This information could not be verified against the knowledge base.")
    assert not is_refusal("Paris is the capital of France.")
