"""Unit tests for per-statement risk highlighting."""

from backend.highlighter import build_highlights, claim_risk, level_for, split_segments
from backend.verifier import CONTRADICTED, INSUFFICIENT, SUPPORTED


def res(claim, status, entail=0.9, contra=0.02):
    return {"claim": claim, "status": status,
            "entailment_score": entail, "contradiction_score": contra, "neutral_score": 0.1}


def test_split_keeps_every_character():
    text = "Water boils at 100 C. It freezes at 0 C!\n\nIs that all? Dr. Smith agrees."
    assert "".join(split_segments(text)) == text


def test_claim_risk_anchors():
    assert claim_risk(res("a", SUPPORTED, entail=0.95)) <= 30
    assert claim_risk(res("a", INSUFFICIENT)) == 60
    assert claim_risk(res("a", CONTRADICTED, contra=0.5)) == 80
    assert claim_risk(res("a", CONTRADICTED, contra=0.95)) == 95


def test_levels_follow_30_60_bands():
    assert level_for(30) == "LOW"
    assert level_for(60) == "MEDIUM"
    assert level_for(80) == "HIGH"
    assert level_for(None) == "NONE"


def test_sentences_get_colours_from_their_claims():
    answer = "The Sun is a star. The Moon is made of cheese. Paris is lovely, I think."
    results = [
        res("The Sun is a star.", SUPPORTED),
        res("The Moon is made of cheese.", CONTRADICTED, contra=0.9),
    ]
    out = build_highlights(answer, results)
    assert "".join(s["text"] for s in out) == answer
    assert [s["level"] for s in out] == ["LOW", "HIGH", "NONE"]
    assert out[2]["risk"] is None


def test_worst_claim_wins_for_a_sentence():
    answer = "Mars is red and has two moons."
    results = [res("Mars is red.", SUPPORTED), res("Mars has two moons.", INSUFFICIENT)]
    assert build_highlights(answer, results)[0]["level"] == "MEDIUM"
