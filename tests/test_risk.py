"""Unit tests for the transparent risk-score formula."""

from backend import config
from backend.risk_scorer import aggregate_signals, compute_risk, needs_correction
from backend.verifier import CONTRADICTED, INSUFFICIENT, SUPPORTED


def make_result(status: str, entail=0.9, contra=0.02, neutral=0.08):
    return {
        "status": status,
        "entailment_score": entail,
        "contradiction_score": contra,
        "neutral_score": neutral,
    }


def test_all_supported_gives_low_risk():
    results = [make_result(SUPPORTED), make_result(SUPPORTED)]
    signals = aggregate_signals(results, model_uncertainty=0.0, nli_uncertainty_value=0.1)
    risk = compute_risk(signals)
    assert risk["risk_score"] <= config.LOW_RISK_MAX
    assert risk["status"] == "LOW_RISK"
    assert risk["verification_status"] == "VERIFIED"
    assert risk["confidence_score"] == 100 - risk["risk_score"]


def test_contradicted_claims_push_risk_up():
    results = [make_result(CONTRADICTED, entail=0.05, contra=0.9, neutral=0.05)]
    signals = aggregate_signals(results, model_uncertainty=0.5, nli_uncertainty_value=0.1)
    risk = compute_risk(signals)
    assert risk["risk_score"] >= 40
    assert risk["verification_status"] == "CONTRADICTED"
    assert needs_correction(risk["risk_score"]) is True


def test_unsupported_claims_are_medium_risk():
    results = [make_result(INSUFFICIENT, entail=0.3, contra=0.1, neutral=0.6)]
    signals = aggregate_signals(results, model_uncertainty=0.2, nli_uncertainty_value=0.4)
    risk = compute_risk(signals)
    assert risk["verification_status"] == "UNVERIFIED"
    assert 0 <= risk["risk_score"] <= 100


def test_score_is_bounded_and_confidence_is_complement():
    """Worst case on every signal must saturate at exactly 100."""
    worst = {
        "total_claims": 2, "supported": 0, "contradicted": 1, "insufficient": 1,
        "contradiction_risk": 1.0, "unsupported_risk": 1.0,
        "nli_uncertainty": 1.0, "model_uncertainty": 1.0,
    }
    risk = compute_risk(worst)
    assert risk["risk_score"] == 100
    assert risk["confidence_score"] == 0
    assert risk["status"] == "HIGH_RISK"

    best = {
        "total_claims": 2, "supported": 2, "contradicted": 0, "insufficient": 0,
        "contradiction_risk": 0.0, "unsupported_risk": 0.0,
        "nli_uncertainty": 0.0, "model_uncertainty": 0.0,
    }
    assert compute_risk(best)["risk_score"] == 0


def test_no_claims_is_handled_gracefully():
    signals = aggregate_signals([], model_uncertainty=0.1, nli_uncertainty_value=0.0)
    risk = compute_risk(signals)
    assert signals["total_claims"] == 0
    assert risk["verification_status"] == "NO_CLAIMS_FOUND"
    assert risk["risk_score"] < 40
