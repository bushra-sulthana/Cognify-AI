"""
risk_scorer.py
--------------
Turns the verification signals into a transparent 0-100 "Hallucination Risk Score".

    Risk = 40% * contradiction risk
         + 30% * unsupported claim risk
         + 20% * NLI uncertainty
         + 10% * model uncertainty      (weights are configurable in config.py)

IMPORTANT (scientific honesty):
This number is an engineered indicator produced by this framework.  It is NOT a
calibrated probability that the answer is false.  The retriever, the NLI model
and the LLM can all be wrong, and the knowledge base is not complete.
"""

from typing import Dict, List

from backend import config
from backend.verifier import CONTRADICTED, INSUFFICIENT, SUPPORTED


def aggregate_signals(results: List[Dict], model_uncertainty: float, nli_uncertainty_value: float) -> Dict:
    """Convert per-claim verification results into the four risk signals (0..1)."""
    total = len(results)
    supported = sum(1 for r in results if r["status"] == SUPPORTED)
    contradicted = sum(1 for r in results if r["status"] == CONTRADICTED)
    insufficient = sum(1 for r in results if r["status"] == INSUFFICIENT)

    if total == 0:
        # No factual claims found: usually a refusal or a purely conversational
        # answer.  There is little to hallucinate about, so risk stays modest.
        contradiction_risk, unsupported_risk = 0.0, 0.20
        nli_uncertainty_value = max(nli_uncertainty_value, 0.20)
    else:
        contradiction_risk = contradicted / total
        unsupported_risk = insufficient / total

    return {
        "total_claims": total,
        "supported": supported,
        "contradicted": contradicted,
        "insufficient": insufficient,
        "contradiction_risk": round(contradiction_risk, 4),
        "unsupported_risk": round(unsupported_risk, 4),
        "nli_uncertainty": round(float(nli_uncertainty_value), 4),
        "model_uncertainty": round(float(model_uncertainty), 4),
    }


def compute_risk(signals: Dict) -> Dict:
    """Weighted sum -> risk score (0-100), confidence, risk band, verification status."""
    contributions = {
        "contradiction": config.W_CONTRADICTION * signals["contradiction_risk"],
        "unsupported": config.W_UNSUPPORTED * signals["unsupported_risk"],
        "nli_uncertainty": config.W_NLI_UNCERTAINTY * signals["nli_uncertainty"],
        "model_uncertainty": config.W_MODEL_UNCERTAINTY * signals["model_uncertainty"],
    }
    weight_sum = (
        config.W_CONTRADICTION
        + config.W_UNSUPPORTED
        + config.W_NLI_UNCERTAINTY
        + config.W_MODEL_UNCERTAINTY
    ) or 1.0

    raw = sum(contributions.values()) / weight_sum      # normalise to 0..1
    risk_score = int(round(max(0.0, min(1.0, raw)) * 100))
    confidence_score = 100 - risk_score

    if risk_score <= config.LOW_RISK_MAX:
        status = "LOW_RISK"
    elif risk_score <= config.MEDIUM_RISK_MAX:
        status = "MEDIUM_RISK"
    else:
        status = "HIGH_RISK"

    total = signals["total_claims"]
    if total == 0:
        verification_status = "NO_CLAIMS_FOUND"
    elif signals["contradicted"] > 0:
        verification_status = "CONTRADICTED"
    elif signals["supported"] == total:
        verification_status = "VERIFIED"
    elif signals["supported"] > 0:
        verification_status = "PARTIALLY_VERIFIED"
    else:
        verification_status = "UNVERIFIED"

    return {
        "risk_score": risk_score,
        "confidence_score": confidence_score,
        "status": status,
        "verification_status": verification_status,
        "breakdown": {
            "contradiction_risk": signals["contradiction_risk"],
            "unsupported_risk": signals["unsupported_risk"],
            "nli_uncertainty": signals["nli_uncertainty"],
            "model_uncertainty": signals["model_uncertainty"],
            "weighted_contribution": {k: round(v * 100, 2) for k, v in contributions.items()},
        },
    }


def needs_correction(risk_score: int) -> bool:
    """A correction round is triggered when the risk is above the threshold."""
    return risk_score > config.RISK_THRESHOLD
