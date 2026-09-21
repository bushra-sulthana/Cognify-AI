"""
detector.py
-----------
Hallucination detection = claim extraction + evidence retrieval + verification
+ risk scoring.  This module glues those pieces together and returns one
structured report that both the API and the self-correction loop use.
"""

from typing import Dict, List, Optional

from backend.claim_extractor import extract_claims
from backend.generator import estimate_model_uncertainty
from backend.logger import get_logger
from backend.ollama_client import OllamaClient
from backend.rag import KnowledgeBase
from backend.risk_scorer import aggregate_signals, compute_risk
from backend.verifier import nli_uncertainty, verify_claims

log = get_logger("cognify.detector")


def detect(
    answer: str,
    kb: KnowledgeBase,
    client: OllamaClient,
    query: str = "",
    warnings: Optional[List[str]] = None,
) -> Dict:
    """
    Run the full detection stage on one answer.

    Returns a dictionary with: claims, results (per-claim verification),
    signals, risk_score, confidence_score, status, verification_status.
    """
    warnings = warnings if warnings is not None else []

    # 1. Which factual statements does the answer make?
    claims = extract_claims(client, answer)

    # 2. Check every claim against the local knowledge base.
    results = verify_claims(claims, kb)
    if kb.backend == "empty":
        warnings.append(
            "The knowledge base is empty, so no claim could be verified. "
            "Add documents to knowledge_base/ and restart the server."
        )
    if results and all(r["verifier"] == "lexical" for r in results):
        warnings.append(
            "Verification used the lightweight lexical fallback instead of the NLI model."
        )

    # 3. Uncertainty signals.
    model_uncertainty, uncertainty_reason = estimate_model_uncertainty(answer)
    verifier_uncertainty = nli_uncertainty(results)

    # 4. Aggregate -> risk score.
    signals = aggregate_signals(results, model_uncertainty, verifier_uncertainty)
    risk = compute_risk(signals)

    log.info(
        "Detection: %d claim(s) | supported=%d contradicted=%d insufficient=%d | "
        "risk=%d (%s) | model uncertainty %.2f (%s)",
        signals["total_claims"], signals["supported"], signals["contradicted"],
        signals["insufficient"], risk["risk_score"], risk["status"],
        model_uncertainty, uncertainty_reason,
    )

    return {
        "query": query,
        "answer": answer,
        "claims": claims,
        "results": results,
        "signals": signals,
        "warnings": warnings,
        **risk,
    }
