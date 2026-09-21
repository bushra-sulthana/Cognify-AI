"""
pipeline.py
-----------
The self-correction loop - the heart of Cognify-AI.

    generate -> detect -> score
        risk <= threshold  ->  return the answer
        risk >  threshold  ->  correct with evidence -> verify again
    stop when the risk is acceptable OR MAX_ITERATIONS is reached.

The loop can never run forever: the number of correction rounds is bounded by
MAX_ITERATIONS (default 3).
"""

import time
from typing import Dict, List, Optional

from backend import config
from backend.corrector import correct_answer
from backend.detector import detect
from backend.generator import generate_answer
from backend.logger import get_logger, shorten
from backend.ollama_client import OllamaClient, OllamaError
from backend.rag import KnowledgeBase
from backend.risk_scorer import needs_correction


log = get_logger("cognify.pipeline")


def _record(iteration: int, answer: str, report: Dict) -> Dict:
    return {
        "iteration": iteration,
        "answer": answer,
        "risk_score": report["risk_score"],
        "confidence_score": report["confidence_score"],
        "status": report["status"],
        "supported": report["signals"]["supported"],
        "contradicted": report["signals"]["contradicted"],
        "insufficient": report["signals"]["insufficient"],
    }


def run_pipeline(
    query: str,
    kb: KnowledgeBase,
    client: OllamaClient,
    max_iterations: Optional[int] = None,
) -> Dict:
    """Run generation + verification + self-correction for one user query."""
    started = time.time()
    max_iterations = config.MAX_ITERATIONS if max_iterations is None else max_iterations
    warnings: List[str] = []

    log.info("=" * 70)
    log.info("QUERY: %s", query)

    # ---- 1. initial generation --------------------------------------- #
    original_answer = generate_answer(client, query)
    answer = original_answer
    history: List[Dict] = []
    iterations = 0

    # ---- 2. detect -> correct -> detect ... --------------------------- #
    report = detect(answer, kb, client, query=query, warnings=warnings)
    history.append(_record(0, answer, report))

    while needs_correction(report["risk_score"]) and iterations < max_iterations:
        iterations += 1
        log.info(
            "Risk %d > threshold %d -> correction round %d/%d",
            report["risk_score"], config.RISK_THRESHOLD, iterations, max_iterations,
        )
        try:
            candidate = correct_answer(client, query, answer, report["results"])
        except OllamaError as exc:
            warnings.append(f"Correction round {iterations} failed: {exc}")
            iterations -= 1
            break

        candidate_report = detect(candidate, kb, client, query=query, warnings=warnings)
        history.append(_record(iterations, candidate, candidate_report))

        # Keep the rewrite only if it is not worse than what we already had.
        if candidate_report["risk_score"] <= report["risk_score"]:
            answer, report = candidate, candidate_report
        else:
            warnings.append(
                f"Correction round {iterations} increased the risk score "
                f"({report['risk_score']} -> {candidate_report['risk_score']}); "
                "the previous answer was kept."
            )
            break

    if needs_correction(report["risk_score"]) and iterations >= max_iterations:
        warnings.append(
            f"Maximum of {max_iterations} correction round(s) reached while the risk score "
            f"was still {report['risk_score']}. Treat this answer with caution."
        )

    elapsed = round(time.time() - started, 2)
    log.info(
        "FINAL: risk=%d confidence=%d status=%s iterations=%d (%.2fs) | %s",
        report["risk_score"], report["confidence_score"], report["status"],
        iterations, elapsed, shorten(answer),
    )

    return {
        "query": query,
        "original_answer": original_answer,
        "final_answer": answer,
        "risk_score": report["risk_score"],
        "confidence_score": report["confidence_score"],
        "status": report["status"],
        "verification_status": report["verification_status"],
        "iterations": iterations,
        "corrected": answer.strip() != original_answer.strip(),
        "claims": report["claims"],
        "evidence": report["results"],
        "risk_breakdown": report["breakdown"],
        "history": history,
        "warnings": warnings,
        "elapsed_seconds": elapsed,
    }
