"""
highlighter.py
--------------
Per-statement hallucination-risk highlighting.

The detector already verifies every extracted claim.  This module maps those
claim results back onto the sentences of the answer the user actually sees, so
the frontend can colour each statement:

    risk 30  -> GREEN   (low risk:    claim supported by evidence)
    risk 60  -> YELLOW  (medium risk: not enough evidence to confirm it)
    risk 80+ -> RED     (high risk:   evidence contradicts the claim)

Sentences that no claim maps to (greetings, opinions, questions ...) get
level "NONE" and are left uncoloured - we do not invent a risk for them.

IMPORTANT (scientific honesty): like the overall risk score, these values are
engineered indicators, not calibrated probabilities.
"""

import re
from typing import Dict, List, Optional

from backend import config
from backend.rag import tokenize
from backend.verifier import CONTRADICTED, INSUFFICIENT, SUPPORTED

RISK_SUPPORTED = 30
RISK_INSUFFICIENT = 60
RISK_CONTRADICTED = 80

# A claim is attached to a sentence when this share of the claim's content
# words appear in the sentence.  Claims are rewritten by the LLM (pronouns
# replaced), so an exact string match would miss many of them.
MATCH_MIN_OVERLAP = 0.5


def split_segments(text: str) -> List[str]:
    """Split text into sentence-like pieces WITHOUT losing any characters.

    Joining the returned list always gives back the original text, so the
    frontend can render the answer exactly as before - only coloured.
    """
    protected = re.sub(r"\b(Mr|Mrs|Dr|Prof|St|etc|e\.g|i\.e)\.", r"\1<DOT>", text)
    # Break after . ! ? (or a newline) and keep trailing whitespace attached.
    parts = re.findall(r".+?(?:[.!?]+(?=\s|$)|\n+|$)\s*", protected, flags=re.S)
    segments = [p.replace("<DOT>", ".") for p in parts if p]
    if "".join(segments) != text:        # safety net: never alter the text
        return [text]
    return segments


def claim_risk(result: Dict) -> int:
    """Risk (0-100) of ONE verified claim, anchored on 30 / 60 / 80."""
    status = result.get("status")
    if status == CONTRADICTED:
        return max(RISK_CONTRADICTED, int(round(result.get("contradiction_score", 0.0) * 100)))
    if status == INSUFFICIENT:
        return RISK_INSUFFICIENT
    if status == SUPPORTED:
        weak = int(round((1.0 - result.get("entailment_score", 1.0)) * 100))
        return min(RISK_SUPPORTED, weak)
    return RISK_INSUFFICIENT


def level_for(risk: Optional[int]) -> str:
    if risk is None:
        return "NONE"
    if risk <= config.LOW_RISK_MAX:
        return "LOW"
    if risk <= config.MEDIUM_RISK_MAX:
        return "MEDIUM"
    return "HIGH"


def _overlap(claim: str, sentence: str) -> float:
    claim_tokens = set(tokenize(claim))
    if not claim_tokens:
        return 0.0
    return len(claim_tokens & set(tokenize(sentence))) / len(claim_tokens)


def build_highlights(answer: str, results: List[Dict]) -> List[Dict]:
    """Return [{text, risk, level, status, claim}] covering the whole answer."""
    highlights: List[Dict] = []
    for segment in split_segments(answer):
        risk: Optional[int] = None
        matched: Optional[Dict] = None
        for result in results:
            if _overlap(result["claim"], segment) >= MATCH_MIN_OVERLAP:
                value = claim_risk(result)
                if risk is None or value > risk:      # worst claim wins
                    risk, matched = value, result
        highlights.append(
            {
                "text": segment,
                "risk": risk,
                "level": level_for(risk),
                "status": matched["status"] if matched else "",
                "claim": matched["claim"] if matched else "",
            }
        )
    return highlights
