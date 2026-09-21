"""
generator.py
------------
Step 1 of the pipeline: ask the local LLM for an answer.

It also produces a cheap "model uncertainty" signal.  Ollama does not expose
token log-probabilities through its simple /api/generate endpoint, so instead we
look at linguistic hedging ("maybe", "I think", "possibly").  This is a
heuristic, not a calibrated probability, and it only carries 10% of the final
risk score.
"""

import re
from typing import Tuple

from backend.logger import get_logger, shorten
from backend.ollama_client import OllamaClient

log = get_logger("cognify.generator")

ANSWER_SYSTEM_PROMPT = (
    "You are a careful factual assistant. Answer the user's question directly and "
    "concisely in 2-5 sentences. State facts plainly. If you are not sure about "
    "something, say so instead of inventing details."
)

# Words that suggest the model is unsure about its own output
HEDGE_PATTERNS = [
    r"\bi think\b", r"\bi believe\b", r"\bmaybe\b", r"\bperhaps\b", r"\bpossibly\b",
    r"\bprobably\b", r"\bmight be\b", r"\bcould be\b", r"\bnot sure\b", r"\bunsure\b",
    r"\bas far as i know\b", r"\bi am not certain\b", r"\bit seems\b", r"\bapparently\b",
    r"\bsome sources\b", r"\bgenerally\b",
]

# Phrases that mean the model correctly refused / admitted a knowledge gap.
REFUSAL_PATTERNS = [
    r"could not be verified", r"cannot be verified", r"no information",
    r"not available in the (provided )?evidence", r"i do not have", r"i don't have",
    r"insufficient evidence", r"the evidence does not", r"not supported by the evidence",
]


def generate_answer(client: OllamaClient, query: str) -> str:
    """Generate the initial (unverified) answer for a user question."""
    answer = client.generate(prompt=query, system=ANSWER_SYSTEM_PROMPT)
    log.info("Initial response: %s", shorten(answer))
    return answer


def is_refusal(text: str) -> bool:
    """True when the answer honestly states that it could not verify something."""
    lowered = text.lower()
    return any(re.search(p, lowered) for p in REFUSAL_PATTERNS)


def estimate_model_uncertainty(text: str) -> Tuple[float, str]:
    """
    Return (uncertainty in 0..1, short explanation).

    Heuristic:
      * each hedging expression adds 0.15 (capped at 0.6)
      * a very short answer (< 40 chars) adds 0.2
      * an honest 'could not be verified' answer is NOT penalised
    """
    lowered = text.lower()
    hedges = [p for p in HEDGE_PATTERNS if re.search(p, lowered)]
    score = min(0.6, 0.15 * len(hedges))

    if len(text.strip()) < 40:
        score += 0.2

    if is_refusal(text):
        score = min(score, 0.2)  # admitting a gap is good behaviour, not a risk

    score = max(0.0, min(1.0, score))
    reason = (
        f"{len(hedges)} hedging expression(s) detected" if hedges else "no hedging detected"
    )
    return score, reason
