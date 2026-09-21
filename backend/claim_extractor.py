"""
claim_extractor.py
------------------
Step 2 of the pipeline: split an answer into individual factual claims.

Primary strategy : ask the LLM to return JSON  {"claims": ["...", "..."]}
Fallback strategy: rule-based sentence splitting (used when the LLM returns
                   broken JSON, when Ollama fails, or when the JSON is empty).

An LLM very often wraps JSON in prose or markdown fences, so the parser here is
deliberately defensive.
"""

import json
import re
from typing import List

from backend import config
from backend.logger import get_logger
from backend.ollama_client import OllamaClient, OllamaError

log = get_logger("cognify.claims")

CLAIM_SYSTEM_PROMPT = (
    "You extract factual claims from text. You always reply with raw JSON only - "
    "no markdown, no explanation."
)

CLAIM_PROMPT_TEMPLATE = """Split the TEXT below into separate, self-contained factual claims.

Rules:
- Each claim must be a single complete sentence that can be checked as true or false.
- Replace pronouns with the real subject so each claim stands alone.
- Ignore opinions, questions, greetings and instructions.
- Maximum {max_claims} claims.

Return exactly this JSON structure and nothing else:
{{"claims": ["claim one", "claim two"]}}

TEXT:
\"\"\"{text}\"\"\"
"""

# Sentences that are not factual claims (used by the fallback splitter)
_NON_CLAIM_STARTS = (
    "however", "in summary", "let me know", "would you like", "i hope",
    "feel free", "in conclusion", "sure", "of course", "as an ai",
)


def _extract_json_block(raw: str) -> str:
    """Return the first balanced {...} block found in the LLM output."""
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    start = cleaned.find("{")
    if start == -1:
        return ""
    depth = 0
    for index in range(start, len(cleaned)):
        if cleaned[index] == "{":
            depth += 1
        elif cleaned[index] == "}":
            depth -= 1
            if depth == 0:
                return cleaned[start : index + 1]
    return ""


def parse_claims_json(raw: str) -> List[str]:
    """Safely turn a (possibly malformed) LLM reply into a list of claims."""
    block = _extract_json_block(raw)
    if block:
        try:
            data = json.loads(block)
            claims = data.get("claims", []) if isinstance(data, dict) else []
            if isinstance(claims, list):
                return [str(c).strip() for c in claims if str(c).strip()]
        except json.JSONDecodeError:
            log.warning("LLM returned invalid JSON for claim extraction - using fallback.")

    # Second chance: a bare JSON array somewhere in the text
    match = re.search(r"\[(?:[^\[\]]|\n)*\]", raw)
    if match:
        try:
            items = json.loads(match.group(0))
            if isinstance(items, list):
                return [str(c).strip() for c in items if str(c).strip()]
        except json.JSONDecodeError:
            pass
    return []


def split_sentences(text: str) -> List[str]:
    """Rule-based fallback claim extraction."""
    # Protect common abbreviations from the sentence splitter
    protected = re.sub(r"\b(Mr|Mrs|Dr|Prof|St|etc|e\.g|i\.e)\.", r"\1<DOT>", text)
    parts = re.split(r"(?<=[.!?])\s+", protected)
    sentences = []
    for part in parts:
        sentence = part.replace("<DOT>", ".").strip()
        if not sentence:
            continue
        lowered = sentence.lower()
        if sentence.endswith("?"):
            continue
        if lowered.startswith(_NON_CLAIM_STARTS):
            continue
        sentences.append(sentence)
    return sentences


def _clean(claims: List[str]) -> List[str]:
    """Remove duplicates, bullets and claims that are too short."""
    seen, cleaned = set(), []
    for claim in claims:
        claim = re.sub(r"^[\-\*\d\.\)\s]+", "", claim).strip().strip('"')
        if len(claim) < config.MIN_CLAIM_LENGTH:
            continue
        key = claim.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(claim)
        if len(cleaned) >= config.MAX_CLAIMS:
            break
    return cleaned


def extract_claims(client: OllamaClient, answer: str) -> List[str]:
    """Public entry point: LLM extraction with an automatic rule-based fallback."""
    claims: List[str] = []
    try:
        raw = client.generate(
            prompt=CLAIM_PROMPT_TEMPLATE.format(max_claims=config.MAX_CLAIMS, text=answer),
            system=CLAIM_SYSTEM_PROMPT,
            temperature=0.0,
        )
        claims = parse_claims_json(raw)
    except OllamaError as exc:
        log.warning("Claim extraction via LLM failed (%s) - using sentence fallback.", exc)

    if not claims:
        claims = split_sentences(answer)

    claims = _clean(claims)
    log.info("Extracted %d claim(s): %s", len(claims), claims)
    return claims
