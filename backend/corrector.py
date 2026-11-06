"""
corrector.py
------------
Step 4 of the pipeline: rewrite an answer using only the retrieved evidence.

The correction prompt contains three things:
  * the answer that failed verification,
  * the claims that were contradicted or unsupported,
  * the trusted evidence retrieved from the local knowledge base.

The LLM is explicitly told to say "could not be verified" instead of inventing
new facts, which is what turns a hallucination into an honest answer.
"""

import re
from typing import Dict, List

from backend.levels import level_instruction, level_length
from backend.logger import get_logger, shorten
from backend.ollama_client import OllamaClient
from backend.verifier import CONTRADICTED, SUPPORTED

log = get_logger("cognify.corrector")

CORRECTION_SYSTEM_PROMPT = (
    "You are a strict fact-grounded editor. You rewrite answers so that every "
    "sentence is supported by the provided evidence. You never add new facts."
)

CORRECTION_TEMPLATE = """You generated the following answer to the question: "{query}"

ANSWER:
{answer}

The following claims could NOT be verified, or were CONTRADICTED by trusted sources:
{problem_claims}

TRUSTED EVIDENCE (from the verified knowledge base):
{evidence}

Rewrite the answer using ONLY information supported by the evidence above.
Do not invent facts, numbers, names or dates.
If a contradicted claim is corrected by the evidence, state the correct fact.
If the evidence is insufficient for part of the question, explicitly write that
this information could not be verified against the knowledge base.
{style}

Write the corrected answer between <answer> and </answer> tags.
Write nothing outside the tags: no preamble, no reasoning, no explanation."""


def _format_problem_claims(results: List[Dict]) -> str:
    lines = []
    for result in results:
        if result["status"] == SUPPORTED:
            continue
        marker = "CONTRADICTED" if result["status"] == CONTRADICTED else "UNSUPPORTED"
        lines.append(f"- [{marker}] {result['claim']}")
    return "\n".join(lines) if lines else "- (no specific claim could be verified)"


def _format_evidence(results: List[Dict], limit: int = 6) -> str:
    """Collect unique evidence snippets across all claims."""
    seen, lines = set(), []
    for result in results:
        for item in result.get("evidence", []):
            text = " ".join(item["text"].split())
            if text in seen:
                continue
            seen.add(text)
            lines.append(f"- ({item['source']}) {text}")
            if len(lines) >= limit:
                return "\n".join(lines)
    return "\n".join(lines) if lines else "- (no relevant evidence was found in the knowledge base)"


def build_correction_prompt(query: str, answer: str, results: List[Dict], level: int = None) -> str:
    style = (
        f"{level_instruction(level)} Keep the same style and length ({level_length(level)})."
        if level is not None else "Keep it short (2-5 sentences)."
    )
    return CORRECTION_TEMPLATE.format(
        style=style,
        query=query or "(not provided)",
        answer=answer,
        problem_claims=_format_problem_claims(results),
        evidence=_format_evidence(results),
    )


class CorrectionRejected(Exception):
    """The model did not return a usable rewrite (the previous answer is kept)."""


_REASONING_START = re.compile(
    r"^\s*(we are given|we have|let'?s|let me|okay|ok,|hmm|first,|so,|wait|the user|"
    r"i need|i should|i'll|the original answer|the problem|the task|looking at)",
    re.I,
)
_REASONING_PHRASES = (
    "the problem says", "rewrite the answer", "the user ", "the claims", "trusted evidence",
    "the evidence provided", "evidence base", "original answer", "we are given", "let's check",
)


def looks_like_reasoning(text: str, original: str) -> bool:
    """True when the 'answer' is the model thinking aloud instead of an answer."""
    lowered = text.lower()
    if _REASONING_START.match(text):
        return True
    if sum(phrase in lowered for phrase in _REASONING_PHRASES) >= 2:
        return True
    return len(text) > max(3 * len(original), 900)


def extract_answer(raw: str) -> str:
    """Take the text inside <answer>...</answer>; refuse anything else."""
    found = re.findall(r"<answer>(.*?)</answer>", raw, flags=re.S | re.I)
    if found:
        return found[-1].strip()
    match = re.search(r"<answer>(.*)$", raw, flags=re.S | re.I)   # closing tag missing
    if match:
        return match.group(1).strip()
    raise CorrectionRejected("the model did not use the <answer> tags")


def correct_answer(client: OllamaClient, query: str, answer: str, results: List[Dict], level: int = None) -> str:
    """Ask the local LLM for an evidence-grounded rewrite."""
    prompt = build_correction_prompt(query, answer, results, level)
    corrected = client.generate(prompt=prompt, system=CORRECTION_SYSTEM_PROMPT, temperature=0.0)

    cleaned = extract_answer(corrected).strip().strip('"')
    if not cleaned or looks_like_reasoning(cleaned, answer):
        raise CorrectionRejected("the model returned its reasoning instead of an answer")
    # Small models sometimes wrap the rewrite in quotes or a label
    for prefix in ("Corrected answer:", "CORRECTED ANSWER:", "Answer:"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
    log.info("Corrected response: %s", shorten(cleaned))
    return cleaned
