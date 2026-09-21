"""
verifier.py
-----------
Step 3 of the pipeline: decide whether the retrieved evidence SUPPORTS,
CONTRADICTS, or is INSUFFICIENT for each extracted claim.

Two verifiers are available:

1. NLI (Natural Language Inference) - a local Hugging Face cross-encoder
   (default: cross-encoder/nli-deberta-v3-base).  Input: premise = evidence,
   hypothesis = claim.  Output: entailment / neutral / contradiction scores.

2. Lexical fallback - pure Python word-overlap plus negation and number
   mismatch checks.  It is used when the NLI model is disabled (ENABLE_NLI=false),
   too heavy for the machine, or fails to load.  Much weaker, but it keeps the
   whole framework running on any laptop.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional

from backend import config
from backend.logger import get_logger, shorten
from backend.rag import KnowledgeBase, tokenize

log = get_logger("cognify.verifier")

SUPPORTED = "SUPPORTED"
CONTRADICTED = "CONTRADICTED"
INSUFFICIENT = "INSUFFICIENT"

_NEGATIONS = {"not", "no", "never", "cannot", "isn't", "aren't", "doesn't", "don't", "didn't", "false"}


# --------------------------------------------------------------------------- #
# NLI model wrapper (lazy loaded so the server starts fast)
# --------------------------------------------------------------------------- #
class NLIModel:
    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name = model_name or config.NLI_MODEL
        self.available = False
        self._tokenizer = None
        self._model = None
        self._label_order: List[str] = ["contradiction", "entailment", "neutral"]
        self._loaded = False

    def load(self) -> bool:
        """Load the model once. Returns True when the model is usable."""
        if self._loaded:
            return self.available
        self._loaded = True

        if not config.ENABLE_NLI:
            log.info("NLI disabled in config - using the lexical verifier.")
            return False

        try:
            import torch  # noqa: F401  (imported to fail early with a clear message)
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            log.info("Loading NLI model %s (first run downloads ~400 MB) ...", self.model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self._model.eval()

            # Read the real label order from the model config instead of guessing
            id2label = getattr(self._model.config, "id2label", None)
            if id2label:
                self._label_order = [str(id2label[i]).lower() for i in sorted(id2label)]
            self.available = True
            log.info("NLI model ready. Label order: %s", self._label_order)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "NLI model could not be loaded (%s). Using the lexical fallback verifier. "
                "Set ENABLE_NLI=false in .env to silence this warning.", exc
            )
            self.available = False
        return self.available

    def predict(self, premise: str, hypothesis: str) -> Optional[Dict[str, float]]:
        """Return {'entailment': p, 'neutral': p, 'contradiction': p} or None."""
        if not self.load():
            return None
        try:
            import torch

            inputs = self._tokenizer(
                premise, hypothesis, return_tensors="pt", truncation=True, max_length=512
            )
            with torch.no_grad():
                logits = self._model(**inputs).logits[0]
            probabilities = torch.softmax(logits, dim=-1).tolist()

            scores = {"entailment": 0.0, "neutral": 0.0, "contradiction": 0.0}
            for label, probability in zip(self._label_order, probabilities):
                for key in scores:
                    if key in label:
                        scores[key] = float(probability)
            return scores
        except Exception as exc:  # noqa: BLE001
            log.warning("NLI inference failed (%s) - falling back to lexical scoring.", exc)
            self.available = False
            return None


nli_model = NLIModel()


# --------------------------------------------------------------------------- #
# Lexical fallback verifier
# --------------------------------------------------------------------------- #
def lexical_scores(premise: str, hypothesis: str) -> Dict[str, float]:
    """
    Very small heuristic entailment estimator.

    entailment    ~ how many content words of the claim appear in the evidence
    contradiction ~ negation mismatch or conflicting numbers on a shared topic
    """
    claim_tokens = tokenize(hypothesis)
    evidence_tokens = tokenize(premise)
    if not claim_tokens or not evidence_tokens:
        return {"entailment": 0.0, "neutral": 1.0, "contradiction": 0.0}

    evidence_set = set(evidence_tokens)
    overlap = sum(1 for token in claim_tokens if token in evidence_set) / len(claim_tokens)

    claim_words = set(re.findall(r"[a-z']+", hypothesis.lower()))
    evidence_words = set(re.findall(r"[a-z']+", premise.lower()))
    negation_mismatch = bool(claim_words & _NEGATIONS) != bool(evidence_words & _NEGATIONS)

    claim_numbers = set(re.findall(r"\d+(?:\.\d+)?", hypothesis))
    evidence_numbers = set(re.findall(r"\d+(?:\.\d+)?", premise))
    number_conflict = bool(claim_numbers) and bool(evidence_numbers) and not (claim_numbers & evidence_numbers)

    contradiction = 0.0
    if overlap > 0.35 and negation_mismatch:
        contradiction = 0.55
    if overlap > 0.35 and number_conflict:
        contradiction = max(contradiction, 0.60)

    entailment = max(0.0, min(1.0, overlap * (0.4 if contradiction else 1.0)))
    neutral = max(0.0, 1.0 - entailment - contradiction)
    total = entailment + neutral + contradiction or 1.0
    return {
        "entailment": entailment / total,
        "neutral": neutral / total,
        "contradiction": contradiction / total,
    }


# --------------------------------------------------------------------------- #
# Claim verification
# --------------------------------------------------------------------------- #
def verify_claim(claim: str, kb: KnowledgeBase, top_k: Optional[int] = None) -> Dict:
    """Retrieve evidence for one claim and classify it."""
    evidence = kb.search(claim, top_k=top_k or config.TOP_K)
    usable = [e for e in evidence if e["score"] >= config.RETRIEVAL_MIN_SCORE]

    base = {
        "claim": claim,
        "status": INSUFFICIENT,
        "entailment_score": 0.0,
        "contradiction_score": 0.0,
        "neutral_score": 1.0,
        "verifier": "nli" if nli_model.available else "lexical",
        "best_evidence": "",
        "best_evidence_source": "",
        "retrieval_score": round(evidence[0]["score"], 4) if evidence else 0.0,
        "evidence": [
            {"text": e["text"], "source": e["source"], "retrieval_score": e["score"]}
            for e in evidence[:3]
        ],
    }

    if not usable:
        log.info("Claim INSUFFICIENT (no relevant evidence): %s", shorten(claim, 120))
        return base

    best_entailment = {"score": -1.0, "item": None, "scores": None}
    best_contradiction = {"score": -1.0, "item": None, "scores": None}
    verifier_used = "lexical"

    for item in usable:
        scores = nli_model.predict(item["text"], claim)
        if scores is None:
            scores = lexical_scores(item["text"], claim)
        else:
            verifier_used = "nli"

        if scores["entailment"] > best_entailment["score"]:
            best_entailment = {"score": scores["entailment"], "item": item, "scores": scores}
        if scores["contradiction"] > best_contradiction["score"]:
            best_contradiction = {"score": scores["contradiction"], "item": item, "scores": scores}

    entailment = best_entailment["score"]
    contradiction = best_contradiction["score"]

    if contradiction >= config.NLI_CONTRADICTION_THRESHOLD and contradiction > entailment:
        chosen, status = best_contradiction, CONTRADICTED
    elif entailment >= config.NLI_ENTAILMENT_THRESHOLD:
        chosen, status = best_entailment, SUPPORTED
    else:
        chosen, status = best_entailment, INSUFFICIENT

    scores = chosen["scores"] or {"entailment": 0.0, "neutral": 1.0, "contradiction": 0.0}
    base.update(
        {
            "status": status,
            "entailment_score": round(scores["entailment"], 4),
            "contradiction_score": round(scores["contradiction"], 4),
            "neutral_score": round(scores["neutral"], 4),
            "verifier": verifier_used,
            "best_evidence": chosen["item"]["text"] if chosen["item"] else "",
            "best_evidence_source": chosen["item"]["source"] if chosen["item"] else "",
            "retrieval_score": chosen["item"]["score"] if chosen["item"] else 0.0,
        }
    )
    log.info("Claim %s (%.2f entail / %.2f contra): %s", status, entailment, contradiction, shorten(claim, 120))
    return base


def verify_claims(claims: List[str], kb: KnowledgeBase) -> List[Dict]:
    return [verify_claim(claim, kb) for claim in claims]


def nli_uncertainty(results: List[Dict]) -> float:
    """
    Average uncertainty of the verifier itself.

    For each claim the verifier outputs three probabilities.  If the largest one
    is close to 1.0 the verifier is confident; if all three are similar it is
    unsure.  uncertainty = 1 - max(probability).
    """
    if not results:
        return 0.0
    total = 0.0
    for result in results:
        top = max(
            result["entailment_score"], result["contradiction_score"], result["neutral_score"]
        )
        total += 1.0 - top
    return round(total / len(results), 4)
