"""
config.py
---------
Single place where every tunable setting of Cognify-AI lives.

Values are read from environment variables (optionally loaded from a .env file
in the project root).  Every value has a sensible default, so the project runs
even if you never create a .env file.

No API keys are used anywhere in this project.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root = folder that contains backend/, frontend/, knowledge_base/ ...
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env if it exists (never fails if the file is missing)
load_dotenv(BASE_DIR / ".env")


# ---------------------------------------------------------------------------
# Small helpers so that a typo in .env does not crash the whole application
# ---------------------------------------------------------------------------
def _get_str(key: str, default: str) -> str:
    value = os.getenv(key)
    return value.strip() if value and value.strip() else default


def _get_int(key: str, default: int) -> int:
    try:
        return int(_get_str(key, str(default)))
    except ValueError:
        return default


def _get_float(key: str, default: float) -> float:
    try:
        return float(_get_str(key, str(default)))
    except ValueError:
        return default


def _get_bool(key: str, default: bool) -> bool:
    return _get_str(key, str(default)).lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# 1. Ollama (local LLM backend)
# ---------------------------------------------------------------------------
OLLAMA_HOST: str = _get_str("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL: str = _get_str("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_TIMEOUT: int = _get_int("OLLAMA_TIMEOUT", 180)          # seconds
OLLAMA_TEMPERATURE: float = _get_float("OLLAMA_TEMPERATURE", 0.2)
OLLAMA_NUM_PREDICT: int = _get_int("OLLAMA_NUM_PREDICT", 400)  # max new tokens

# ---------------------------------------------------------------------------
# 2. Paths
# ---------------------------------------------------------------------------
KNOWLEDGE_BASE_DIR: Path = Path(_get_str("KNOWLEDGE_BASE_DIR", str(BASE_DIR / "knowledge_base")))
DATA_DIR: Path = Path(_get_str("DATA_DIR", str(BASE_DIR / "data")))
INDEX_DIR: Path = DATA_DIR / "vector_index"
LOG_FILE: Path = DATA_DIR / "cognify.log"

# ---------------------------------------------------------------------------
# 3. RAG / retrieval
# ---------------------------------------------------------------------------
EMBEDDING_MODEL: str = _get_str("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE: int = _get_int("CHUNK_SIZE", 700)        # characters per chunk
CHUNK_OVERLAP: int = _get_int("CHUNK_OVERLAP", 120)  # characters of overlap
TOP_K: int = _get_int("TOP_K", 4)                    # evidence chunks per claim
# Cosine similarity below this value is treated as "no usable evidence"
RETRIEVAL_MIN_SCORE: float = _get_float("RETRIEVAL_MIN_SCORE", 0.25)

# ---------------------------------------------------------------------------
# 4. NLI (entailment) verification
# ---------------------------------------------------------------------------
ENABLE_NLI: bool = _get_bool("ENABLE_NLI", True)
NLI_MODEL: str = _get_str("NLI_MODEL", "cross-encoder/nli-deberta-v3-base")
NLI_ENTAILMENT_THRESHOLD: float = _get_float("NLI_ENTAILMENT_THRESHOLD", 0.60)
NLI_CONTRADICTION_THRESHOLD: float = _get_float("NLI_CONTRADICTION_THRESHOLD", 0.50)

# ---------------------------------------------------------------------------
# 5. Claim extraction
# ---------------------------------------------------------------------------
MAX_CLAIMS: int = _get_int("MAX_CLAIMS", 8)
MIN_CLAIM_LENGTH: int = _get_int("MIN_CLAIM_LENGTH", 12)  # characters

# ---------------------------------------------------------------------------
# 6. Hallucination Risk Score weights (must describe a transparent formula)
#    Risk = 40% contradiction + 30% unsupported + 20% NLI uncertainty
#           + 10% model uncertainty
# ---------------------------------------------------------------------------
W_CONTRADICTION: float = _get_float("W_CONTRADICTION", 0.40)
W_UNSUPPORTED: float = _get_float("W_UNSUPPORTED", 0.30)
W_NLI_UNCERTAINTY: float = _get_float("W_NLI_UNCERTAINTY", 0.20)
W_MODEL_UNCERTAINTY: float = _get_float("W_MODEL_UNCERTAINTY", 0.10)

# Risk bands (0-100)
LOW_RISK_MAX: int = _get_int("LOW_RISK_MAX", 30)
MEDIUM_RISK_MAX: int = _get_int("MEDIUM_RISK_MAX", 60)

# ---------------------------------------------------------------------------
# 7. Self-correction loop
# ---------------------------------------------------------------------------
# A correction is triggered when risk_score > RISK_THRESHOLD
RISK_THRESHOLD: int = _get_int("RISK_THRESHOLD", 30)
MAX_ITERATIONS: int = _get_int("MAX_ITERATIONS", 3)

# ---------------------------------------------------------------------------
# 8. Server / logging
# ---------------------------------------------------------------------------
HOST: str = _get_str("HOST", "127.0.0.1")
PORT: int = _get_int("PORT", 8000)
LOG_LEVEL: str = _get_str("LOG_LEVEL", "INFO").upper()

# Make sure the writable folders exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)
KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)


def as_dict() -> dict:
    """Configuration that is safe to expose through GET /api/config."""
    return {
        "ollama_host": OLLAMA_HOST,
        "ollama_model": OLLAMA_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "nli_model": NLI_MODEL if ENABLE_NLI else "lexical-fallback",
        "nli_enabled": ENABLE_NLI,
        "top_k": TOP_K,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "max_claims": MAX_CLAIMS,
        "risk_threshold": RISK_THRESHOLD,
        "max_iterations": MAX_ITERATIONS,
        "risk_weights": {
            "contradiction": W_CONTRADICTION,
            "unsupported": W_UNSUPPORTED,
            "nli_uncertainty": W_NLI_UNCERTAINTY,
            "model_uncertainty": W_MODEL_UNCERTAINTY,
        },
        "risk_bands": {"low_max": LOW_RISK_MAX, "medium_max": MEDIUM_RISK_MAX},
        "knowledge_base_dir": str(KNOWLEDGE_BASE_DIR),
    }
