"""
models.py
---------
Pydantic request/response schemas.  FastAPI uses these to validate every
incoming request and to document the API automatically at /docs.
"""

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


# --------------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------------- #
class ChatRequest(BaseModel):
    query: str = Field(..., description="The user question")
    max_iterations: Optional[int] = Field(
        None, ge=0, le=5, description="Override the maximum number of self-corrections"
    )

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Query cannot be empty.")
        if len(cleaned) > 2000:
            raise ValueError("Query is too long (max 2000 characters).")
        return cleaned


class VerifyRequest(BaseModel):
    """Verify an answer that was produced anywhere (no generation step)."""

    query: str = Field("", description="Optional question the answer belongs to")
    answer: str = Field(..., description="The answer text to verify")

    @field_validator("answer")
    @classmethod
    def answer_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Answer cannot be empty.")
        return value.strip()


class CorrectRequest(BaseModel):
    """Run exactly one correction round on a given answer."""

    query: str = Field(..., description="The original question")
    answer: str = Field(..., description="The answer that should be corrected")

    @field_validator("query", "answer")
    @classmethod
    def not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Field cannot be empty.")
        return value.strip()


# --------------------------------------------------------------------------- #
# Responses
# --------------------------------------------------------------------------- #
class EvidenceItem(BaseModel):
    text: str
    source: str
    retrieval_score: float


class ClaimResult(BaseModel):
    claim: str
    status: str                       # SUPPORTED | CONTRADICTED | INSUFFICIENT
    entailment_score: float
    contradiction_score: float
    neutral_score: float
    verifier: str                     # "nli" or "lexical"
    best_evidence: str
    best_evidence_source: str
    retrieval_score: float
    evidence: List[EvidenceItem] = []


class RiskBreakdown(BaseModel):
    contradiction_risk: float
    unsupported_risk: float
    nli_uncertainty: float
    model_uncertainty: float
    weighted_contribution: dict


class IterationRecord(BaseModel):
    iteration: int
    answer: str
    risk_score: int
    confidence_score: int
    status: str
    supported: int
    contradicted: int
    insufficient: int


class ChatResponse(BaseModel):
    query: str
    original_answer: str
    final_answer: str
    risk_score: int
    confidence_score: int
    status: str                       # LOW_RISK | MEDIUM_RISK | HIGH_RISK
    verification_status: str          # VERIFIED | PARTIALLY_VERIFIED | UNVERIFIED
    iterations: int
    corrected: bool
    claims: List[str]
    evidence: List[ClaimResult]
    risk_breakdown: RiskBreakdown
    history: List[IterationRecord]
    warnings: List[str] = []
    elapsed_seconds: float = 0.0


class VerifyResponse(BaseModel):
    query: str
    answer: str
    risk_score: int
    confidence_score: int
    status: str
    verification_status: str
    claims: List[str]
    evidence: List[ClaimResult]
    risk_breakdown: RiskBreakdown
    warnings: List[str] = []


class CorrectResponse(BaseModel):
    query: str
    original_answer: str
    corrected_answer: str
    risk_before: int
    risk_after: int
    status_after: str
    evidence: List[ClaimResult]
    warnings: List[str] = []


class HealthResponse(BaseModel):
    status: str
    ollama_running: bool
    ollama_model: str
    model_available: bool
    available_models: List[str]
    knowledge_base_documents: int
    knowledge_base_chunks: int
    embedding_backend: str
    nli_backend: str
    message: str
