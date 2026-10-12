"""
main.py
-------
FastAPI application: REST API + static frontend for Cognify-AI.

Endpoints
    GET  /api/health   - is Ollama running? is the model installed? KB stats
    GET  /api/config   - the active configuration (weights, thresholds, models)
    POST /api/chat     - full pipeline: generate -> detect -> correct -> verify
    POST /api/verify   - verify an answer you already have (no generation)
    POST /api/correct  - run exactly one correction round on an answer

Start with:
    uvicorn backend.main:app --reload --port 8000
Then open http://localhost:8000
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend import config
from backend.corrector import correct_answer
from backend.detector import detect
from backend.logger import get_logger
from backend.models import (
    ChatRequest,
    ChatResponse,
    CorrectRequest,
    CorrectResponse,
    HealthResponse,
    VerifyRequest,
    VerifyResponse,
)
from backend.ollama_client import OllamaClient, OllamaError
from backend.pipeline import run_pipeline
from backend.rag import knowledge_base
from backend.verifier import nli_model

log = get_logger("cognify.api")

FRONTEND_DIR = config.BASE_DIR / "frontend"
client = OllamaClient()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the knowledge base once, when the server starts."""
    log.info("Starting Cognify-AI ...")
    try:
        knowledge_base.build()
    except Exception as exc:  # noqa: BLE001 - the API must start even if the KB fails
        log.error("Knowledge base could not be built: %s", exc)
    status = client.health()
    log.info("Ollama status: %s", status["message"])
    yield
    log.info("Cognify-AI stopped.")


app = FastAPI(
    title="Cognify-AI",
    description=(
        "Self-Correcting LLM Framework that detects and reduces hallucination risk "
        "using a local Ollama model, a local RAG knowledge base and NLI verification."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# The frontend can also be opened as a file:// page during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Health and configuration
# --------------------------------------------------------------------------- #
@app.get("/api/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    status = client.health()
    stats = knowledge_base.stats()
    overall = "ok" if status["running"] and status["model_available"] else "degraded"
    return HealthResponse(
        status=overall,
        ollama_running=status["running"],
        ollama_model=client.model,
        model_available=status["model_available"],
        available_models=status["models"],
        knowledge_base_documents=stats["documents"],
        knowledge_base_chunks=stats["chunks"],
        embedding_backend=stats["backend"],
        nli_backend="nli" if nli_model.available else ("nli (not loaded yet)" if config.ENABLE_NLI else "lexical"),
        message=status["message"],
    )


@app.get("/api/config", tags=["system"])
def get_config() -> dict:
    """Active configuration. No secrets or API keys exist in this project."""
    return config.as_dict()


@app.post("/api/reload-kb", tags=["system"])
def reload_knowledge_base() -> dict:
    """Rebuild the vector index after you add documents to knowledge_base/."""
    try:
        knowledge_base.build(force=True)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Rebuild failed: {exc}") from exc
    return knowledge_base.stats()


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #
@app.post("/api/chat", response_model=ChatResponse, tags=["pipeline"])
def chat(request: ChatRequest) -> ChatResponse:
    """Generate an answer, verify it, and self-correct it when the risk is high."""
    try:
        result = run_pipeline(
            query=request.query,
            kb=knowledge_base,
            client=client,
            max_iterations=request.max_iterations,
        )
    except OllamaError as exc:
        # 503: the local LLM service is unavailable - the message explains the fix
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("Unexpected pipeline error")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}") from exc
    return ChatResponse(**result)


@app.post("/api/verify", response_model=VerifyResponse, tags=["pipeline"])
def verify(request: VerifyRequest) -> VerifyResponse:
    """Verify an existing answer against the knowledge base (no generation)."""
    try:
        report = detect(request.answer, knowledge_base, client, query=request.query)
    except Exception as exc:  # noqa: BLE001
        log.exception("Verification failed")
        raise HTTPException(status_code=500, detail=f"Verification error: {exc}") from exc

    return VerifyResponse(
        query=request.query,
        answer=request.answer,
        risk_score=report["risk_score"],
        confidence_score=report["confidence_score"],
        status=report["status"],
        verification_status=report["verification_status"],
        claims=report["claims"],
        evidence=report["results"],
        risk_breakdown=report["breakdown"],
        warnings=report["warnings"],
    )


@app.post("/api/correct", response_model=CorrectResponse, tags=["pipeline"])
def correct(request: CorrectRequest) -> CorrectResponse:
    """Run one evidence-grounded correction round on a given answer."""
    try:
        before = detect(request.answer, knowledge_base, client, query=request.query)
        corrected = correct_answer(client, request.query, request.answer, before["results"])
        after = detect(corrected, knowledge_base, client, query=request.query)
    except OllamaError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("Correction failed")
        raise HTTPException(status_code=500, detail=f"Correction error: {exc}") from exc

    return CorrectResponse(
        query=request.query,
        original_answer=request.answer,
        corrected_answer=corrected,
        risk_before=before["risk_score"],
        risk_after=after["risk_score"],
        status_after=after["status"],
        evidence=after["results"],
        warnings=after["warnings"],
    )


# --------------------------------------------------------------------------- #
# Frontend (served by the same server, so there are no CORS problems)
# --------------------------------------------------------------------------- #
@app.get("/", include_in_schema=False)
def index():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return JSONResponse(
        {"message": "Cognify-AI API is running. Open /docs for the API."}
    )


if FRONTEND_DIR.exists():

    app.mount(
        "/frontend",
        StaticFiles(directory=str(FRONTEND_DIR)),
        name="frontend",
    )

