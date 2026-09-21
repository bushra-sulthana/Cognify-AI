"""
Shared pytest setup.

The tests must run WITHOUT Ollama, WITHOUT downloading the NLI model and
WITHOUT building embeddings, so heavy components are switched off here.
"""

import sys
from pathlib import Path

import pytest

# Allow "import backend..." when pytest is started from the project root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.rag import Chunk, KnowledgeBase  # noqa: E402
from backend.verifier import nli_model  # noqa: E402

# Pretend the NLI model was already checked and is unavailable -> lexical verifier
nli_model._loaded = True
nli_model.available = False


@pytest.fixture
def fake_kb() -> KnowledgeBase:
    """Small in-memory knowledge base that uses keyword retrieval only."""
    kb = KnowledgeBase()
    kb.backend = "keyword"
    kb.document_count = 1
    kb.chunks = [
        Chunk(
            text="Paris is the capital and largest city of France.",
            source="geography.txt", chunk_id=0,
        ),
        Chunk(
            text="Canberra is the capital city of Australia. Sydney is the largest city "
                 "of Australia, but Sydney is not the capital of Australia.",
            source="geography.txt", chunk_id=1,
        ),
        Chunk(
            text="At standard atmospheric pressure, water boils at 100 degrees Celsius.",
            source="science.txt", chunk_id=2,
        ),
    ]
    return kb
