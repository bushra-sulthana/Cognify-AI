"""
rag.py
------
Local trusted knowledge base (the "R" and the store of the RAG pipeline).

Pipeline inside this file:
  1. load  .txt / .md / .pdf files from knowledge_base/
  2. split them into overlapping character chunks
  3. embed the chunks locally with a Sentence-Transformers model
  4. store the vectors in FAISS (with a pure-NumPy fallback)
  5. retrieve the most relevant chunks for a claim or question

Everything runs offline.  No OpenAI embeddings, no API key.
If the embedding model cannot be loaded (no internet on first run, low memory,
corrupted cache), the class automatically falls back to a keyword-overlap
retriever so the application still works.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from backend import config
from backend.logger import get_logger

log = get_logger("cognify.rag")

SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown", ".pdf"}

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "of", "in", "on", "at", "to", "for",
    "and", "or", "it", "its", "that", "this", "these", "those", "as", "by", "with",
    "from", "be", "been", "has", "have", "had", "which", "who", "what", "when",
    "where", "how", "than", "then", "there", "their", "also", "can", "will",
}


@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: int


def tokenize(text: str) -> List[str]:
    """Lowercase content words - used by the keyword fallback retriever."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 1]


# --------------------------------------------------------------------------- #
# Document loading
# --------------------------------------------------------------------------- #
def read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        log.warning("pypdf is not installed - skipping PDF %s", path.name)
        return ""
    try:
        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception as exc:  # noqa: BLE001 - a malformed PDF must not stop startup
        log.warning("Could not read PDF %s: %s", path.name, exc)
        return ""


def read_text(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except (UnicodeDecodeError, OSError):
            continue
    log.warning("Could not read %s with any known encoding.", path.name)
    return ""


def load_documents(folder: Path) -> Dict[str, str]:
    """Return {filename: text} for every readable document in the folder."""
    documents: Dict[str, str] = {}
    if not folder.exists():
        log.warning("Knowledge base folder %s does not exist.", folder)
        return documents

    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        text = read_pdf(path) if path.suffix.lower() == ".pdf" else read_text(path)
        if text and text.strip():
            documents[path.name] = text
        else:
            log.warning("Document %s is empty or unreadable - skipped.", path.name)
    return documents


def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    """Split on paragraph boundaries where possible, then pack into chunks."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(current) + len(paragraph) + 1 <= size:
            current = f"{current}\n{paragraph}".strip()
        else:
            if current:
                chunks.append(current)
            # A single very long paragraph is hard-split with overlap
            while len(paragraph) > size:
                chunks.append(paragraph[:size])
                paragraph = paragraph[max(0, size - overlap):]
            current = paragraph
    if current:
        chunks.append(current)
    return [c for c in chunks if c.strip()]


# --------------------------------------------------------------------------- #
# Knowledge base
# --------------------------------------------------------------------------- #
class KnowledgeBase:
    """Loads documents, builds the vector index and answers retrieval queries."""

    def __init__(self, folder: Optional[Path] = None) -> None:
        self.folder = Path(folder or config.KNOWLEDGE_BASE_DIR)
        self.chunks: List[Chunk] = []
        self.document_count = 0
        self.backend = "not-initialised"   # faiss | numpy | keyword | empty
        self._embedder = None
        self._matrix = None                # numpy array of normalised embeddings
        self._faiss_index = None

    # -------------------------- build ---------------------------------- #
    def build(self, force: bool = False) -> None:
        """Load documents and create embeddings. Safe to call more than once."""
        documents = load_documents(self.folder)
        self.document_count = len(documents)
        self.chunks = []

        for name, text in documents.items():
            for i, piece in enumerate(chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)):
                self.chunks.append(Chunk(text=piece, source=name, chunk_id=i))

        if not self.chunks:
            self.backend = "empty"
            log.warning(
                "Knowledge base is EMPTY. Add .txt/.md/.pdf files to %s and restart. "
                "Verification will report INSUFFICIENT evidence for every claim.",
                self.folder,
            )
            return

        log.info("Loaded %d document(s) -> %d chunk(s).", self.document_count, len(self.chunks))
        self._build_embeddings(force=force)

    def _fingerprint(self) -> str:
        """Hash of the chunk texts + model name, so we know when to re-embed."""
        hasher = hashlib.sha256()
        hasher.update(config.EMBEDDING_MODEL.encode("utf-8"))
        for chunk in self.chunks:
            hasher.update(chunk.source.encode("utf-8"))
            hasher.update(chunk.text.encode("utf-8"))
        return hasher.hexdigest()

    def _build_embeddings(self, force: bool = False) -> None:
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # noqa: BLE001
            log.warning("sentence-transformers unavailable (%s) - keyword retrieval only.", exc)
            self.backend = "keyword"
            return

        try:
            log.info("Loading embedding model %s ...", config.EMBEDDING_MODEL)
            self._embedder = SentenceTransformer(config.EMBEDDING_MODEL)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "Embedding model could not be loaded (%s). "
                "Falling back to keyword retrieval - results will be less accurate.", exc
            )
            self.backend = "keyword"
            return

        cache_vectors = config.INDEX_DIR / "embeddings.npy"
        cache_meta = config.INDEX_DIR / "index_meta.json"
        fingerprint = self._fingerprint()
        vectors = None

        if not force and cache_vectors.exists() and cache_meta.exists():
            try:
                meta = json.loads(cache_meta.read_text(encoding="utf-8"))
                if meta.get("fingerprint") == fingerprint:
                    vectors = np.load(cache_vectors)
                    log.info("Re-using cached embeddings (%s).", cache_vectors.name)
            except Exception as exc:  # noqa: BLE001
                log.warning("Could not read embedding cache: %s", exc)

        if vectors is None:
            log.info("Embedding %d chunk(s) locally ...", len(self.chunks))
            vectors = self._embedder.encode(
                [c.text for c in self.chunks],
                batch_size=16,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,
            ).astype("float32")
            try:
                np.save(cache_vectors, vectors)
                cache_meta.write_text(
                    json.dumps({"fingerprint": fingerprint, "chunks": len(self.chunks)}, indent=2),
                    encoding="utf-8",
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("Could not cache embeddings: %s", exc)

        self._matrix = vectors

        # FAISS gives fast similarity search; NumPy is the fallback.
        try:
            import faiss

            index = faiss.IndexFlatIP(vectors.shape[1])  # inner product == cosine (normalised)
            index.add(vectors)
            self._faiss_index = index
            self.backend = "faiss"
        except Exception as exc:  # noqa: BLE001
            log.warning("FAISS unavailable (%s) - using NumPy similarity search.", exc)
            self.backend = "numpy"

        log.info("Vector store ready (backend=%s).", self.backend)

    # -------------------------- search --------------------------------- #
    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """Return the most relevant chunks as [{text, source, score}, ...]."""
        top_k = top_k or config.TOP_K
        if not self.chunks or not query.strip():
            return []

        if self.backend in {"faiss", "numpy"} and self._embedder is not None:
            try:
                return self._vector_search(query, top_k)
            except Exception as exc:  # noqa: BLE001
                log.warning("Vector search failed (%s) - falling back to keywords.", exc)

        return self._keyword_search(query, top_k)

    def _vector_search(self, query: str, top_k: int) -> List[Dict]:
        import numpy as np

        vector = self._embedder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")

        if self.backend == "faiss" and self._faiss_index is not None:
            scores, indices = self._faiss_index.search(vector, min(top_k, len(self.chunks)))
            pairs = list(zip(indices[0].tolist(), scores[0].tolist()))
        else:
            sims = (self._matrix @ vector[0])
            order = np.argsort(-sims)[:top_k]
            pairs = [(int(i), float(sims[i])) for i in order]

        results = []
        for idx, score in pairs:
            if idx < 0:
                continue
            chunk = self.chunks[idx]
            results.append({"text": chunk.text, "source": chunk.source, "score": round(float(score), 4)})
        return results

    def _keyword_search(self, query: str, top_k: int) -> List[Dict]:
        """TF-style keyword overlap - crude, but never fails and needs no model."""
        query_tokens = set(tokenize(query))
        if not query_tokens:
            return []

        scored = []
        for chunk in self.chunks:
            chunk_tokens = set(tokenize(chunk.text))
            if not chunk_tokens:
                continue
            overlap = len(query_tokens & chunk_tokens)
            if overlap == 0:
                continue
            # Cosine-like normalisation so scores stay in a comparable 0..1 range
            score = overlap / math.sqrt(len(query_tokens) * len(chunk_tokens))
            scored.append({"text": chunk.text, "source": chunk.source, "score": round(score, 4)})

        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]

    # -------------------------- info ----------------------------------- #
    def stats(self) -> Dict:
        return {
            "documents": self.document_count,
            "chunks": len(self.chunks),
            "backend": self.backend,
            "folder": str(self.folder),
        }


# A single shared instance used by the API
knowledge_base = KnowledgeBase()
