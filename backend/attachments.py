"""
attachments.py
--------------
Photos and files that the user adds with the "+" button.

Documents (.pdf .txt .md .csv .json)
    The text is read and given to the model as context.  It is ALSO added as
    extra evidence for verification, because the user supplied it as a source.

Images (.png .jpg .webp .gif)
    The default model (llama3.2:1b) cannot see images.  If OLLAMA_VISION_MODEL
    is set (for example "llava" or "llama3.2-vision"), that model describes the
    image and the description is used as context.  Without a vision model the
    image is skipped and the user is told so - nothing is invented.
    An image description is NOT used as verification evidence, because it is
    written by a model and would only be "checking the model with itself".
"""

import base64
import binascii
import io
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from backend import config
from backend.logger import get_logger
from backend.ollama_client import OllamaClient, OllamaError
from backend.rag import Chunk, KnowledgeBase, chunk_text

log = get_logger("cognify.attachments")

DOC_EXTENSIONS = {"pdf", "txt", "md", "csv", "json"}
IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "gif"}
MAX_TEXT_CHARS = 60_000      # per document
MAX_PDF_PAGES = 50


class AttachmentError(ValueError):
    """The user supplied an attachment we cannot accept (shown as HTTP 400)."""


@dataclass
class ParsedAttachments:
    documents: Dict[str, str] = field(default_factory=dict)     # name -> text
    images: List[Tuple[str, str]] = field(default_factory=list)  # (name, base64)
    notes: List[str] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.documents and not self.images


def _extension(name: str) -> str:
    return name.rsplit(".", 1)[-1].lower() if "." in name else ""


def _safe_name(name: str) -> str:
    return re.sub(r"[^\w.\- ()]", "_", name).strip()[:120] or "file"


def _pdf_text(raw: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:                                   # pragma: no cover
        raise AttachmentError("PDF support needs the 'pypdf' package.") from exc
    try:
        reader = PdfReader(io.BytesIO(raw))
        pages = reader.pages[:MAX_PDF_PAGES]
        return "\n\n".join((page.extract_text() or "") for page in pages)
    except Exception as exc:                                     # noqa: BLE001
        raise AttachmentError("This PDF could not be read.") from exc


def parse_attachments(items: List[Dict]) -> ParsedAttachments:
    """Validate and decode the attachments sent by the browser."""
    parsed = ParsedAttachments()
    if len(items) > config.MAX_ATTACHMENTS:
        raise AttachmentError(f"You can attach at most {config.MAX_ATTACHMENTS} files per message.")

    limit = config.MAX_ATTACHMENT_MB * 1024 * 1024
    for item in items:
        name = _safe_name(str(item.get("name", "file")))
        ext = _extension(name)
        try:
            raw = base64.b64decode(item.get("data", ""), validate=False)
        except (binascii.Error, ValueError) as exc:
            raise AttachmentError(f"'{name}' is not valid file data.") from exc
        if not raw:
            raise AttachmentError(f"'{name}' is empty.")
        if len(raw) > limit:
            raise AttachmentError(f"'{name}' is larger than {config.MAX_ATTACHMENT_MB} MB.")

        if ext in IMAGE_EXTENSIONS:
            parsed.images.append((name, base64.b64encode(raw).decode()))
        elif ext in DOC_EXTENSIONS:
            text = _pdf_text(raw) if ext == "pdf" else raw.decode("utf-8", errors="replace")
            text = text.strip()[:MAX_TEXT_CHARS]
            if not text:
                parsed.notes.append(f"No readable text was found in '{name}'.")
                continue
            parsed.documents[name] = text
        else:
            raise AttachmentError(
                f"'{name}' has an unsupported type. Allowed: photos (png, jpg, webp, gif) "
                "and documents (pdf, txt, md, csv, json)."
            )
    return parsed


# --------------------------------------------------------------------------- #
# Images -> text (only with a vision model)
# --------------------------------------------------------------------------- #
def describe_images(client: OllamaClient, images: List[Tuple[str, str]], notes: List[str]) -> Dict[str, str]:
    """Return {image name: description}. Fills `notes` when images are skipped."""
    if not images:
        return {}
    if not config.OLLAMA_VISION_MODEL:
        notes.append(
            "Photo(s) were attached but no vision model is configured, so they were not "
            "read. Set OLLAMA_VISION_MODEL (for example llava) in .env to enable photos."
        )
        return {}

    descriptions: Dict[str, str] = {}
    for name, data in images:
        try:
            text = client.generate(
                prompt="Describe this image factually. Transcribe any visible text exactly. "
                       "Do not guess what you cannot see.",
                model=config.OLLAMA_VISION_MODEL,
                images=[data],
                temperature=0.0,
            )
            descriptions[name] = text.strip()
        except OllamaError as exc:
            notes.append(f"Photo '{name}' could not be read: {exc}")
    return descriptions


# --------------------------------------------------------------------------- #
# Evidence overlay: the user's documents + the normal knowledge base
# --------------------------------------------------------------------------- #
def _document_kb(documents: Dict[str, str]) -> KnowledgeBase:
    kb = KnowledgeBase()
    kb.backend = "keyword"
    kb.document_count = len(documents)
    for name, text in documents.items():
        for i, piece in enumerate(chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)):
            kb.chunks.append(Chunk(text=piece, source=f"attachment: {name}", chunk_id=i))
    return kb


class EvidenceOverlay:
    """Looks like a KnowledgeBase to the detector, but also searches attachments."""

    def __init__(self, base: KnowledgeBase, documents: Dict[str, str]) -> None:
        self._base = base
        self._docs = _document_kb(documents)

    @property
    def backend(self) -> str:
        return self._base.backend if self._base.backend != "empty" else "keyword"

    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        top_k = top_k or config.TOP_K
        merged = self._base.search(query, top_k) + self._docs.search(query, top_k)
        merged.sort(key=lambda item: item["score"], reverse=True)
        return merged[: top_k * 2]

    def __getattr__(self, name):          # stats(), chunks, ... behave like the base KB
        return getattr(self._base, name)


def build_context(documents: Dict[str, str], descriptions: Dict[str, str], query: str) -> str:
    """Text block placed in the prompt. Long documents are cut to the relevant chunks."""
    parts: List[str] = []
    budget = config.ATTACHMENT_PROMPT_CHARS
    total = sum(len(t) for t in documents.values())

    for name, text in documents.items():
        if total <= budget:
            parts.append(f"[Document: {name}]\n{text}")
        else:
            kb = _document_kb({name: text})
            hits = kb.search(query, top_k=3) or [{"text": text[: budget // max(1, len(documents))]}]
            snippet = "\n...\n".join(h["text"] for h in hits)
            parts.append(f"[Document (relevant parts): {name}]\n{snippet[: budget // len(documents)]}")
    for name, text in descriptions.items():
        parts.append(f"[Photo description by a vision model, may contain errors: {name}]\n{text}")
    return "\n\n".join(parts)
