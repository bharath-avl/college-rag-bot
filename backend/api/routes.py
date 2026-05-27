"""
FastAPI route definitions for the College RAG Chatbot.

Endpoints:
    POST   /upload              — Ingest a PDF through the full RAG pipeline.
    POST   /chat                — Stream an answer via the LangGraph agent.
    GET    /documents           — List all indexed documents from SQLite.
    DELETE /documents/{filename} — Remove a document from ChromaDB and SQLite.
    GET    /health              — Service health check.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import aiosqlite
from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.db.models import CREATE_DOCUMENTS_TABLE
from backend.graph.graph import run_query
from backend.rag.chunker import Chunker
from backend.rag.ingestor import PDFIngestor
from backend.rag.vectorstore import VectorStore
from backend.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

# ── defaults ────────────────────────────────────────────────────────────────

_DEFAULT_UPLOAD_DIR: str = os.path.join("data", "uploads")
_DEFAULT_DB_PATH: str = os.path.join("data", "college_rag.db")


# ── request / response schemas ──────────────────────────────────────────────


class ChatRequest(BaseModel):
    """Body schema for the ``POST /chat`` endpoint."""

    query: str = Field(..., min_length=1, description="User's question.")
    session_id: str = Field(
        ..., min_length=1, description="Unique session/thread identifier."
    )


# ═════════════════════════════════════════════════════════════════════════════
# POST /upload
# ═════════════════════════════════════════════════════════════════════════════


@router.post("/upload")
async def upload_pdf(file: UploadFile) -> dict:
    """
    Accept a multipart PDF upload and run the full RAG ingest pipeline.

    1. Save the file to ``PDF_UPLOAD_DIR``.
    2. Extract text via ``PDFIngestor``.
    3. Split into chunks via ``Chunker``.
    4. Embed and store in ChromaDB via ``VectorStore``.

    Returns:
        A dict with ``status``, ``chunks_indexed``, and ``filename``.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400, detail="Only PDF files are accepted."
        )

    upload_dir = os.getenv("PDF_UPLOAD_DIR", _DEFAULT_UPLOAD_DIR)
    Path(upload_dir).mkdir(parents=True, exist_ok=True)

    file_path = os.path.join(upload_dir, file.filename)

    # ── 1. Save uploaded file ───────────────────────────────────────────
    logger.info("Saving uploaded file to %s", file_path)
    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)

    # ── 2. Ingest: extract text + save metadata to SQLite ───────────────
    logger.info("Running PDFIngestor on %s", file_path)
    ingestor = PDFIngestor()
    doc_result = await ingestor.ingest(file_path)

    # ── 3. Chunk ────────────────────────────────────────────────────────
    logger.info("Chunking document: %s", file.filename)
    chunker = Chunker()
    chunks = chunker.split(doc_result)

    # ── 4. Embed & store ────────────────────────────────────────────────
    logger.info("Indexing %d chunks into ChromaDB", len(chunks))
    vs = VectorStore()
    chunks_indexed = await vs.add_documents(chunks)

    logger.info(
        "Upload complete — filename=%s, chunks_indexed=%d",
        file.filename,
        chunks_indexed,
    )
    return {
        "status": "ok",
        "chunks_indexed": chunks_indexed,
        "filename": file.filename,
    }


# ═════════════════════════════════════════════════════════════════════════════
# POST /chat
# ═════════════════════════════════════════════════════════════════════════════


@router.post("/chat")
async def chat(body: ChatRequest) -> StreamingResponse:
    """
    Stream an answer from the LangGraph agent as Server-Sent Events.

    Each SSE event is a JSON line::

        data: {"token": "...", "done": false, "sources": []}

    The final event has ``done=true`` and may include citation sources.
    """
    logger.info("Chat request — session=%s, query=%r", body.session_id, body.query)

    async def _event_stream():
        """Yield SSE-formatted events from the graph."""
        sources: list[dict] = []
        try:
            async for token in run_query(body.query, body.session_id):
                event = json.dumps(
                    {"token": token, "done": False, "sources": []},
                    ensure_ascii=False,
                )
                yield f"data: {event}\n\n"

            # Final event — signal completion
            done_event = json.dumps(
                {"token": "", "done": True, "sources": sources},
                ensure_ascii=False,
            )
            yield f"data: {done_event}\n\n"

        except Exception as exc:
            logger.exception("Error during chat streaming")
            error_event = json.dumps(
                {"token": "", "done": True, "sources": [], "error": str(exc)},
                ensure_ascii=False,
            )
            yield f"data: {error_event}\n\n"

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ═════════════════════════════════════════════════════════════════════════════
# GET /documents
# ═════════════════════════════════════════════════════════════════════════════


@router.get("/documents")
async def list_documents() -> list[dict]:
    """
    Return all indexed documents from the SQLite ``documents`` table.

    Each document dict contains ``id``, ``filename``, ``subject``,
    ``pages``, and ``uploaded_at``.
    """
    db_path = os.getenv("DB_PATH", _DEFAULT_DB_PATH)

    logger.info("Listing documents from %s", db_path)

    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        await db.execute(CREATE_DOCUMENTS_TABLE)
        cursor = await db.execute(
            "SELECT id, filename, subject, pages, uploaded_at FROM documents"
        )
        rows = await cursor.fetchall()

    documents = [
        {
            "id": row["id"],
            "filename": row["filename"],
            "subject": row["subject"],
            "pages": row["pages"],
            "uploaded_at": row["uploaded_at"],
        }
        for row in rows
    ]

    logger.info("Found %d documents", len(documents))
    return documents


# ═════════════════════════════════════════════════════════════════════════════
# DELETE /documents/{filename}
# ═════════════════════════════════════════════════════════════════════════════


@router.delete("/documents/{filename}")
async def delete_document(filename: str) -> dict:
    """
    Delete a document from both ChromaDB and SQLite.

    Args:
        filename: The original PDF filename (e.g. ``math_101.pdf``).

    Returns:
        ``{"status": "deleted", "filename": str}``

    Raises:
        HTTPException 404: If the document is not found in ChromaDB.
    """
    logger.info("Delete request for filename=%r", filename)

    # ── Remove chunks from ChromaDB ─────────────────────────────────────
    vs = VectorStore()
    deleted = await vs.delete_by_source(filename)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"No indexed chunks found for '{filename}'.",
        )

    # ── Remove metadata row from SQLite ─────────────────────────────────
    db_path = os.getenv("DB_PATH", _DEFAULT_DB_PATH)
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "DELETE FROM documents WHERE filename = ?", (filename,)
        )
        await db.commit()

    # ── Remove the uploaded file from disk (best-effort) ────────────────
    upload_dir = os.getenv("PDF_UPLOAD_DIR", _DEFAULT_UPLOAD_DIR)
    file_path = Path(upload_dir) / filename
    if file_path.exists():
        file_path.unlink()
        logger.info("Deleted file from disk: %s", file_path)

    logger.info("Document deleted: %s", filename)
    return {"status": "deleted", "filename": filename}


# ═════════════════════════════════════════════════════════════════════════════
# GET /health
# ═════════════════════════════════════════════════════════════════════════════


@router.get("/health")
async def health_check() -> dict:
    """
    Service health check.

    Returns:
        ``{"status": "ok", "chroma_count": int, "version": "1.0.0"}``
    """
    try:
        vs = VectorStore()
        chroma_count = await vs.count()
    except Exception:
        logger.exception("Health check — ChromaDB unreachable")
        chroma_count = -1

    return {
        "status": "ok",
        "chroma_count": chroma_count,
        "version": "1.0.0",
    }
