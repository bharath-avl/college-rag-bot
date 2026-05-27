"""
Tests for FastAPI API endpoints.

All external dependencies (VectorStore, PDFIngestor, Chunker, graph.run_query,
aiosqlite) are mocked so tests run without a database, embedding model, or LLM.

Run with::

    uv run pytest tests/test_api.py -v
"""

from __future__ import annotations

import json
from datetime import datetime
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.documents import Document

from backend.api.main import app
from backend.db.models import DocumentResult


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture()
def fake_doc_result() -> DocumentResult:
    """A ``DocumentResult`` returned by a mocked ``PDFIngestor.ingest()``."""
    return DocumentResult(
        text="Syllabus content for testing.",
        filename="test.pdf",
        subject="Test",
        pages=2,
        uploaded_at=datetime(2026, 5, 27, 12, 0, 0),
    )


@pytest.fixture()
def fake_chunks() -> list[Document]:
    """Chunks returned by a mocked ``Chunker.split()``."""
    return [
        Document(page_content="chunk 1", metadata={"source_file": "test.pdf"}),
        Document(page_content="chunk 2", metadata={"source_file": "test.pdf"}),
        Document(page_content="chunk 3", metadata={"source_file": "test.pdf"}),
    ]


@pytest.fixture()
def fake_pdf_bytes() -> bytes:
    """Minimal fake PDF content (just needs to be non-empty bytes)."""
    return b"%PDF-1.4 fake pdf content for testing"


# ── Shared mock for VectorStore singleton so __init__ never runs ────────

@pytest.fixture(autouse=True)
def _mock_vectorstore_init():
    """
    Prevent the real ``VectorStore.__init__`` from loading the embedding
    model during test collection / import of the app module.
    """
    with patch("backend.rag.vectorstore.VectorStore.__init__", return_value=None):
        yield


# ═══════════════════════════════════════════════════════════════════════════
# POST /upload
# ═══════════════════════════════════════════════════════════════════════════


class TestUploadEndpoint:
    """Tests for ``POST /upload``."""

    @pytest.mark.asyncio
    async def test_upload_pdf_returns_200_with_chunk_count(
        self,
        fake_doc_result: DocumentResult,
        fake_chunks: list[Document],
        fake_pdf_bytes: bytes,
    ) -> None:
        """
        Uploading a valid PDF should return 200 with the filename and
        number of chunks indexed.
        """
        mock_ingestor = MagicMock()
        mock_ingestor.ingest = AsyncMock(return_value=fake_doc_result)

        mock_chunker = MagicMock()
        mock_chunker.split.return_value = fake_chunks

        mock_vs = MagicMock()
        mock_vs.add_documents = AsyncMock(return_value=3)

        with (
            patch("backend.api.routes.PDFIngestor", return_value=mock_ingestor),
            patch("backend.api.routes.Chunker", return_value=mock_chunker),
            patch("backend.api.routes.VectorStore", return_value=mock_vs),
            patch("backend.api.routes.Path.mkdir"),
            patch("builtins.open", mock_open()),
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post(
                    "/upload",
                    files={"file": ("test.pdf", BytesIO(fake_pdf_bytes), "application/pdf")},
                )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["chunks_indexed"] == 3
        assert data["filename"] == "test.pdf"

    @pytest.mark.asyncio
    async def test_upload_non_pdf_returns_400(self, fake_pdf_bytes: bytes) -> None:
        """
        Uploading a non-PDF file should return a 400 error.
        """
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/upload",
                files={"file": ("notes.txt", BytesIO(fake_pdf_bytes), "text/plain")},
            )

        assert resp.status_code == 400
        assert "PDF" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_upload_calls_pipeline_in_order(
        self,
        fake_doc_result: DocumentResult,
        fake_chunks: list[Document],
        fake_pdf_bytes: bytes,
    ) -> None:
        """
        Upload should call PDFIngestor → Chunker → VectorStore in order.
        """
        mock_ingestor = MagicMock()
        mock_ingestor.ingest = AsyncMock(return_value=fake_doc_result)

        mock_chunker = MagicMock()
        mock_chunker.split.return_value = fake_chunks

        mock_vs = MagicMock()
        mock_vs.add_documents = AsyncMock(return_value=3)

        with (
            patch("backend.api.routes.PDFIngestor", return_value=mock_ingestor),
            patch("backend.api.routes.Chunker", return_value=mock_chunker),
            patch("backend.api.routes.VectorStore", return_value=mock_vs),
            patch("backend.api.routes.Path.mkdir"),
            patch("builtins.open", mock_open()),
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                await ac.post(
                    "/upload",
                    files={"file": ("pipeline.pdf", BytesIO(fake_pdf_bytes), "application/pdf")},
                )

        mock_ingestor.ingest.assert_awaited_once()
        mock_chunker.split.assert_called_once_with(fake_doc_result)
        mock_vs.add_documents.assert_awaited_once_with(fake_chunks)


# ═══════════════════════════════════════════════════════════════════════════
# POST /chat
# ═══════════════════════════════════════════════════════════════════════════


class TestChatEndpoint:
    """Tests for ``POST /chat`` (SSE streaming)."""

    @pytest.mark.asyncio
    async def test_chat_returns_streaming_sse_response(self) -> None:
        """
        Chat should return a streaming response with ``text/event-stream``
        content type and well-formed SSE events.
        """

        async def _fake_run_query(query: str, session_id: str):
            yield "Hello"
            yield " world"

        with patch("backend.api.routes.run_query", side_effect=_fake_run_query):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post(
                    "/chat",
                    json={"query": "What is calculus?", "session_id": "sess-1"},
                )

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

        # Parse SSE events
        lines = resp.text.strip().split("\n\n")
        events = []
        for line in lines:
            if line.startswith("data: "):
                events.append(json.loads(line[len("data: "):]))

        # Should have: "Hello", " world", and the final done=True event
        assert len(events) == 3

        # First two tokens
        assert events[0]["token"] == "Hello"
        assert events[0]["done"] is False
        assert events[1]["token"] == " world"
        assert events[1]["done"] is False

        # Final event
        assert events[2]["done"] is True
        assert "sources" in events[2]

    @pytest.mark.asyncio
    async def test_chat_error_returns_done_with_error(self) -> None:
        """
        If ``run_query`` raises, the stream should send a final event
        with ``done=True`` and an ``error`` field.
        """

        async def _failing_run_query(query: str, session_id: str):
            raise RuntimeError("LLM failed")
            yield  # noqa: RET503 — make it an async generator

        with patch("backend.api.routes.run_query", side_effect=_failing_run_query):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post(
                    "/chat",
                    json={"query": "fail please", "session_id": "sess-err"},
                )

        assert resp.status_code == 200

        lines = resp.text.strip().split("\n\n")
        events = [
            json.loads(line[len("data: "):])
            for line in lines
            if line.startswith("data: ")
        ]

        # Should have exactly one error event
        assert len(events) >= 1
        last_event = events[-1]
        assert last_event["done"] is True
        assert "error" in last_event
        assert "LLM failed" in last_event["error"]


# ═══════════════════════════════════════════════════════════════════════════
# GET /documents
# ═══════════════════════════════════════════════════════════════════════════


class TestDocumentsEndpoint:
    """Tests for ``GET /documents``."""

    @pytest.mark.asyncio
    async def test_list_documents_returns_list(self) -> None:
        """
        GET /documents should return a JSON array of document records.
        """
        fake_rows = [
            {"id": 1, "filename": "math.pdf", "subject": "Math", "pages": 5, "uploaded_at": "2026-01-15T10:30:00"},
            {"id": 2, "filename": "cs.pdf", "subject": "Cs", "pages": 3, "uploaded_at": "2026-02-20T14:00:00"},
        ]

        # Build mock rows that support dict-style access
        mock_rows = []
        for row_data in fake_rows:
            mock_row = MagicMock()
            mock_row.__getitem__ = lambda self, key, rd=row_data: rd[key]
            mock_rows.append(mock_row)

        mock_cursor = AsyncMock()
        mock_cursor.fetchall.return_value = mock_rows

        mock_db = AsyncMock()
        mock_db.execute.return_value = mock_cursor

        with patch("backend.api.routes.aiosqlite") as mock_aiosqlite:
            mock_aiosqlite.connect.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_aiosqlite.connect.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_aiosqlite.Row = MagicMock()

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.get("/documents")

        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["filename"] == "math.pdf"
        assert data[1]["filename"] == "cs.pdf"

    @pytest.mark.asyncio
    async def test_list_documents_empty(self) -> None:
        """
        GET /documents with no indexed docs returns an empty list.
        """
        mock_cursor = AsyncMock()
        mock_cursor.fetchall.return_value = []

        mock_db = AsyncMock()
        mock_db.execute.return_value = mock_cursor

        with patch("backend.api.routes.aiosqlite") as mock_aiosqlite:
            mock_aiosqlite.connect.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_aiosqlite.connect.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_aiosqlite.Row = MagicMock()

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.get("/documents")

        assert resp.status_code == 200
        assert resp.json() == []


# ═══════════════════════════════════════════════════════════════════════════
# DELETE /documents/{filename}
# ═══════════════════════════════════════════════════════════════════════════


class TestDeleteDocumentEndpoint:
    """Tests for ``DELETE /documents/{filename}``."""

    @pytest.mark.asyncio
    async def test_delete_existing_document_returns_deleted(self) -> None:
        """
        Deleting a document that exists in ChromaDB should return
        ``{"status": "deleted", "filename": "test.pdf"}``.
        """
        mock_vs = MagicMock()
        mock_vs.delete_by_source = AsyncMock(return_value=True)

        mock_db = AsyncMock()

        with (
            patch("backend.api.routes.VectorStore", return_value=mock_vs),
            patch("backend.api.routes.aiosqlite") as mock_aiosqlite,
            patch("backend.api.routes.Path.exists", return_value=False),
        ):
            mock_aiosqlite.connect.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_aiosqlite.connect.return_value.__aexit__ = AsyncMock(return_value=False)

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.delete("/documents/test.pdf")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "deleted"
        assert data["filename"] == "test.pdf"

        mock_vs.delete_by_source.assert_awaited_once_with("test.pdf")

    @pytest.mark.asyncio
    async def test_delete_nonexistent_document_returns_404(self) -> None:
        """
        Deleting a document not found in ChromaDB should return 404.
        """
        mock_vs = MagicMock()
        mock_vs.delete_by_source = AsyncMock(return_value=False)

        with patch("backend.api.routes.VectorStore", return_value=mock_vs):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.delete("/documents/nonexistent.pdf")

        assert resp.status_code == 404
        assert "nonexistent.pdf" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_delete_also_removes_sqlite_row(self) -> None:
        """
        After deleting from ChromaDB, the SQLite row must also be removed.
        """
        mock_vs = MagicMock()
        mock_vs.delete_by_source = AsyncMock(return_value=True)

        mock_db = AsyncMock()

        with (
            patch("backend.api.routes.VectorStore", return_value=mock_vs),
            patch("backend.api.routes.aiosqlite") as mock_aiosqlite,
            patch("backend.api.routes.Path.exists", return_value=False),
        ):
            mock_aiosqlite.connect.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_aiosqlite.connect.return_value.__aexit__ = AsyncMock(return_value=False)

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                await ac.delete("/documents/remove_me.pdf")

        # Verify the DELETE SQL was executed
        mock_db.execute.assert_awaited_once_with(
            "DELETE FROM documents WHERE filename = ?", ("remove_me.pdf",),
        )
        mock_db.commit.assert_awaited_once()


# ═══════════════════════════════════════════════════════════════════════════
# GET /health
# ═══════════════════════════════════════════════════════════════════════════


class TestHealthEndpoint:
    """Tests for ``GET /health``."""

    @pytest.mark.asyncio
    async def test_health_returns_ok_with_chroma_count(self) -> None:
        """
        Health endpoint should return status=ok and the current chroma count.
        """
        mock_vs = MagicMock()
        mock_vs.count = AsyncMock(return_value=42)

        with patch("backend.api.routes.VectorStore", return_value=mock_vs):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.get("/health")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["chroma_count"] == 42
        assert data["version"] == "1.0.0"

    @pytest.mark.asyncio
    async def test_health_returns_negative_count_on_chroma_error(self) -> None:
        """
        If ChromaDB is unreachable, health should still return 200 but
        with ``chroma_count=-1``.
        """
        mock_vs = MagicMock()
        mock_vs.count = AsyncMock(side_effect=ConnectionError("DB down"))

        with patch("backend.api.routes.VectorStore", return_value=mock_vs):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.get("/health")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["chroma_count"] == -1
