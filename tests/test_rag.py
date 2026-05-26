"""
Tests for RAG processing module: PDFIngestor, Chunker, and VectorStore.

All external calls (ChromaDB, aiosqlite, fitz) are mocked so tests run
without a real database, embedding model, or PDF files.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document

from backend.db.models import DocumentResult
from backend.rag.chunker import Chunker
from backend.rag.ingestor import PDFIngestor


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture()
def sample_doc_result() -> DocumentResult:
    """
    Build a ``DocumentResult`` with enough text to produce multiple chunks.
    """
    # ~2400 chars ⇒ should produce ~3 chunks at chunk_size=800
    long_text = (
        "This is page one of the syllabus covering linear algebra. " * 20
        + "\n"
        + "This is page two covering calculus fundamentals. " * 20
        + "\n"
        + "This is page three covering probability and statistics. " * 20
    )
    return DocumentResult(
        text=long_text,
        filename="math_101.pdf",
        subject="Math 101",
        pages=3,
        uploaded_at=datetime(2026, 1, 15, 10, 30, 0),
    )


@pytest.fixture()
def short_doc_result() -> DocumentResult:
    """
    Build a ``DocumentResult`` whose text fits in a single chunk.
    """
    return DocumentResult(
        text="Short syllabus content.",
        filename="short.pdf",
        subject="Short",
        pages=1,
        uploaded_at=datetime(2026, 1, 15, 10, 30, 0),
    )


# ═══════════════════════════════════════════════════════════════════════════
# PDFIngestor tests
# ═══════════════════════════════════════════════════════════════════════════


class TestPDFIngestor:
    """Tests for ``PDFIngestor``."""

    @pytest.mark.asyncio
    async def test_ingest_extracts_text_and_metadata(self, tmp_path: Path) -> None:
        """
        Ingestor should return a ``DocumentResult`` with text from each page
        and correct metadata derived from the filename.
        """
        fake_pdf = tmp_path / "physics_201.pdf"
        fake_pdf.touch()  # file must exist for the Path.exists() check

        mock_page_0 = MagicMock()
        mock_page_0.get_text.return_value = "Page one text."
        mock_page_1 = MagicMock()
        mock_page_1.get_text.return_value = "Page two text."

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=2)
        mock_doc.load_page.side_effect = [mock_page_0, mock_page_1]

        with (
            patch("backend.rag.ingestor.fitz.open", return_value=mock_doc),
            patch("backend.rag.ingestor.aiosqlite") as mock_aiosqlite,
        ):
            # aiosqlite.connect() returns an async context manager
            mock_db = AsyncMock()
            mock_aiosqlite.connect.return_value.__aenter__ = AsyncMock(
                return_value=mock_db,
            )
            mock_aiosqlite.connect.return_value.__aexit__ = AsyncMock(
                return_value=False,
            )

            ingestor = PDFIngestor(db_path=":memory:")
            result = await ingestor.ingest(str(fake_pdf))

        assert isinstance(result, DocumentResult)
        assert "Page one text." in result.text
        assert "Page two text." in result.text
        assert result.filename == "physics_201.pdf"
        assert result.subject == "Physics 201"
        assert result.pages == 2

    @pytest.mark.asyncio
    async def test_ingest_file_not_found_raises(self) -> None:
        """
        Calling ingest with a non-existent path must raise ``FileNotFoundError``.
        """
        ingestor = PDFIngestor(db_path=":memory:")
        with pytest.raises(FileNotFoundError):
            await ingestor.ingest("/nonexistent/path/fake.pdf")

    @pytest.mark.asyncio
    async def test_ingest_zero_pages_raises(self, tmp_path: Path) -> None:
        """
        A PDF with zero pages must raise ``ValueError``.
        """
        fake_pdf = tmp_path / "empty.pdf"
        fake_pdf.touch()

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=0)

        with patch("backend.rag.ingestor.fitz.open", return_value=mock_doc):
            ingestor = PDFIngestor(db_path=":memory:")
            with pytest.raises(ValueError, match="zero pages"):
                await ingestor.ingest(str(fake_pdf))

    @pytest.mark.asyncio
    async def test_save_metadata_calls_sqlite(self, tmp_path: Path) -> None:
        """
        Verify that document metadata is actually persisted to SQLite.
        """
        fake_pdf = tmp_path / "cs_301.pdf"
        fake_pdf.touch()

        mock_page = MagicMock()
        mock_page.get_text.return_value = "Algorithms."

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.load_page.return_value = mock_page

        with (
            patch("backend.rag.ingestor.fitz.open", return_value=mock_doc),
            patch("backend.rag.ingestor.aiosqlite") as mock_aiosqlite,
        ):
            mock_db = AsyncMock()
            mock_aiosqlite.connect.return_value.__aenter__ = AsyncMock(
                return_value=mock_db,
            )
            mock_aiosqlite.connect.return_value.__aexit__ = AsyncMock(
                return_value=False,
            )

            ingestor = PDFIngestor(db_path=":memory:")
            await ingestor.ingest(str(fake_pdf))

        # Two execute calls: CREATE TABLE + INSERT
        assert mock_db.execute.await_count == 2
        assert mock_db.commit.await_count == 1

    def test_derive_subject_from_filename(self) -> None:
        """
        Subject derivation should strip extension, replace separators,
        and title-case.
        """
        assert PDFIngestor._derive_subject("math_101.pdf") == "Math 101"
        assert PDFIngestor._derive_subject("data-structures.pdf") == "Data Structures"
        assert PDFIngestor._derive_subject("AI.pdf") == "Ai"


# ═══════════════════════════════════════════════════════════════════════════
# Chunker tests
# ═══════════════════════════════════════════════════════════════════════════


class TestChunker:
    """Tests for ``Chunker``."""

    def test_split_produces_multiple_chunks(
        self, sample_doc_result: DocumentResult,
    ) -> None:
        """
        A long document should be split into more than one chunk.
        """
        chunker = Chunker(chunk_size=800, chunk_overlap=100)
        chunks = chunker.split(sample_doc_result)

        assert len(chunks) > 1
        assert all(isinstance(c, Document) for c in chunks)

    def test_chunk_size_respected(
        self, sample_doc_result: DocumentResult,
    ) -> None:
        """
        Every chunk's ``page_content`` must be at most ``chunk_size`` characters.
        """
        chunk_size = 800
        chunker = Chunker(chunk_size=chunk_size, chunk_overlap=100)
        chunks = chunker.split(sample_doc_result)

        for chunk in chunks:
            assert len(chunk.page_content) <= chunk_size

    def test_metadata_tags_present(
        self, sample_doc_result: DocumentResult,
    ) -> None:
        """
        Each chunk must carry the four required metadata keys.
        """
        chunker = Chunker()
        chunks = chunker.split(sample_doc_result)
        required_keys = {"source_file", "subject", "page_number", "chunk_index"}

        for chunk in chunks:
            assert required_keys.issubset(chunk.metadata.keys())

    def test_metadata_values_correct(
        self, sample_doc_result: DocumentResult,
    ) -> None:
        """
        Metadata must reflect the source document.
        """
        chunker = Chunker()
        chunks = chunker.split(sample_doc_result)

        for i, chunk in enumerate(chunks):
            assert chunk.metadata["source_file"] == "math_101.pdf"
            assert chunk.metadata["subject"] == "Math 101"
            assert chunk.metadata["chunk_index"] == i
            assert chunk.metadata["page_number"] >= 1

    def test_short_text_single_chunk(
        self, short_doc_result: DocumentResult,
    ) -> None:
        """
        Text shorter than ``chunk_size`` should produce exactly one chunk.
        """
        chunker = Chunker(chunk_size=800, chunk_overlap=100)
        chunks = chunker.split(short_doc_result)

        assert len(chunks) == 1
        assert chunks[0].page_content == "Short syllabus content."

    def test_empty_text_returns_empty_list(self) -> None:
        """
        An empty-text ``DocumentResult`` should produce zero chunks.
        """
        doc = DocumentResult(
            text="",
            filename="blank.pdf",
            subject="Blank",
            pages=1,
        )
        chunker = Chunker()
        chunks = chunker.split(doc)

        assert chunks == []


# ═══════════════════════════════════════════════════════════════════════════
# VectorStore tests
# ═══════════════════════════════════════════════════════════════════════════


class TestVectorStore:
    """
    Tests for ``VectorStore``.

    Every test patches the singleton to avoid loading the real
    HuggingFace embedding model or connecting to ChromaDB.
    """

    @staticmethod
    def _make_vectorstore() -> MagicMock:
        """
        Build a fully-mocked ``VectorStore`` instance without triggering
        ``__init__`` (which would try to load the real embedding model).
        """
        from backend.rag.vectorstore import VectorStore

        # Reset singleton so each test gets a fresh one
        VectorStore._instance = None

        vs = object.__new__(VectorStore)
        vs._initialised = True
        vs._store = MagicMock()
        vs._store._collection = MagicMock()
        return vs

    @pytest.mark.asyncio
    async def test_add_documents_returns_count(self) -> None:
        """
        ``add_documents`` should return the number of docs passed in.
        """
        vs = self._make_vectorstore()
        vs._store.add_documents = MagicMock()

        docs = [
            Document(page_content="chunk1", metadata={"source_file": "a.pdf"}),
            Document(page_content="chunk2", metadata={"source_file": "a.pdf"}),
            Document(page_content="chunk3", metadata={"source_file": "a.pdf"}),
        ]

        count = await vs.add_documents(docs)
        assert count == 3

    @pytest.mark.asyncio
    async def test_add_documents_empty_returns_zero(self) -> None:
        """
        Passing an empty list should return 0 and skip the Chroma call.
        """
        vs = self._make_vectorstore()
        count = await vs.add_documents([])

        assert count == 0
        vs._store.add_documents.assert_not_called()

    @pytest.mark.asyncio
    async def test_similarity_search_returns_documents(self) -> None:
        """
        ``similarity_search`` must return a ``list[Document]``.
        """
        vs = self._make_vectorstore()

        expected = [
            Document(page_content="result1", metadata={"source_file": "b.pdf"}),
            Document(page_content="result2", metadata={"source_file": "b.pdf"}),
        ]
        vs._store.similarity_search = MagicMock(return_value=expected)

        results = await vs.similarity_search("What is calculus?", k=2)

        assert len(results) == 2
        assert all(isinstance(r, Document) for r in results)
        assert results[0].page_content == "result1"

    @pytest.mark.asyncio
    async def test_mmr_search_returns_documents(self) -> None:
        """
        ``mmr_search`` must return a ``list[Document]``.
        """
        vs = self._make_vectorstore()

        expected = [
            Document(page_content="diverse1", metadata={}),
        ]
        vs._store.max_marginal_relevance_search = MagicMock(return_value=expected)

        results = await vs.mmr_search("probability", k=1)

        assert len(results) == 1
        assert results[0].page_content == "diverse1"

    @pytest.mark.asyncio
    async def test_delete_by_source_returns_true(self) -> None:
        """
        ``delete_by_source`` returns ``True`` when matching chunks exist.
        """
        vs = self._make_vectorstore()

        vs._store._collection.get = MagicMock(
            return_value={"ids": ["id1", "id2"]},
        )
        vs._store._collection.delete = MagicMock()

        deleted = await vs.delete_by_source("math_101.pdf")

        assert deleted is True
        vs._store._collection.delete.assert_called_once_with(["id1", "id2"])

    @pytest.mark.asyncio
    async def test_delete_by_source_returns_false_when_not_found(self) -> None:
        """
        ``delete_by_source`` returns ``False`` when no chunks match.
        """
        vs = self._make_vectorstore()

        vs._store._collection.get = MagicMock(
            return_value={"ids": []},
        )

        deleted = await vs.delete_by_source("nonexistent.pdf")

        assert deleted is False

    @pytest.mark.asyncio
    async def test_count_returns_integer(self) -> None:
        """
        ``count`` must return the integer total from the collection.
        """
        vs = self._make_vectorstore()
        vs._store._collection.count = MagicMock(return_value=42)

        total = await vs.count()

        assert total == 42

    @pytest.mark.asyncio
    async def test_singleton_reset_works(self) -> None:
        """
        After resetting ``_instance``, a new mock can be created cleanly.
        """
        from backend.rag.vectorstore import VectorStore

        VectorStore._instance = None
        vs1 = self._make_vectorstore()
        vs2 = self._make_vectorstore()

        # Both are distinct mocks (singleton is reset each time)
        assert vs1 is not vs2
