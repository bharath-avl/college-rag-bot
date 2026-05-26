"""
Document chunking logic using LangChain's RecursiveCharacterTextSplitter.

Splits a ``DocumentResult`` into tagged ``Document`` objects ready for
embedding and storage in the vector store.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.db.models import DocumentResult

# ── defaults ────────────────────────────────────────────────────────────────

_DEFAULT_CHUNK_SIZE: int = 800
_DEFAULT_CHUNK_OVERLAP: int = 100


class Chunker:
    """
    Split an ingested document into metadata-tagged LangChain ``Document`` chunks.

    Usage::

        chunker = Chunker()
        chunks = chunker.split(document_result)

    Args:
        chunk_size: Maximum character length of each chunk.
        chunk_overlap: Number of overlapping characters between consecutive chunks.
    """

    def __init__(
        self,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        """
        Initialise the chunker with configurable split parameters.

        Args:
            chunk_size: Maximum number of characters per chunk.
            chunk_overlap: Number of characters that overlap between chunks.
        """
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def split(self, doc_result: DocumentResult) -> list[Document]:
        """
        Split a ``DocumentResult`` into a list of tagged ``Document`` chunks.

        Each returned ``Document`` carries metadata with:
        - **source_file**: Original PDF filename.
        - **subject**: Derived subject name.
        - **page_number**: 1-indexed page the chunk originated from (best-effort
          estimate based on ``\\n`` page boundaries produced by the ingestor).
        - **chunk_index**: 0-indexed position of this chunk in the full sequence.

        Args:
            doc_result: The ``DocumentResult`` returned by ``PDFIngestor.ingest()``.

        Returns:
            A list of LangChain ``Document`` objects with text and metadata.
        """
        raw_chunks: list[str] = self.splitter.split_text(doc_result.text)

        # Build page boundary map so we can tag each chunk with a page number.
        page_boundaries: list[int] = self._build_page_boundaries(doc_result.text)

        documents: list[Document] = []
        running_offset: int = 0

        for chunk_index, chunk_text in enumerate(raw_chunks):
            # Find the character offset of this chunk in the original text.
            chunk_start: int = doc_result.text.find(chunk_text, running_offset)
            if chunk_start == -1:
                # Fallback: if exact match fails, keep previous offset.
                chunk_start = running_offset

            page_number: int = self._page_for_offset(chunk_start, page_boundaries)

            documents.append(
                Document(
                    page_content=chunk_text,
                    metadata={
                        "source_file": doc_result.filename,
                        "subject": doc_result.subject,
                        "page_number": page_number,
                        "chunk_index": chunk_index,
                    },
                )
            )

            # Advance the running offset past the current chunk to handle
            # duplicate text across pages.
            running_offset = chunk_start + len(chunk_text)

        return documents

    # ── private helpers ─────────────────────────────────────────────────

    @staticmethod
    def _build_page_boundaries(text: str) -> list[int]:
        """
        Build a list of character offsets where each page begins.

        The ingestor joins pages with ``\\n``, so page boundaries are
        approximated by splitting on newlines and accumulating lengths.

        Args:
            text: The full document text produced by the ingestor.

        Returns:
            A sorted list of cumulative character offsets marking each page start.
        """
        pages: list[str] = text.split("\n")
        boundaries: list[int] = []
        offset: int = 0
        for page in pages:
            boundaries.append(offset)
            offset += len(page) + 1  # +1 for the newline character
        return boundaries

    @staticmethod
    def _page_for_offset(offset: int, boundaries: list[int]) -> int:
        """
        Return the 1-indexed page number for a given character *offset*.

        Uses a simple linear scan (fast enough for typical syllabus PDFs).

        Args:
            offset: The character offset in the full document text.
            boundaries: Sorted list of page-start offsets from
                        ``_build_page_boundaries``.

        Returns:
            The 1-indexed page number containing *offset*.
        """
        page: int = 1
        for i, boundary in enumerate(boundaries):
            if offset >= boundary:
                page = i + 1
            else:
                break
        return page
