"""
PDF syllabus ingestor using PyMuPDF (fitz).

Extracts text page-by-page, derives metadata from the filename,
and persists document metadata to a SQLite "documents" table via aiosqlite.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import aiosqlite
import fitz  # PyMuPDF

from backend.db.models import CREATE_DOCUMENTS_TABLE, DocumentResult
from backend.utils.config import get_env_var

# Default database path — overridable via DB_PATH env var
_DEFAULT_DB_PATH: str = os.path.join("data", "college_rag.db")


class PDFIngestor:
    """
    Ingest a PDF file: extract text, derive metadata, and store
    the document record in SQLite.

    Usage::

        ingestor = PDFIngestor()
        result = await ingestor.ingest("data/syllabi/math_101.pdf")
    """

    def __init__(self, db_path: str | None = None) -> None:
        """
        Initialise the ingestor.

        Args:
            db_path: Path to the SQLite database file.
                     Falls back to DB_PATH env var, then ``data/college_rag.db``.
        """
        self.db_path: str = db_path or os.getenv("DB_PATH", _DEFAULT_DB_PATH)

    # ── public API ──────────────────────────────────────────────────────

    async def ingest(self, file_path: str) -> DocumentResult:
        """
        Extract text from a PDF and persist its metadata to SQLite.

        Args:
            file_path: Absolute or relative path to the PDF file.

        Returns:
            A ``DocumentResult`` containing the full text and metadata.

        Raises:
            FileNotFoundError: If *file_path* does not exist.
            ValueError: If the file is not a valid PDF or has zero pages.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")

        text, total_pages = self._extract_text(path)
        filename = path.name
        subject = self._derive_subject(filename)
        uploaded_at = datetime.utcnow()

        result = DocumentResult(
            text=text,
            filename=filename,
            subject=subject,
            pages=total_pages,
            uploaded_at=uploaded_at,
        )

        await self._save_metadata(result)
        return result

    # ── private helpers ─────────────────────────────────────────────────

    @staticmethod
    def _extract_text(path: Path) -> tuple[str, int]:
        """
        Open *path* with PyMuPDF and return ``(full_text, page_count)``.

        Args:
            path: Path object pointing to the PDF.

        Returns:
            A tuple of the concatenated page text and the total page count.

        Raises:
            ValueError: If the document contains zero pages.
        """
        doc: fitz.Document = fitz.open(str(path))
        total_pages: int = len(doc)

        if total_pages == 0:
            doc.close()
            raise ValueError(f"PDF has zero pages: {path}")

        pages_text: list[str] = []
        for page_num in range(total_pages):
            page: fitz.Page = doc.load_page(page_num)
            pages_text.append(page.get_text())

        doc.close()
        return "\n".join(pages_text), total_pages

    @staticmethod
    def _derive_subject(filename: str) -> str:
        """
        Derive a human-readable subject name from the PDF filename.

        Strips the extension, replaces underscores/hyphens with spaces,
        and title-cases the result.

        Args:
            filename: The basename of the uploaded file (e.g. ``math_101.pdf``).

        Returns:
            A cleaned subject string (e.g. ``"Math 101"``).
        """
        stem = Path(filename).stem
        return stem.replace("_", " ").replace("-", " ").strip().title()

    async def _save_metadata(self, result: DocumentResult) -> None:
        """
        Insert the document metadata into the SQLite ``documents`` table.

        Creates the table if it does not yet exist.

        Args:
            result: The ``DocumentResult`` to persist.
        """
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(CREATE_DOCUMENTS_TABLE)
            await db.execute(
                """
                INSERT INTO documents (filename, subject, pages, uploaded_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    result.filename,
                    result.subject,
                    result.pages,
                    result.uploaded_at.isoformat(),
                ),
            )
            await db.commit()
