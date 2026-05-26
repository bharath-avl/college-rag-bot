"""
Pydantic v2 data models and database schema definitions.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class DocumentResult(BaseModel):
    """
    Result of ingesting a single PDF document.

    Attributes:
        text: Full extracted text content of the PDF.
        filename: Original filename of the uploaded PDF.
        subject: Subject name derived from the filename.
        pages: Total number of pages in the PDF.
        uploaded_at: Timestamp of when the document was ingested.
    """

    text: str
    filename: str
    subject: str
    pages: int = Field(ge=1)
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)


# ── SQL schema for the "documents" table ────────────────────────────────────

CREATE_DOCUMENTS_TABLE: str = """
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    filename    TEXT    NOT NULL,
    subject     TEXT    NOT NULL,
    pages       INTEGER NOT NULL,
    uploaded_at TEXT    NOT NULL
);
"""
