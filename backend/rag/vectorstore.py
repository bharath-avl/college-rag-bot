"""
ChromaDB vector store operations — singleton wrapper.

All ChromaDB operations in the project **must** go through this module.
Uses ``langchain_chroma.Chroma`` with ``HuggingFaceEmbeddings`` (BAAI/bge-small-en-v1.5)
which runs locally on Apple Silicon (M2) at zero API cost.
"""

from __future__ import annotations

import asyncio
import os
from functools import partial
from threading import Lock
from typing import ClassVar

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from backend.utils.logger import get_logger

logger = get_logger(__name__)

# ── defaults ────────────────────────────────────────────────────────────────

_DEFAULT_PERSIST_DIR: str = os.path.join("data", "chroma")
_DEFAULT_COLLECTION: str = "college_docs"
_EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"


class VectorStore:
    """
    Singleton wrapper around ChromaDB for document embedding and retrieval.

    Only **one** Chroma connection is kept alive for the lifetime of the
    process.  All public methods are ``async`` — blocking Chroma calls are
    off-loaded to the default executor via ``asyncio.get_event_loop().run_in_executor``.

    Usage::

        vs = VectorStore()        # always returns the same instance
        n  = await vs.add_documents(chunks)
        results = await vs.similarity_search("What is calculus?")
    """

    # ── singleton machinery ─────────────────────────────────────────────

    _instance: ClassVar[VectorStore | None] = None
    _lock: ClassVar[Lock] = Lock()

    def __new__(cls, *args: object, **kwargs: object) -> VectorStore:
        """
        Ensure only a single ``VectorStore`` instance exists process-wide.
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialised = False  # type: ignore[attr-defined]
            return cls._instance

    def __init__(
        self,
        persist_directory: str | None = None,
        collection_name: str = _DEFAULT_COLLECTION,
    ) -> None:
        """
        Initialise the vector store (runs only once due to singleton guard).

        Args:
            persist_directory: Path to the ChromaDB persistence directory.
                               Falls back to ``CHROMA_PERSIST_DIR`` env var,
                               then ``data/chroma``.
            collection_name: Name of the Chroma collection to use.
        """
        if self._initialised:  # type: ignore[attr-defined]
            return

        self._persist_dir: str = (
            persist_directory
            or os.getenv("CHROMA_PERSIST_DIR", _DEFAULT_PERSIST_DIR)
        )
        self._collection_name: str = collection_name

        logger.info(
            "Initialising embeddings model: %s (local, no API cost)",
            _EMBEDDING_MODEL,
        )
        self._embeddings = HuggingFaceEmbeddings(
            model_name=_EMBEDDING_MODEL,
            model_kwargs={"device": "mps"},       # Apple M2 Metal
            encode_kwargs={"normalize_embeddings": True},
        )

        logger.info(
            "Connecting to ChromaDB — persist_dir=%s, collection=%s",
            self._persist_dir,
            self._collection_name,
        )
        self._store = Chroma(
            collection_name=self._collection_name,
            embedding_function=self._embeddings,
            persist_directory=self._persist_dir,
        )

        self._initialised = True  # type: ignore[attr-defined]
        logger.info("VectorStore singleton ready")

    # ── async helpers ───────────────────────────────────────────────────

    @staticmethod
    async def _run_sync(func, *args):  # noqa: ANN001, ANN002, ANN202
        """
        Run a synchronous *func* in the default thread-pool executor.

        Args:
            func: Any callable.
            *args: Positional arguments forwarded to *func*.

        Returns:
            Whatever *func* returns.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, partial(func, *args))

    # ── public API ──────────────────────────────────────────────────────

    async def add_documents(self, docs: list[Document]) -> int:
        """
        Embed and store a list of LangChain ``Document`` objects.

        Args:
            docs: Documents (with ``page_content`` and ``metadata``) to add.

        Returns:
            The number of documents successfully added.
        """
        if not docs:
            logger.warning("add_documents called with an empty list — skipping")
            return 0

        logger.info("Adding %d document chunks to ChromaDB …", len(docs))
        await self._run_sync(self._store.add_documents, docs)
        logger.info("Successfully added %d chunks", len(docs))
        return len(docs)

    async def similarity_search(self, query: str, k: int = 5) -> list[Document]:
        """
        Return the *k* most similar documents for *query* using cosine similarity.

        Args:
            query: The user's natural-language question.
            k: Number of results to return.

        Returns:
            A list of up to *k* ``Document`` objects ranked by relevance.
        """
        logger.info("Similarity search — query=%r, k=%d", query, k)
        results: list[Document] = await self._run_sync(
            self._store.similarity_search, query, k,
        )
        logger.info("Similarity search returned %d results", len(results))
        return results

    async def mmr_search(self, query: str, k: int = 5) -> list[Document]:
        """
        Return the *k* most relevant-yet-diverse documents using
        Maximal Marginal Relevance (MMR).

        Args:
            query: The user's natural-language question.
            k: Number of results to return.

        Returns:
            A list of up to *k* ``Document`` objects balancing relevance and diversity.
        """
        logger.info("MMR search — query=%r, k=%d", query, k)
        results: list[Document] = await self._run_sync(
            self._store.max_marginal_relevance_search, query, k,
        )
        logger.info("MMR search returned %d results", len(results))
        return results

    async def delete_by_source(self, filename: str) -> bool:
        """
        Delete all chunks whose ``source_file`` metadata matches *filename*.

        Args:
            filename: The original PDF filename (e.g. ``"math_101.pdf"``).

        Returns:
            ``True`` if at least one document was deleted, ``False`` otherwise.
        """
        logger.info("Deleting chunks where source_file=%r", filename)

        # Retrieve matching doc IDs via the underlying Chroma collection.
        collection = self._store._collection  # noqa: SLF001
        results = await self._run_sync(
            collection.get,
            None,  # ids
            {"source_file": filename},  # where filter
        )

        ids_to_delete: list[str] = results.get("ids", [])
        if not ids_to_delete:
            logger.warning("No chunks found for source_file=%r", filename)
            return False

        await self._run_sync(collection.delete, ids_to_delete)
        logger.info(
            "Deleted %d chunks for source_file=%r", len(ids_to_delete), filename,
        )
        return True

    async def count(self) -> int:
        """
        Return the total number of documents stored in the collection.

        Returns:
            An integer count of stored chunks.
        """
        collection = self._store._collection  # noqa: SLF001
        total: int = await self._run_sync(collection.count)
        logger.info("Collection contains %d chunks", total)
        return total
