"""
Main FastAPI application entry point for the College RAG Chatbot.

Configures:
- CORS middleware (allows localhost:8501 for the Streamlit frontend).
- Startup event: initialises the ChromaDB singleton and creates SQLite tables.
- Global exception handler for unhandled errors.
- Includes the API router from ``backend.api.routes``.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

import aiosqlite
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.routes import router
from backend.db.models import CREATE_DOCUMENTS_TABLE
from backend.rag.vectorstore import VectorStore
from backend.utils.logger import get_logger

logger = get_logger(__name__)

# ── defaults ────────────────────────────────────────────────────────────────

_DEFAULT_DB_PATH: str = os.path.join("data", "college_rag.db")


# ═════════════════════════════════════════════════════════════════════════════
# Lifespan (startup / shutdown)
# ═════════════════════════════════════════════════════════════════════════════


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    **Startup**:
    1. Ensure the ``data/`` directory exists.
    2. Initialise the ``VectorStore`` singleton (loads the embedding model once).
    3. Create the SQLite ``documents`` table if it doesn't exist.

    **Shutdown**:
    Log a clean shutdown message.
    """
    # ── Startup ─────────────────────────────────────────────────────────
    logger.info("Starting College RAG Chatbot API …")

    # Ensure data dir
    data_dir = os.path.dirname(os.getenv("DB_PATH", _DEFAULT_DB_PATH))
    if data_dir:
        os.makedirs(data_dir, exist_ok=True)

    # Initialise ChromaDB singleton (loads embedding model once)
    logger.info("Initialising ChromaDB singleton …")
    VectorStore()
    logger.info("ChromaDB singleton ready")

    # Create SQLite tables
    db_path = os.getenv("DB_PATH", _DEFAULT_DB_PATH)
    logger.info("Ensuring SQLite tables exist at %s", db_path)
    async with aiosqlite.connect(db_path) as db:
        await db.execute(CREATE_DOCUMENTS_TABLE)
        await db.commit()
    logger.info("SQLite tables ready")

    logger.info("Startup complete ✓")

    yield

    # ── Shutdown ────────────────────────────────────────────────────────
    logger.info("Shutting down College RAG Chatbot API …")


# ═════════════════════════════════════════════════════════════════════════════
# App factory
# ═════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="College RAG Chatbot API",
    description="RAG-powered chatbot for college syllabus Q&A",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS middleware ─────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",       # Streamlit default
        "http://127.0.0.1:8501",
        "http://localhost:3000",        # Dev convenience
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Include API router ─────────────────────────────────────────────────────

app.include_router(router)


# ═════════════════════════════════════════════════════════════════════════════
# Global exception handler
# ═════════════════════════════════════════════════════════════════════════════


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catch-all handler for unhandled exceptions.

    Logs the full traceback and returns a clean JSON error response
    so the client always receives structured output.

    Args:
        request: The incoming ``Request`` object.
        exc: The unhandled exception.

    Returns:
        A ``JSONResponse`` with status 500 and ``{"error": str}``.
    """
    logger.exception("Unhandled exception on %s %s", request.method, request.url)
    return JSONResponse(
        status_code=500,
        content={"error": str(exc)},
    )
