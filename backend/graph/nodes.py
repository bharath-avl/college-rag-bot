"""
LangGraph workflow nodes.

Each node is an ``async`` function that receives the current ``GraphState``
and returns a **partial** dict of fields to merge back into the state.

Model allocation strategy (cost-optimised):
    ┌──────────────────────┬────────────────────────────────┐
    │  gemini-2.0-flash    │  intent_router, relevance_     │
    │  (cheap / fast)      │  grader                        │
    ├──────────────────────┼────────────────────────────────┤
    │  gemini-2.5-pro      │  query_rewriter, generator     │
    │  (smart / quality)   │  (reasoning-heavy)             │
    └──────────────────────┴────────────────────────────────┘
"""

from __future__ import annotations

import os
import re

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from backend.graph.state import GraphState
from backend.rag.vectorstore import VectorStore
from backend.utils.logger import get_logger

logger = get_logger(__name__)

# ── Valid intents ───────────────────────────────────────────────────────────

VALID_INTENTS: set[str] = {
    "syllabus",
    "attendance",
    "faculty",
    "exam_prep",
    "timetable",
    "general",
}

# ── LLM factory helpers ────────────────────────────────────────────────────

_GOOGLE_API_KEY: str | None = os.getenv("GOOGLE_API_KEY")


def _flash_llm(temperature: float = 0.0) -> ChatGoogleGenerativeAI:
    """
    Return a *cheap / fast* Gemini Flash model for classification tasks.

    Args:
        temperature: Sampling temperature (default ``0.0`` for determinism).

    Returns:
        A configured ``ChatGoogleGenerativeAI`` instance.
    """
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=_GOOGLE_API_KEY,
        temperature=temperature,
    )


def _pro_llm(temperature: float = 0.3) -> ChatGoogleGenerativeAI:
    """
    Return a *smart / high-quality* Gemini Pro model for reasoning tasks.

    Args:
        temperature: Sampling temperature (default ``0.3`` for creativity).

    Returns:
        A configured ``ChatGoogleGenerativeAI`` instance.
    """
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        google_api_key=_GOOGLE_API_KEY,
        temperature=temperature,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Node 1 — Intent Router
# ═══════════════════════════════════════════════════════════════════════════


async def intent_router(state: GraphState) -> dict:
    """
    Classify the user query into one of the six intent categories.

    Uses **gemini-2.0-flash** — classification is a cheap, low-latency task.

    Args:
        state: Current graph state (reads ``query``).

    Returns:
        ``{"intent": <str>}`` where the string is one of ``VALID_INTENTS``.
    """
    query: str = state["query"]
    logger.info("Classifying intent for query: %r", query)

    llm = _flash_llm(temperature=0.0)

    system_prompt = (
        "You are an intent classifier for a college chatbot.\n"
        "Classify the user's question into exactly ONE of these categories:\n"
        "  syllabus    — questions about course content, topics, syllabus PDFs\n"
        "  attendance  — questions about attendance records, percentages, eligibility\n"
        "  faculty     — questions about professors, staff, department contacts\n"
        "  exam_prep   — questions about exams, previous papers, study tips\n"
        "  timetable   — questions about class schedules, room numbers, timings\n"
        "  general     — anything that does not fit the above\n\n"
        "Respond with ONLY the category name, nothing else."
    )

    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        ("human", query),
    ])

    intent = response.content.strip().lower()

    # Guard: if the model returns something unexpected, default to "general"
    if intent not in VALID_INTENTS:
        logger.warning("LLM returned unknown intent %r — defaulting to 'general'", intent)
        intent = "general"

    logger.info("Intent classified as: %s", intent)
    return {"intent": intent}


# ═══════════════════════════════════════════════════════════════════════════
# Node 2 — Retriever
# ═══════════════════════════════════════════════════════════════════════════


async def retriever(state: GraphState) -> dict:
    """
    Retrieve relevant documents from ChromaDB via MMR search.

    No LLM is used — this node only calls the ``VectorStore`` singleton.
    Also extracts citation metadata from each document into ``sources``.

    Args:
        state: Current graph state (reads ``query``).

    Returns:
        ``{"retrieved_docs": [...], "sources": [...]}``
    """
    query: str = state["query"]
    logger.info("Retrieving documents for query: %r", query)

    vs = VectorStore()
    docs: list[Document] = await vs.mmr_search(query, k=5)

    sources: list[dict] = [
        {
            "source_file": doc.metadata.get("source_file", "unknown"),
            "subject": doc.metadata.get("subject", "unknown"),
            "page_number": doc.metadata.get("page_number", 0),
            "chunk_index": doc.metadata.get("chunk_index", 0),
        }
        for doc in docs
    ]

    logger.info("Retrieved %d documents with %d source citations", len(docs), len(sources))
    return {"retrieved_docs": docs, "sources": sources}


# ═══════════════════════════════════════════════════════════════════════════
# Node 3 — Relevance Grader
# ═══════════════════════════════════════════════════════════════════════════

_RELEVANCE_THRESHOLD: float = 0.6
_MAX_REWRITES: int = 2


async def relevance_grader(state: GraphState) -> dict:
    """
    Score how well the retrieved documents answer the user's query.

    Uses **gemini-2.0-flash** — grading is cheap and runs on every query.

    Logic:
        - If ``relevance_score < 0.6`` **and** ``rewrite_count < 2``:
          set ``needs_rewrite = True`` so the graph loops back.
        - Otherwise ``needs_rewrite = False``.

    Args:
        state: Current graph state (reads ``query``, ``retrieved_docs``,
               ``rewrite_count``).

    Returns:
        ``{"relevance_score": <float>, "needs_rewrite": <bool>}``
    """
    query: str = state["query"]
    docs: list[Document] = state.get("retrieved_docs", [])
    rewrite_count: int = state.get("rewrite_count", 0)

    if not docs:
        logger.warning("No documents retrieved — score 0.0")
        return {
            "relevance_score": 0.0,
            "needs_rewrite": rewrite_count < _MAX_REWRITES,
        }

    llm = _flash_llm(temperature=0.0)

    context_text = "\n---\n".join(doc.page_content for doc in docs)

    system_prompt = (
        "You are a relevance grader for a college RAG chatbot.\n"
        "Given a user QUERY and CONTEXT (retrieved documents), rate how well\n"
        "the context answers the query.\n\n"
        "Respond with ONLY a single decimal number between 0.0 and 1.0.\n"
        "  1.0 = context perfectly answers the query\n"
        "  0.0 = context is completely irrelevant\n"
        "No explanation, just the number."
    )

    human_message = (
        f"QUERY: {query}\n\n"
        f"CONTEXT:\n{context_text}"
    )

    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        ("human", human_message),
    ])

    # Parse score — extract the first float-like pattern from the response
    raw = response.content.strip()
    match = re.search(r"(\d+\.?\d*)", raw)
    score = float(match.group(1)) if match else 0.0
    score = max(0.0, min(1.0, score))  # clamp to [0, 1]

    needs_rewrite = score < _RELEVANCE_THRESHOLD and rewrite_count < _MAX_REWRITES

    logger.info(
        "Relevance score=%.2f, needs_rewrite=%s (rewrite_count=%d)",
        score, needs_rewrite, rewrite_count,
    )
    return {"relevance_score": score, "needs_rewrite": needs_rewrite}


# ═══════════════════════════════════════════════════════════════════════════
# Node 4 — Query Rewriter
# ═══════════════════════════════════════════════════════════════════════════


async def query_rewriter(state: GraphState) -> dict:
    """
    Rewrite the user query to improve retrieval results.

    Uses **gemini-2.5-pro** — rewriting requires genuine reasoning about
    what the user really meant and how to phrase it for vector search.

    Increments ``rewrite_count`` each time it runs.

    Args:
        state: Current graph state (reads ``query``, ``intent``,
               ``rewrite_count``).

    Returns:
        ``{"query": <rewritten_query>, "rewrite_count": <int>}``
    """
    original_query: str = state["query"]
    intent: str = state.get("intent", "general")
    rewrite_count: int = state.get("rewrite_count", 0)

    logger.info(
        "Rewriting query (attempt %d): %r", rewrite_count + 1, original_query,
    )

    llm = _pro_llm(temperature=0.3)

    system_prompt = (
        "You are a query rewriter for a college RAG chatbot.\n"
        "The user's original question did not retrieve good results.\n\n"
        f"The detected intent is: {intent}\n\n"
        "Rewrite the question to be more specific and retrieval-friendly.\n"
        "Add relevant academic keywords that might appear in syllabus PDFs.\n"
        "Respond with ONLY the rewritten question, nothing else."
    )

    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        ("human", f"Original question: {original_query}"),
    ])

    rewritten = response.content.strip()
    logger.info("Rewritten query: %r", rewritten)

    return {
        "query": rewritten,
        "rewrite_count": rewrite_count + 1,
        "needs_rewrite": False,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Node 5 — Generator
# ═══════════════════════════════════════════════════════════════════════════


async def generator(state: GraphState) -> dict:
    """
    Generate the final answer using retrieved context.

    Uses **gemini-2.5-pro** — this is the most important node and the
    user-facing output quality depends on it.

    The system prompt enforces:
    - Answer ONLY from provided context.
    - Cite which document the answer comes from.
    - Admit when context is insufficient.

    Args:
        state: Current graph state (reads ``query``, ``intent``,
               ``retrieved_docs``, ``messages``).

    Returns:
        ``{"answer": <str>, "messages": [AIMessage(...)]}``
    """
    query: str = state["query"]
    intent: str = state.get("intent", "general")
    docs: list[Document] = state.get("retrieved_docs", [])

    logger.info("Generating answer for query: %r (intent=%s)", query, intent)

    llm = _pro_llm(temperature=0.3)

    # Build context block from retrieved docs
    if docs:
        context_parts: list[str] = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source_file", "unknown")
            page = doc.metadata.get("page_number", "?")
            context_parts.append(
                f"[Document {i} | Source: {source} | Page: {page}]\n"
                f"{doc.page_content}"
            )
        context_block = "\n\n---\n\n".join(context_parts)
    else:
        context_block = "(No documents were retrieved.)"

    system_prompt = (
        "You are a helpful college assistant chatbot.\n\n"
        "RULES:\n"
        "1. Answer ONLY using the provided context below.\n"
        "2. If the context is insufficient to answer, say so clearly.\n"
        "3. Always cite which document your answer comes from "
        "(e.g. [Source: filename.pdf, Page: X]).\n"
        "4. Be concise but thorough.\n"
        "5. Format your answer with clear structure when appropriate "
        "(bullet points, numbered lists).\n\n"
        f"DETECTED INTENT: {intent}\n\n"
        f"CONTEXT:\n{context_block}"
    )

    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        ("human", query),
    ])

    answer: str = response.content.strip()
    logger.info("Generated answer (%d chars)", len(answer))

    return {
        "answer": answer,
        "messages": [AIMessage(content=answer)],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Node 6 — Fallback
# ═══════════════════════════════════════════════════════════════════════════

_FALLBACK_MESSAGE: str = (
    "I could not find this information in the uploaded documents. "
    "Please contact your department admin or check the college portal."
)


async def fallback(state: GraphState) -> dict:
    """
    Return a static helpful message when retrieval + rewriting failed.

    No LLM is needed — this is a hard-coded safety net.

    Args:
        state: Current graph state (not read, included for signature
               consistency).

    Returns:
        ``{"answer": <fallback_message>, "messages": [AIMessage(...)]}``
    """
    logger.info("Falling back to static response")
    return {
        "answer": _FALLBACK_MESSAGE,
        "messages": [AIMessage(content=_FALLBACK_MESSAGE)],
    }
