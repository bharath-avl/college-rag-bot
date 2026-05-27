"""
Workflow graph compilation using LangGraph StateGraph.

Assembles all nodes and edges into a single compiled graph with
SqliteSaver checkpointing for multi-turn conversation memory.

Graph topology::

    START
      │
      ▼
    intent_router
      │
      ▼
    retriever
      │
      ▼
    relevance_grader
      │
      ├── needs_rewrite=True  AND count < 2 ──► query_rewriter ──► retriever (loop)
      ├── score >= 0.6                       ──► generator       ──► END
      └── else (low score, no rewrites left) ──► fallback        ──► END
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph

from backend.graph.edges import after_relevance_grader
from backend.graph.nodes import (
    fallback,
    generator,
    intent_router,
    query_rewriter,
    relevance_grader,
    retriever,
)
from backend.graph.state import GraphState
from backend.utils.logger import get_logger

logger = get_logger(__name__)

# ── Memory DB path ──────────────────────────────────────────────────────────

_MEMORY_DB_PATH: str = os.getenv("MEMORY_DB_PATH", os.path.join("data", "memory.db"))

# ═══════════════════════════════════════════════════════════════════════════
# Build the graph
# ═══════════════════════════════════════════════════════════════════════════

workflow = StateGraph(GraphState)

# ── Add nodes ───────────────────────────────────────────────────────────────

workflow.add_node("intent_router", intent_router)
workflow.add_node("retriever", retriever)
workflow.add_node("relevance_grader", relevance_grader)
workflow.add_node("query_rewriter", query_rewriter)
workflow.add_node("generator", generator)
workflow.add_node("fallback", fallback)

# ── Linear edges ────────────────────────────────────────────────────────────

workflow.add_edge(START, "intent_router")
workflow.add_edge("intent_router", "retriever")
workflow.add_edge("retriever", "relevance_grader")

# ── Conditional edge after relevance grading ────────────────────────────────

workflow.add_conditional_edges(
    "relevance_grader",
    after_relevance_grader,
    {
        "query_rewriter": "query_rewriter",
        "generator": "generator",
        "fallback": "fallback",
    },
)

# ── Rewrite loop: rewriter feeds back into retriever ────────────────────────

workflow.add_edge("query_rewriter", "retriever")

# ── Terminal edges ──────────────────────────────────────────────────────────

workflow.add_edge("generator", END)
workflow.add_edge("fallback", END)


# ── Checkpointer state ─────────────────────────────────────────────────────

_checkpointer_ctx = None
_checkpointer = None


async def get_app():
    """
    Compile and return the LangGraph application with checkpointing.

    The ``AsyncSqliteSaver`` is opened once and kept alive for the
    process lifetime.  Call this at server startup and reuse the result.

    Returns:
        A compiled LangGraph ``CompiledGraph`` ready for ``.ainvoke()``
        and ``.astream()``.
    """
    global _checkpointer_ctx, _checkpointer

    if _checkpointer is None:
        _checkpointer_ctx = AsyncSqliteSaver.from_conn_string(_MEMORY_DB_PATH)
        _checkpointer = await _checkpointer_ctx.__aenter__()
        logger.info("Opened AsyncSqliteSaver at %s", _MEMORY_DB_PATH)

    app = workflow.compile(checkpointer=_checkpointer)
    logger.info("LangGraph app compiled with SqliteSaver at %s", _MEMORY_DB_PATH)
    return app


# ═══════════════════════════════════════════════════════════════════════════
# Streaming helper
# ═══════════════════════════════════════════════════════════════════════════


async def run_query(query: str, session_id: str) -> AsyncGenerator[str, None]:
    """
    Stream answer tokens for a user query.

    Initialises the graph state with the user's message, then iterates
    over streamed events and yields answer tokens as they arrive.

    Args:
        query: The user's natural-language question.
        session_id: Unique conversation/thread identifier for multi-turn
                    memory (maps to the checkpointer's ``thread_id``).

    Yields:
        Individual string tokens of the generated answer.

    Usage::

        async for token in run_query("What is in the math syllabus?", "user-123"):
            print(token, end="", flush=True)
    """
    app = await get_app()

    config = {"configurable": {"thread_id": session_id}}

    initial_state: dict = {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "intent": "",
        "retrieved_docs": [],
        "relevance_score": 0.0,
        "answer": "",
        "sources": [],
        "needs_rewrite": False,
        "rewrite_count": 0,
        "error": None,
    }

    logger.info("Running query for session=%s: %r", session_id, query)

    async for event in app.astream(initial_state, config=config):
        # astream yields {node_name: state_update} dicts
        for node_name, node_output in event.items():
            if isinstance(node_output, dict) and "answer" in node_output:
                yield node_output["answer"]
