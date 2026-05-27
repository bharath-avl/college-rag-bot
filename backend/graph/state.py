"""
State definition for the LangGraph workflow.

This module defines the single ``GraphState`` TypedDict that flows through
every node and edge in the conversation graph.  Each field documents which
node is responsible for **setting** it so the data-flow is always clear.

Typical flow::

    USER MESSAGE
      → classify_intent  (sets intent)
      → retrieve         (sets retrieved_docs, sources)
      → grade_relevance  (sets relevance_score, needs_rewrite)
      → [rewrite_query]  (updates query, rewrite_count)   ← conditional
      → generate         (sets answer)
      → END
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class GraphState(TypedDict):
    """
    Shared state that is threaded through every node in the graph.

    ─── Chat history ───────────────────────────────────────────────────

    messages : list[AnyMessage]
        Full conversation history.  Uses the ``add_messages`` reducer so
        that each node can *append* new messages rather than replacing
        the entire list.
        **Set by:** every node that produces a user or AI message.

    ─── Current turn ───────────────────────────────────────────────────

    query : str
        The user's current natural-language question (may be rewritten
        by the ``rewrite_query`` node for better retrieval).
        **Set by:** router / entry point; updated by ``rewrite_query``.

    intent : str
        Classified intent for the current query.  One of:
        ``"syllabus"`` | ``"attendance"`` | ``"faculty"``
        | ``"exam_prep"`` | ``"timetable"`` | ``"general"``.
        **Set by:** ``classify_intent`` node.

    ─── Retrieval ──────────────────────────────────────────────────────

    retrieved_docs : list[Document]
        LangChain ``Document`` objects returned from the ChromaDB
        similarity / MMR search.
        **Set by:** ``retrieve`` node.

    relevance_score : float
        Aggregated relevance score in the range ``[0.0, 1.0]``.
        Indicates how well ``retrieved_docs`` answer the ``query``.
        **Set by:** ``grade_relevance`` node.

    ─── Generation ─────────────────────────────────────────────────────

    answer : str
        The final natural-language response returned to the user.
        **Set by:** ``generate`` node.

    sources : list[dict]
        Citation metadata for each retrieved document used in the answer.
        Each dict typically contains ``source_file``, ``subject``,
        ``page_number``, and ``chunk_index``.
        **Set by:** ``retrieve`` node (extracted from doc metadata).

    ─── Query rewriting ────────────────────────────────────────────────

    needs_rewrite : bool
        Flag indicating the current ``query`` should be rewritten
        because ``relevance_score`` is below threshold.
        **Set by:** ``grade_relevance`` node.

    rewrite_count : int
        Number of times the query has been rewritten so far in this
        turn.  Capped at ``2`` — after that the graph falls back to a
        generic "I don't know" response.
        **Set by:** ``rewrite_query`` node (incremented each pass).

    ─── Error handling ─────────────────────────────────────────────────

    error : str | None
        If any node encounters a non-recoverable error it stores the
        message here so downstream nodes and the frontend can surface it.
        ``None`` when everything is fine.
        **Set by:** any node that catches an exception.
    """

    # Chat history (append-only via add_messages reducer)
    messages: Annotated[list[AnyMessage], add_messages]

    # Current turn
    query: str
    intent: str

    # Retrieval
    retrieved_docs: list[Document]
    relevance_score: float

    # Generation
    answer: str
    sources: list[dict]

    # Query rewriting
    needs_rewrite: bool
    rewrite_count: int

    # Error handling
    error: str | None
