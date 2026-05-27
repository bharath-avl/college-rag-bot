"""
Conditional edge routing logic for the LangGraph workflow.

Each function inspects the current ``GraphState`` and returns the **name**
of the next node the graph should transition to.
"""

from __future__ import annotations

from backend.graph.state import GraphState
from backend.utils.logger import get_logger

logger = get_logger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────

_RELEVANCE_THRESHOLD: float = 0.6
_MAX_REWRITES: int = 2


def after_relevance_grader(state: GraphState) -> str:
    """
    Decide what happens after the relevance grader scores retrieved docs.

    Routing logic:
        ┌─────────────────────────────────────────────────────────────┐
        │  needs_rewrite=True  AND rewrite_count < 2  → rewriter     │
        │  needs_rewrite=True  AND rewrite_count >= 2 → fallback     │
        │  needs_rewrite=False AND score >= 0.6       → generator    │
        │  score < 0.6 (no rewrites left)             → fallback     │
        └─────────────────────────────────────────────────────────────┘

    Args:
        state: Current graph state (reads ``needs_rewrite``,
               ``rewrite_count``, ``relevance_score``).

    Returns:
        The node name to route to: ``"query_rewriter"``, ``"generator"``,
        or ``"fallback"``.
    """
    needs_rewrite: bool = state.get("needs_rewrite", False)
    rewrite_count: int = state.get("rewrite_count", 0)
    relevance_score: float = state.get("relevance_score", 0.0)

    if needs_rewrite and rewrite_count < _MAX_REWRITES:
        logger.info(
            "Routing → query_rewriter (score=%.2f, rewrite_count=%d)",
            relevance_score, rewrite_count,
        )
        return "query_rewriter"

    if relevance_score >= _RELEVANCE_THRESHOLD:
        logger.info("Routing → generator (score=%.2f)", relevance_score)
        return "generator"

    # Score too low and no rewrites left
    logger.info(
        "Routing → fallback (score=%.2f, rewrite_count=%d)",
        relevance_score, rewrite_count,
    )
    return "fallback"
