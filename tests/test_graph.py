"""
Tests for the LangGraph agent: nodes, edges, and full graph traversal.

All external calls (Gemini LLM, VectorStore / ChromaDB) are mocked so
tests run without API keys, network access, or a real vector database.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from backend.graph.edges import after_relevance_grader
from backend.graph.nodes import (
    _FALLBACK_MESSAGE,
    fallback,
    generator,
    intent_router,
    query_rewriter,
    relevance_grader,
    retriever,
)

# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

_SAMPLE_DOCS: list[Document] = [
    Document(
        page_content="Operating Systems covers process management, memory.",
        metadata={
            "source_file": "os_syllabus.pdf",
            "subject": "Operating Systems",
            "page_number": 1,
            "chunk_index": 0,
        },
    ),
    Document(
        page_content="Topics include scheduling, deadlocks, file systems.",
        metadata={
            "source_file": "os_syllabus.pdf",
            "subject": "Operating Systems",
            "page_number": 2,
            "chunk_index": 1,
        },
    ),
]


def _make_llm_response(text: str) -> MagicMock:
    """
    Build a mock LLM response object with ``.content`` attribute.
    """
    resp = MagicMock()
    resp.content = text
    return resp


def _base_state(**overrides) -> dict:
    """
    Return a minimal valid ``GraphState``-compatible dict with sensible
    defaults, applying any *overrides*.
    """
    state = {
        "messages": [HumanMessage(content="test")],
        "query": "What topics are in OS?",
        "intent": "",
        "retrieved_docs": [],
        "relevance_score": 0.0,
        "answer": "",
        "sources": [],
        "needs_rewrite": False,
        "rewrite_count": 0,
        "error": None,
    }
    state.update(overrides)
    return state


# ═══════════════════════════════════════════════════════════════════════════
# 1 · Happy path — individual node tests that compose into the happy flow
# ═══════════════════════════════════════════════════════════════════════════


class TestHappyPath:
    """End-to-end happy path tested node-by-node."""

    @pytest.mark.asyncio
    async def test_intent_router_then_retriever_then_grader_then_generator(
        self,
    ) -> None:
        """
        Simulate the full happy path through individual node calls:
        intent_router → retriever → relevance_grader → generator.
        """
        state = _base_state()

        # ── Node 1: intent_router ───────────────────────────────────────
        mock_flash = AsyncMock(return_value=_make_llm_response("syllabus"))
        with patch("backend.graph.nodes._flash_llm") as mock_factory:
            mock_factory.return_value.ainvoke = mock_flash
            result = await intent_router(state)

        assert result["intent"] == "syllabus"
        state.update(result)

        # ── Node 2: retriever ───────────────────────────────────────────
        mock_vs = MagicMock()
        mock_vs.mmr_search = AsyncMock(return_value=_SAMPLE_DOCS)
        with patch("backend.graph.nodes.VectorStore", return_value=mock_vs):
            result = await retriever(state)

        assert len(result["retrieved_docs"]) == 2
        assert len(result["sources"]) == 2
        assert result["sources"][0]["source_file"] == "os_syllabus.pdf"
        state.update(result)

        # ── Node 3: relevance_grader (score is good) ────────────────────
        mock_flash2 = AsyncMock(return_value=_make_llm_response("0.85"))
        with patch("backend.graph.nodes._flash_llm") as mock_factory:
            mock_factory.return_value.ainvoke = mock_flash2
            result = await relevance_grader(state)

        assert result["relevance_score"] == 0.85
        assert result["needs_rewrite"] is False
        state.update(result)

        # ── Edge routing should go to generator ─────────────────────────
        route = after_relevance_grader(state)
        assert route == "generator"

        # ── Node 5: generator ───────────────────────────────────────────
        mock_pro = AsyncMock(
            return_value=_make_llm_response(
                "OS covers process management and scheduling. "
                "[Source: os_syllabus.pdf, Page: 1]"
            ),
        )
        with patch("backend.graph.nodes._pro_llm") as mock_factory:
            mock_factory.return_value.ainvoke = mock_pro
            result = await generator(state)

        assert "process management" in result["answer"]
        assert isinstance(result["messages"][0], AIMessage)


# ═══════════════════════════════════════════════════════════════════════════
# 2 · Rewrite path — grader rejects, query is rewritten, then succeeds
# ═══════════════════════════════════════════════════════════════════════════


class TestRewritePath:
    """Grader rejects → rewriter → retriever → grader approves → generator."""

    @pytest.mark.asyncio
    async def test_rewrite_then_approve(self) -> None:
        """
        First grading returns low score ⇒ rewrite ⇒ second grading passes.
        """
        state = _base_state(
            retrieved_docs=_SAMPLE_DOCS,
            intent="syllabus",
            rewrite_count=0,
        )

        # ── Grader: low score, triggers rewrite ─────────────────────────
        mock_flash = AsyncMock(return_value=_make_llm_response("0.35"))
        with patch("backend.graph.nodes._flash_llm") as mock_factory:
            mock_factory.return_value.ainvoke = mock_flash
            grade_result = await relevance_grader(state)

        assert grade_result["relevance_score"] == 0.35
        assert grade_result["needs_rewrite"] is True
        state.update(grade_result)

        # ── Edge should route to query_rewriter ─────────────────────────
        assert after_relevance_grader(state) == "query_rewriter"

        # ── Rewriter: improve the query ─────────────────────────────────
        mock_pro = AsyncMock(
            return_value=_make_llm_response(
                "What are the main topics covered in Operating Systems syllabus?"
            ),
        )
        with patch("backend.graph.nodes._pro_llm") as mock_factory:
            mock_factory.return_value.ainvoke = mock_pro
            rewrite_result = await query_rewriter(state)

        assert "Operating Systems" in rewrite_result["query"]
        assert rewrite_result["rewrite_count"] == 1
        assert rewrite_result["needs_rewrite"] is False
        state.update(rewrite_result)

        # ── Retriever again (same mocked docs) ─────────────────────────
        mock_vs = MagicMock()
        mock_vs.mmr_search = AsyncMock(return_value=_SAMPLE_DOCS)
        with patch("backend.graph.nodes.VectorStore", return_value=mock_vs):
            retriever_result = await retriever(state)
        state.update(retriever_result)

        # ── Grader: now approves ────────────────────────────────────────
        mock_flash2 = AsyncMock(return_value=_make_llm_response("0.80"))
        with patch("backend.graph.nodes._flash_llm") as mock_factory:
            mock_factory.return_value.ainvoke = mock_flash2
            grade_result2 = await relevance_grader(state)

        assert grade_result2["relevance_score"] == 0.80
        assert grade_result2["needs_rewrite"] is False
        state.update(grade_result2)

        # ── Edge should now route to generator ──────────────────────────
        assert after_relevance_grader(state) == "generator"


# ═══════════════════════════════════════════════════════════════════════════
# 3 · Fallback path — grader rejects twice, then fallback
# ═══════════════════════════════════════════════════════════════════════════


class TestFallbackPath:
    """Two rewrites exhausted → fallback message."""

    @pytest.mark.asyncio
    async def test_fallback_after_max_rewrites(self) -> None:
        """
        After 2 failed rewrite attempts, the edge routes to fallback.
        """
        state = _base_state(
            retrieved_docs=_SAMPLE_DOCS,
            intent="syllabus",
            rewrite_count=0,
        )

        # ── Rewrite loop iteration 1 ───────────────────────────────────
        mock_flash = AsyncMock(return_value=_make_llm_response("0.30"))
        with patch("backend.graph.nodes._flash_llm") as mock_factory:
            mock_factory.return_value.ainvoke = mock_flash
            grade1 = await relevance_grader(state)
        state.update(grade1)
        assert after_relevance_grader(state) == "query_rewriter"

        # Simulate rewriter incrementing count
        state["rewrite_count"] = 1
        state["needs_rewrite"] = False

        # ── Rewrite loop iteration 2 ───────────────────────────────────
        mock_flash2 = AsyncMock(return_value=_make_llm_response("0.25"))
        with patch("backend.graph.nodes._flash_llm") as mock_factory:
            mock_factory.return_value.ainvoke = mock_flash2
            grade2 = await relevance_grader(state)
        state.update(grade2)
        assert after_relevance_grader(state) == "query_rewriter"

        # Simulate rewriter incrementing count to max
        state["rewrite_count"] = 2
        state["needs_rewrite"] = True

        # ── Now edge should route to fallback ───────────────────────────
        assert after_relevance_grader(state) == "fallback"

        # ── Fallback node returns static message ────────────────────────
        fb = await fallback(state)
        assert fb["answer"] == _FALLBACK_MESSAGE
        assert isinstance(fb["messages"][0], AIMessage)

    @pytest.mark.asyncio
    async def test_fallback_on_low_score_no_rewrite_flag(self) -> None:
        """
        Even if needs_rewrite=False, a low score with exhausted rewrites
        still routes to fallback.
        """
        state = _base_state(
            relevance_score=0.2,
            needs_rewrite=False,
            rewrite_count=2,
        )
        assert after_relevance_grader(state) == "fallback"


# ═══════════════════════════════════════════════════════════════════════════
# 4 · Intent classification
# ═══════════════════════════════════════════════════════════════════════════


class TestIntentRouter:
    """Verify intent classification behaviour."""

    @pytest.mark.asyncio
    async def test_classifies_os_question_as_syllabus(self) -> None:
        """
        'What topics in OS?' should classify as 'syllabus'.
        """
        state = _base_state(query="What topics in OS?")

        mock_llm = AsyncMock(return_value=_make_llm_response("syllabus"))
        with patch("backend.graph.nodes._flash_llm") as mock_factory:
            mock_factory.return_value.ainvoke = mock_llm
            result = await intent_router(state)

        assert result["intent"] == "syllabus"

    @pytest.mark.asyncio
    async def test_classifies_attendance_question(self) -> None:
        """
        Attendance-related query should map to 'attendance'.
        """
        state = _base_state(query="What is my attendance percentage?")

        mock_llm = AsyncMock(return_value=_make_llm_response("attendance"))
        with patch("backend.graph.nodes._flash_llm") as mock_factory:
            mock_factory.return_value.ainvoke = mock_llm
            result = await intent_router(state)

        assert result["intent"] == "attendance"

    @pytest.mark.asyncio
    async def test_unknown_intent_defaults_to_general(self) -> None:
        """
        If the LLM returns gibberish, fall back to 'general'.
        """
        state = _base_state(query="abcdef random noise")

        mock_llm = AsyncMock(
            return_value=_make_llm_response("not_a_valid_intent_xyz"),
        )
        with patch("backend.graph.nodes._flash_llm") as mock_factory:
            mock_factory.return_value.ainvoke = mock_llm
            result = await intent_router(state)

        assert result["intent"] == "general"


# ═══════════════════════════════════════════════════════════════════════════
# 5 · Session memory — two queries with the same session_id share history
# ═══════════════════════════════════════════════════════════════════════════


class TestSessionMemory:
    """Verify multi-turn memory via the checkpointer."""

    @pytest.mark.asyncio
    async def test_same_session_shares_messages(self) -> None:
        """
        Two invocations on the same thread_id should accumulate messages
        in the checkpointed state.
        """
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        # Build a fresh in-memory checkpointer for the test
        # from_conn_string is an async context manager in v3
        async with AsyncSqliteSaver.from_conn_string(":memory:") as checkpointer:

            # Compile the graph with the test checkpointer
            from backend.graph.graph import workflow

            app = workflow.compile(checkpointer=checkpointer)

            session_id = "test-session-001"
            config = {"configurable": {"thread_id": session_id}}

            # ── Mock all nodes that do external calls ───────────────────
            mock_pro_gen = AsyncMock(
                return_value=_make_llm_response("OS covers processes."),
            )

            mock_vs = MagicMock()
            mock_vs.mmr_search = AsyncMock(return_value=_SAMPLE_DOCS)

            with (
                patch("backend.graph.nodes._flash_llm") as mock_flash_factory,
                patch("backend.graph.nodes._pro_llm") as mock_pro_factory,
                patch("backend.graph.nodes.VectorStore", return_value=mock_vs),
            ):
                # flash is used by intent_router and relevance_grader
                # We alternate: odd calls → intent, even → grader
                call_count = 0

                async def _flash_side_effect(messages):
                    nonlocal call_count
                    call_count += 1
                    if call_count % 2 == 1:
                        return _make_llm_response("syllabus")
                    return _make_llm_response("0.85")

                mock_flash_factory.return_value.ainvoke = AsyncMock(
                    side_effect=_flash_side_effect,
                )
                mock_pro_factory.return_value.ainvoke = mock_pro_gen

                # ── First query ─────────────────────────────────────────
                state1 = {
                    "messages": [HumanMessage(content="What topics in OS?")],
                    "query": "What topics in OS?",
                    "intent": "",
                    "retrieved_docs": [],
                    "relevance_score": 0.0,
                    "answer": "",
                    "sources": [],
                    "needs_rewrite": False,
                    "rewrite_count": 0,
                    "error": None,
                }
                result1 = await app.ainvoke(state1, config=config)

                assert result1["answer"] == "OS covers processes."
                # Messages should have: HumanMessage + AIMessage
                human_msgs = [
                    m for m in result1["messages"]
                    if isinstance(m, HumanMessage)
                ]
                ai_msgs = [
                    m for m in result1["messages"]
                    if isinstance(m, AIMessage)
                ]
                assert len(human_msgs) >= 1
                assert len(ai_msgs) >= 1

                # Reset mock call count for second query
                call_count = 0
                mock_pro_gen.return_value = _make_llm_response(
                    "Deadlocks involve circular wait."
                )

                # ── Second query (same session) ─────────────────────────
                state2 = {
                    "messages": [HumanMessage(content="Tell me about deadlocks")],
                    "query": "Tell me about deadlocks",
                    "intent": "",
                    "retrieved_docs": [],
                    "relevance_score": 0.0,
                    "answer": "",
                    "sources": [],
                    "needs_rewrite": False,
                    "rewrite_count": 0,
                    "error": None,
                }
                result2 = await app.ainvoke(state2, config=config)

                assert result2["answer"] == "Deadlocks involve circular wait."

                # Messages should now contain BOTH conversations (memory)
                all_human = [
                    m for m in result2["messages"]
                    if isinstance(m, HumanMessage)
                ]
                all_ai = [
                    m for m in result2["messages"]
                    if isinstance(m, AIMessage)
                ]
                # At least 2 human messages (one from each turn)
                assert len(all_human) >= 2
                # At least 2 AI messages (one from each turn)
                assert len(all_ai) >= 2


# ═══════════════════════════════════════════════════════════════════════════
# Edge routing — exhaustive unit tests
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeRouting:
    """Direct unit tests for ``after_relevance_grader``."""

    def test_high_score_goes_to_generator(self) -> None:
        """Score >= 0.6 and no rewrite flag → generator."""
        state = _base_state(
            relevance_score=0.9, needs_rewrite=False, rewrite_count=0,
        )
        assert after_relevance_grader(state) == "generator"

    def test_low_score_first_attempt_goes_to_rewriter(self) -> None:
        """Low score with rewrite budget → query_rewriter."""
        state = _base_state(
            relevance_score=0.3, needs_rewrite=True, rewrite_count=0,
        )
        assert after_relevance_grader(state) == "query_rewriter"

    def test_low_score_second_attempt_goes_to_rewriter(self) -> None:
        """Low score on rewrite attempt 1 → still rewrite."""
        state = _base_state(
            relevance_score=0.4, needs_rewrite=True, rewrite_count=1,
        )
        assert after_relevance_grader(state) == "query_rewriter"

    def test_low_score_max_rewrites_goes_to_fallback(self) -> None:
        """Rewrite budget exhausted → fallback."""
        state = _base_state(
            relevance_score=0.3, needs_rewrite=True, rewrite_count=2,
        )
        assert after_relevance_grader(state) == "fallback"

    def test_threshold_boundary_goes_to_generator(self) -> None:
        """Exactly 0.6 is above threshold → generator."""
        state = _base_state(
            relevance_score=0.6, needs_rewrite=False, rewrite_count=0,
        )
        assert after_relevance_grader(state) == "generator"

    def test_just_below_threshold_goes_to_rewriter(self) -> None:
        """0.59 is below threshold → rewriter if budget remains."""
        state = _base_state(
            relevance_score=0.59, needs_rewrite=True, rewrite_count=0,
        )
        assert after_relevance_grader(state) == "query_rewriter"
