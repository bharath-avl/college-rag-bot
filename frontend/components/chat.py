"""
Streamlit chat area — message history, SSE streaming, intent badges, sources.
"""

from __future__ import annotations

import json
import os

import requests
import sseclient
import streamlit as st

# ── Backend URL ─────────────────────────────────────────────────────────────

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

# ── Intent badge colours ───────────────────────────────────────────────────

INTENT_COLOURS: dict[str, tuple[str, str]] = {
    "syllabus": ("📚", "#6C63FF"),
    "attendance": ("📋", "#00C9A7"),
    "faculty": ("👨‍🏫", "#F77F00"),
    "exam_prep": ("📝", "#E63946"),
    "timetable": ("🕐", "#3A86FF"),
    "general": ("💬", "#888888"),
}


def _render_intent_badge(intent: str) -> None:
    """Display a coloured badge for the classified intent."""
    emoji, colour = INTENT_COLOURS.get(intent, ("💬", "#888"))
    st.markdown(
        f'<span style="display:inline-block; margin-top:6px; padding:3px 12px; '
        f"border-radius:14px; font-size:0.75rem; font-weight:600; "
        f'background:{colour}18; color:{colour}; border:1px solid {colour}44;">'
        f"{emoji} {intent.replace('_', ' ').title()}</span>",
        unsafe_allow_html=True,
    )


def _render_sources(sources: list[dict]) -> None:
    """Render a collapsible section listing citation sources."""
    if not sources:
        return
    with st.expander("📎 Sources", expanded=False):
        for i, src in enumerate(sources, 1):
            filename = src.get("source_file", src.get("filename", "unknown"))
            page = src.get("page_number", "?")
            st.markdown(f"**{i}.** `{filename}` — page {page}")


def _stream_chat_response(query: str, session_id: str) -> tuple[str, list[dict]]:
    """
    POST to /chat with SSE streaming, writing tokens into the current
    ``st.chat_message("assistant")`` context live.

    Returns:
        (full_answer, sources) collected from the stream.
    """
    full_answer = ""
    sources: list[dict] = []
    placeholder = st.empty()

    try:
        resp = requests.post(
            f"{API_BASE}/chat",
            json={"query": query, "session_id": session_id},
            stream=True,
            timeout=120,
            headers={"Accept": "text/event-stream"},
        )
        resp.raise_for_status()

        client = sseclient.SSEClient(resp)

        for event in client.events():
            try:
                data = json.loads(event.data)
            except json.JSONDecodeError:
                continue

            token = data.get("token", "")
            done = data.get("done", False)

            if token:
                full_answer += token
                placeholder.markdown(full_answer + "▌")

            if done:
                sources = data.get("sources", [])
                break

        # Remove cursor
        placeholder.markdown(full_answer)

    except requests.ConnectionError:
        placeholder.error("⚠️ Cannot reach backend. Is it running on :8000?")
    except Exception as exc:
        placeholder.error(f"Streaming error: {exc}")

    return full_answer, sources


def render_chat() -> None:
    """
    Render the main chat area: message history, input box, and streaming
    assistant responses.
    """
    # ── Header ──────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="padding-bottom: 0.5rem;">
            <h1 style="margin:0; font-weight:800; letter-spacing:-1px;">
                🎓 CollegeBot
            </h1>
            <p style="margin:0; color:#888; font-size:0.95rem;">
                Ask anything about your college — syllabi, attendance,
                faculty, exams, timetables.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Message history ─────────────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Re-render metadata on history replay
            if msg["role"] == "assistant":
                if msg.get("intent"):
                    _render_intent_badge(msg["intent"])
                if msg.get("sources"):
                    _render_sources(msg["sources"])

    # ── Chat input ──────────────────────────────────────────────────────
    if prompt := st.chat_input("Ask a question about your college…"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Stream assistant response
        with st.chat_message("assistant"):
            answer, sources = _stream_chat_response(
                prompt, st.session_state.session_id,
            )

            # Try to detect intent from the latest response
            # (The backend graph stores it in the streamed state; for now
            # we parse it from the answer or default to "general")
            intent = _detect_intent_from_answer(prompt)

            if intent:
                _render_intent_badge(intent)
            if sources:
                _render_sources(sources)

        # Persist to session state
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "intent": intent,
                "sources": sources,
            }
        )


def _detect_intent_from_answer(query: str) -> str:
    """
    Simple heuristic intent detection from the user query for badge display.

    In production, the intent comes from the LangGraph ``intent_router`` node.
    This is a fallback when the SSE stream doesn't include intent metadata.
    """
    q_lower = query.lower()
    keyword_map = {
        "syllabus": ["syllabus", "topic", "chapter", "module", "course", "subject"],
        "attendance": ["attendance", "absent", "present", "leave"],
        "faculty": ["faculty", "professor", "teacher", "instructor", "staff"],
        "exam_prep": ["exam", "test", "quiz", "preparation", "study", "marks"],
        "timetable": ["timetable", "schedule", "class time", "lecture time"],
    }
    for intent, keywords in keyword_map.items():
        if any(kw in q_lower for kw in keywords):
            return intent
    return "general"
