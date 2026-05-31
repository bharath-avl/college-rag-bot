"""
Streamlit sidebar — branding, PDF upload, document management, feature pills.
"""

from __future__ import annotations

import os

import requests
import streamlit as st

# ── Backend URL ─────────────────────────────────────────────────────────────

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

# ── Intent colour map ──────────────────────────────────────────────────────

FEATURE_PILLS: list[tuple[str, str, str]] = [
    ("📚", "Syllabus Q&A", "#6C63FF"),
    ("📋", "Attendance", "#00C9A7"),
    ("👨‍🏫", "Faculty Search", "#F77F00"),
    ("📝", "Exam Prep", "#E63946"),
    ("🕐", "Timetable", "#3A86FF"),
]


def render_sidebar() -> None:
    """
    Render the full sidebar: branding, PDF uploader, indexed document list,
    feature pills, and optional LangSmith link.
    """
    with st.sidebar:
        # ── Branding ────────────────────────────────────────────────────
        st.markdown(
            """
            <div style="text-align:center; padding: 0.5rem 0 1rem 0;">
                <span style="font-size:2.8rem;">🎓</span>
                <h2 style="margin:0; font-weight:700; letter-spacing:-0.5px;">
                    CollegeBot
                </h2>
                <p style="margin:0; color:#888; font-size:0.85rem;">
                    AI-powered college assistant
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        # ── PDF Upload ──────────────────────────────────────────────────
        st.markdown("#### 📄 Upload Syllabi")
        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="pdf_uploader",
        )

        if uploaded_files:
            for uploaded_file in uploaded_files:
                # Avoid re-uploading files already processed in this session
                uploaded_key = f"uploaded_{uploaded_file.name}_{uploaded_file.size}"
                if uploaded_key in st.session_state:
                    continue

                with st.spinner(f"Indexing **{uploaded_file.name}** …"):
                    try:
                        resp = requests.post(
                            f"{API_BASE}/upload",
                            files={
                                "file": (
                                    uploaded_file.name,
                                    uploaded_file.getvalue(),
                                    "application/pdf",
                                )
                            },
                            timeout=120,
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            st.success(
                                f"✅ **{data['filename']}** — "
                                f"{data['chunks_indexed']} chunks indexed"
                            )
                            st.session_state[uploaded_key] = True
                        else:
                            detail = resp.json().get("detail", resp.text)
                            st.error(f"Upload failed: {detail}")
                    except requests.ConnectionError:
                        st.error("⚠️ Cannot reach backend. Is it running on :8000?")
                    except Exception as exc:
                        st.error(f"Upload error: {exc}")

        st.divider()

        # ── Indexed Documents ───────────────────────────────────────────
        st.markdown("#### 📂 Indexed Documents")

        try:
            resp = requests.get(f"{API_BASE}/documents", timeout=10)
            if resp.status_code == 200:
                docs = resp.json()
            else:
                docs = []
        except Exception:
            docs = []
            st.caption("_Backend offline — can't fetch documents_")

        if docs:
            for doc in docs:
                col_name, col_del = st.columns([4, 1])
                with col_name:
                    st.markdown(
                        f"<span style='font-size:0.85rem;'>"
                        f"📄 **{doc['filename']}** "
                        f"<span style='color:#888;'>({doc['pages']}p)</span>"
                        f"</span>",
                        unsafe_allow_html=True,
                    )
                with col_del:
                    if st.button(
                        "🗑️",
                        key=f"del_{doc['filename']}",
                        help=f"Delete {doc['filename']}",
                    ):
                        try:
                            del_resp = requests.delete(
                                f"{API_BASE}/documents/{doc['filename']}",
                                timeout=15,
                            )
                            if del_resp.status_code == 200:
                                st.toast(f"Deleted **{doc['filename']}**", icon="🗑️")
                                st.rerun()
                            else:
                                st.error("Delete failed")
                        except Exception as exc:
                            st.error(f"Error: {exc}")
        else:
            st.caption("_No documents indexed yet_")

        st.divider()

        # ── Feature Pills ──────────────────────────────────────────────
        st.markdown("#### ✨ Capabilities")
        pills_html = ""
        for emoji, label, colour in FEATURE_PILLS:
            pills_html += (
                f'<span style="display:inline-block; margin:3px 4px; '
                f"padding:4px 12px; border-radius:20px; font-size:0.78rem; "
                f"font-weight:500; background:{colour}22; color:{colour}; "
                f'border:1px solid {colour}44;">'
                f"{emoji} {label}</span>"
            )
        st.markdown(pills_html, unsafe_allow_html=True)

        st.divider()

        # ── LangSmith link ─────────────────────────────────────────────
        langsmith_project = os.getenv("LANGCHAIN_PROJECT", "")
        tracing_enabled = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
        if tracing_enabled and langsmith_project:
            st.markdown(
                f"🔗 [LangSmith Traces]"
                f"(https://smith.langchain.com/projects/{langsmith_project})",
            )
        else:
            st.caption("_LangSmith tracing disabled_")

        # ── Footer ─────────────────────────────────────────────────────
        st.markdown(
            """
            <div style="position:fixed; bottom:0; padding:8px 0; width:100%;
                        font-size:0.7rem; color:#666; text-align:center;">
                Built with LangGraph + Gemini + ChromaDB
            </div>
            """,
            unsafe_allow_html=True,
        )
