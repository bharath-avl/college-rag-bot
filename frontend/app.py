"""
CollegeBot — Streamlit chatbot UI for the College RAG system.

Launch with::

    uv run streamlit run frontend/app.py
"""

from __future__ import annotations

import uuid

import streamlit as st

from frontend.components.chat import render_chat
from frontend.components.sidebar import render_sidebar

# ═══════════════════════════════════════════════════════════════════════════
# Page config (must be first Streamlit call)
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="CollegeBot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════
# Custom CSS — premium dark-friendly styling
# ═══════════════════════════════════════════════════════════════════════════

st.markdown(
    """
    <style>
    /* ── Google Font ──────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Main content area ────────────────────────────────────────── */
    .stMainBlockContainer {
        max-width: 52rem;
        padding-top: 1.5rem;
    }

    /* ── Sidebar polish ───────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0E1117 0%, #161B22 100%);
        border-right: 1px solid #21262D;
    }
    section[data-testid="stSidebar"] .stDivider {
        border-color: #21262D;
    }

    /* ── Chat message bubbles ─────────────────────────────────────── */
    div[data-testid="stChatMessage"] {
        border-radius: 12px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        animation: fadeIn 0.25s ease-in-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(6px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* ── Chat input ───────────────────────────────────────────────── */
    div[data-testid="stChatInput"] textarea {
        border-radius: 12px !important;
        border: 1px solid #30363D !important;
        font-size: 0.95rem;
    }
    div[data-testid="stChatInput"] textarea:focus {
        border-color: #6C63FF !important;
        box-shadow: 0 0 0 2px #6C63FF33 !important;
    }

    /* ── File uploader ────────────────────────────────────────────── */
    section[data-testid="stFileUploader"] {
        border-radius: 10px;
    }

    /* ── Buttons ──────────────────────────────────────────────────── */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        font-size: 0.82rem;
        transition: all 0.15s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }

    /* ── Expander (Sources) ───────────────────────────────────────── */
    details[data-testid="stExpander"] {
        border-radius: 10px !important;
        border: 1px solid #21262D !important;
    }

    /* ── Toast / alerts ───────────────────────────────────────────── */
    div[data-testid="stToast"] {
        border-radius: 10px;
    }

    /* ── Scrollbar ────────────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb {
        background: #30363D;
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover { background: #484F58; }

    /* ── Hide Streamlit branding ──────────────────────────────────── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════════
# Session state initialisation
# ═══════════════════════════════════════════════════════════════════════════

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# ═══════════════════════════════════════════════════════════════════════════
# Render layout
# ═══════════════════════════════════════════════════════════════════════════

render_sidebar()
render_chat()
