# College RAG Chatbot — Agent Instructions

## Project overview
RAG-based chatbot for a college. Features: PDF syllabus upload and Q&A,
attendance info, faculty/course search, exam prep assistant, timetable retrieval.

## Tech stack
- LangChain + LangGraph for orchestration
- Google Gemini via langchain-google-genai
  - gemini-2.5-pro for complex reasoning nodes
  - gemini-2.0-flash for fast/cheap nodes
- ChromaDB persisted to ./data/chroma
- FastAPI backend in ./backend/
- Streamlit frontend in ./frontend/
- PyMuPDF for PDF parsing
- BGE-small-en-v1.5 embeddings (runs locally on M2)

## Code rules
- Python 3.11 type hints everywhere
- async/await for all I/O
- Pydantic v2 models for all data shapes
- Never hardcode secrets — always os.getenv()
- Docstrings on every function
- Use uv to add packages, never pip directly
- ChatGoogleGenerativeAI from langchain_google_genai for all LLM calls
- GOOGLE_API_KEY from environment

## Architecture rules
- LangGraph StateGraph with TypedDict state
- One file per concern (nodes.py, edges.py, graph.py, state.py)
- All ChromaDB ops go through backend/rag/vectorstore.py
- All tests mock LLM and ChromaDB calls
