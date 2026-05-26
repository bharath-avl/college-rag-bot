"""
Main entry point for the FastAPI backend.
"""

from fastapi import FastAPI

app = FastAPI(title="College RAG Chatbot API")

@app.get("/")
async def root() -> dict[str, str]:
    """
    Root endpoint to verify API health.
    """
    return {"status": "ok", "message": "College RAG Chatbot API is running"}
