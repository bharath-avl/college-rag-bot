"""
Edges and conditional routing logic for the LangGraph workflow.
"""

from backend.graph.state import OverallState

def should_retrieve(state: OverallState) -> str:
    """
    Decide whether to query vectorstore or answer directly.
    """
    return "retrieve"
