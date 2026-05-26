"""
Nodes for the LangGraph workflow.
"""

from backend.graph.state import OverallState

async def retrieve_node(state: OverallState) -> dict:
    """
    Retrieve node that queries the vectorstore for context.
    """
    return {}

async def generate_node(state: OverallState) -> dict:
    """
    Generate node that invokes the Gemini LLM model to generate response.
    """
    return {}
