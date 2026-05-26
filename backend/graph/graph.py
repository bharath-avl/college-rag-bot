"""
Workflow graph compilation using LangGraph StateGraph.
"""

from langgraph.graph import StateGraph
from backend.graph.state import OverallState

# Define workflow graph builder
workflow = StateGraph(OverallState)
