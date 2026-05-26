"""
State definition for the LangGraph workflow.
"""

from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage

class OverallState(TypedDict):
    """
    Represents the state of the chatbot conversation graph.
    """
    messages: list[BaseMessage]
    query: str
    context: list[str]
