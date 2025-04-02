from abc import ABC, abstractmethod
from typing import Dict, Protocol
from langgraph.graph import Graph, StateGraph, END, MessagesState

class GraphBuilder(Protocol):
    """Protocol for building subgraphs."""
    def build(self) -> Graph:
        """Build and return a compiled graph."""
        ...

class BaseAgent(ABC, GraphBuilder):
    """Base class for all agent implementations."""
    
    
    def get_response(self, state: MessagesState) -> str:
        """Get the response from the current state."""
        return state["messages"][-1].content