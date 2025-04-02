from abc import ABC, abstractmethod
from typing import Dict, Protocol
from langgraph.graph import Graph, StateGraph, END, MessagesState

class GraphBuilder(Protocol):
    """Protocol for building subgraphs (Agents in a multi-agent system).
    
    This protocol defines the interface for classes that can build and return
    a compiled LangGraph graph. Any class implementing this protocol must
    provide a build method that returns a Graph object.
    """
    def build(self) -> Graph:
        """Build and return a compiled graph.
        
        Returns:
            Graph: A compiled LangGraph graph ready for execution.
        """
        ...

class BaseAgent(ABC, GraphBuilder):
    """Base class for all agent implementations.
    
    This abstract class serves as the foundation for all agent implementations
    in the system. It inherits from ABC (Abstract Base Class) for defining
    abstract methods and GraphBuilder protocol for ensuring all agents can
    build a graph.
    """
    
    def get_response(self, state: MessagesState) -> str:
        """Get the response from the current state.
        
        Extracts the latest message content from the provided state.
        
        Args:
            state (MessagesState): The current state containing messages.
            
        Returns:
            str: The content of the last message in the state.
            
        Example:
            >>> agent = ConcreteAgent()
            >>> response = agent.get_response(state_with_messages)
            >>> print(response)
            "This is the latest message content"
        """
        return state["messages"][-1].content