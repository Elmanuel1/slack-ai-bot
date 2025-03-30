from langchain_core.messages import AIMessage
from .base_agent import BaseAgent
from langgraph.graph import Graph, StateGraph, END, MessagesState
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from config.prompts import INCIDENT_AGENT_PROMPT
import logging

class IncidentAgent(BaseAgent):
    """Agent for handling incident-related tasks."""
    
    def __init__(
        self,
        llm: BaseChatModel,
        checkpoint_saver: MemorySaver
    ):
        self.llm = llm
        self.checkpoint_saver = checkpoint_saver
        self.logger = logging.getLogger(__name__)
    
    def process_message(self, state: MessagesState) -> MessagesState:
        """Process an incident message."""
        messages = state["messages"]
        current_message = messages[-1].content
        self.logger.debug("Incident agent processing: %s", current_message)
        
        # Create chat prompt for incident responses
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", INCIDENT_AGENT_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{incident}")
        ])
        
        # Process incident
        response = f"*sigh* Another incident to handle. Let me look into: {current_message}. Not that incidents ever end well, but I'll try my best."
        self.logger.debug("Incident agent response: %s", response)
        
        # Return updated state with new message
        return MessagesState(
            messages=[
                *messages,  # Keep existing messages
                AIMessage(content=response)  # Add our response
            ]
        )
        
    def build(self) -> Graph:
        """Build and return a compiled graph for this agent."""
        graph = StateGraph(MessagesState)
        
        graph.set_entry_point("process")

        # Add the process_message node
        graph.add_node("process", self.process_message)
        
        # Set entry point and add edge to END
        graph.add_edge("process", END)
        
        # Compile with checkpointing
        return graph.compile(checkpointer=self.checkpoint_saver) 