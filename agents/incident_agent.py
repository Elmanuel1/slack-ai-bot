from langchain_core.messages import AIMessage
from .base_agent import BaseAgent
from langgraph.graph import Graph, StateGraph, END, MessagesState
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from config.prompts import INCIDENT_AGENT_PROMPT
import logging

class IncidentAgent(BaseAgent):
    """Agent for handling incident-related tasks.
    
    This agent specializes in processing and responding to incident reports.
    It provides structured responses to help users manage and resolve incidents
    according to established procedures and best practices.
    
    The agent implements a simple workflow that processes incident reports and
    generates helpful responses.
    """
    
    def __init__(
        self,
        llm: BaseChatModel,
        checkpoint_saver: MemorySaver
    ):
        """Initialize the incident agent.
        
        Args:
            llm (BaseChatModel): Language model for generating responses.
            checkpoint_saver (MemorySaver): Checkpoint saver for state management.
            
        Example:
            >>> incident_agent = IncidentAgent(
            ...     llm=chat_model,
            ...     checkpoint_saver=memory_saver
            ... )
        """
        self.llm = llm
        self.checkpoint_saver = checkpoint_saver
        self.logger = logging.getLogger(__name__)
    
    def process_message(self, state: MessagesState) -> MessagesState:
        """Process an incident message.
        
        Analyzes the incident report in the input message and generates a
        response with guidance on how to handle the incident.
        
        Args:
            state (MessagesState): The current state containing messages.
            
        Returns:
            MessagesState: Updated state with the agent's response appended.
            
        Example:
            >>> updated_state = incident_agent.process_message(state)
            >>> response = updated_state["messages"][-1].content
        """
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
        """Build and return a compiled graph for this agent.
        
        Constructs a simple LangGraph with a single node for processing
        incident messages.
        
        Returns:
            Graph: A compiled LangGraph workflow ready for execution.
            
        Example:
            >>> workflow = incident_agent.build()
            >>> result = workflow.invoke({"messages": [HumanMessage(content="We have an outage")]})
        """
        graph = StateGraph(MessagesState)
        
        graph.set_entry_point("process")

        # Add the process_message node
        graph.add_node("process", self.process_message)
        
        # Set entry point and add edge to END
        graph.add_edge("process", END)
        
        # Compile with checkpointing
        return graph.compile(checkpointer=self.checkpoint_saver) 