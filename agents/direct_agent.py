from langchain_core.messages import AIMessage
from .base_agent import BaseAgent
from langgraph.graph import Graph, StateGraph, END, MessagesState
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver
from config.prompts import DIRECT_AGENT_PROMPT
import logging

class DirectAgent(BaseAgent):
    """Agent for handling general queries that don't require specialized knowledge."""
    
    def __init__(
        self,
        llm: BaseChatModel,
        checkpoint_saver: MemorySaver
    ):
        self.llm = llm
        self.checkpoint_saver = checkpoint_saver
        self.logger = logging.getLogger(__name__)
    
    def process_message(self, state: MessagesState) -> MessagesState:
        """Process a general query."""
        messages = state["messages"]
        current_message = messages[-1].content
        self.logger.debug("Direct agent processing: %s", current_message)
        
        # Create chat prompt for direct responses
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", DIRECT_AGENT_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ])
        
        # Get response from LLM with full message history
        chat_chain = chat_prompt | self.llm
        response = chat_chain.invoke({
            "question": current_message,
            "history": messages[:-1]  # Pass all previous messages as history
        })
        self.logger.debug("Direct agent response: %s", response.content)
        
        # Return updated state with new message
        return MessagesState(
            messages=[
                *messages,  # Keep existing messages
                AIMessage(content=response.content)  # Add our response
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