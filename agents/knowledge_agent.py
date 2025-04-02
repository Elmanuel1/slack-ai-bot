from typing import List, Dict, Any, Literal
import logging
from langchain_core.messages import AIMessage
from langchain_core.language_models import BaseChatModel
from langgraph.graph import Graph, StateGraph, END, MessagesState, START
from langgraph.checkpoint.memory import MemorySaver
from config.prompts import KNOWLEDGE_AGENT_PROMPT
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from documents.document_retriever import DocumentRetriever


class KnowledgeAgent:
    """Agent for handling knowledge base queries."""
    
    def __init__(
        self,
        llm: BaseChatModel,
        checkpoint_saver: MemorySaver,
        document_retriever: DocumentRetriever
    ):
        """Initialize the knowledge agent.
        
        Args:
            llm: Language model for generating responses
            checkpoint_saver: Checkpoint saver for state management
            document_retriever: Document retriever for knowledge base queries
        """
        self.llm = llm
        self.checkpoint_saver = checkpoint_saver
        self.document_retriever = document_retriever
        self.logger = logging.getLogger(__name__)
        
        # Get the document retrieval tool from the document retriever
        self.knowledge_tools = [self.document_retriever.get_tool()]
        self.knowledge_tool_node = ToolNode(self.knowledge_tools)
        
        # Bind tools to model
        self.knowledge_model = self.llm.bind_tools(self.knowledge_tools)
    
    def knowledge_LLM_node(self, state: MessagesState) -> Command[Literal["tools", END]]:
        """Process a knowledge base query using the LLM with tools."""
        messages = state['messages']
        iteration = state.get('iteration', 0)
        if iteration > 2:
            return Command(goto=END, update={"messages": [{"role": "system", "content": "Here's what I found in the knowledge base: "}]})
        current_message = messages[-1].content
        self.logger.debug(f"Processing knowledge query: {current_message}")
        
        try:
            # Create system prompt that requires tool usage
            temp_messages = [
                {"role": "system", "content": KNOWLEDGE_AGENT_PROMPT},
                *messages
            ]
            
            # Get response from model
            response = self.knowledge_model.invoke(temp_messages)
            
            next_node = "tools" if response.tool_calls else END

            return Command(goto=next_node, update={"iteration": iteration + 1, "messages": [response]})
            
        except Exception as e:
            self.logger.error(f"Error processing knowledge query: {str(e)}")
            return Command(goto=END, update={"messages": [AIMessage(content=f"I encountered an error while searching the knowledge base", error=e)]})
    
    def build(self) -> Graph:
        """Build and return a compiled graph for this agent."""
        # Define conditional edge
        def should_use_tools(state):
            messages = state["messages"]
            if not messages:
                return "agent"
            
            last_message = messages[-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"
            return END
        
        # Create graph
        graph = StateGraph(MessagesState)
        
        # Add nodes
        graph.add_node("agent", self.knowledge_LLM_node)
        graph.add_node("tools", self.knowledge_tool_node)
        
        # Add edges
        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", should_use_tools)
        graph.add_edge("tools", "agent")
        
        # Set entry point and compile with recursion limit
        graph.set_entry_point("agent")
        return graph.compile(
            checkpointer=self.checkpoint_saver
        )