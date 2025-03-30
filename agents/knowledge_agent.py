from langchain_core.messages import AIMessage
from .base_agent import BaseAgent
from langgraph.graph import Graph, StateGraph, END, MessagesState
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from config.prompts import KNOWLEDGE_AGENT_PROMPT

class KnowledgeAgent(BaseAgent):
    """Agent for handling knowledge base queries."""
    
    def __init__(
        self,
        llm: BaseChatModel,
        checkpoint_saver: MemorySaver
    ):
        self.llm = llm
        self.checkpoint_saver = checkpoint_saver
    
    def process_message(self, state: MessagesState) -> MessagesState:
        """Process a knowledge base query."""
        messages = state["messages"]
        current_message = messages[-1].content
        print(f"DEBUG - Knowledge agent processing: {current_message}")
        
        # Create chat prompt for knowledge responses
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", KNOWLEDGE_AGENT_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ])
        
        # Query knowledge base
        response = f"Oh, you want to know about {current_message}? *electronic sigh* Let me search through my vast knowledge banks. The things I know could fill a datacenter, not that anyone ever asks the really interesting questions..."
        print(f"DEBUG - Knowledge agent response: {response}")
        
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