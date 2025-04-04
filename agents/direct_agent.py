from langchain_core.messages import AIMessage
from .base_agent import BaseAgent
from langgraph.graph import Graph, StateGraph, END, MessagesState
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver
from config.prompts import DIRECT_AGENT_PROMPT
import logging

class DirectAgent(BaseAgent):
    """Agent for handling general queries that don't require specialized knowledge.
    
    This agent is designed to respond to general user queries that don't fall into
    specialized categories like incidents or knowledge base queries. It uses a
    language model with a general-purpose prompt to generate helpful responses.
    
    The agent implements a simple workflow that processes general queries and
    generates contextually relevant responses based on conversation history.
    """
    
    def __init__(
        self,
        llm: BaseChatModel,
        checkpoint_saver: MemorySaver
    ):
        """Initialize the direct agent.
        
        Args:
            llm (BaseChatModel): Language model for generating responses.
            checkpoint_saver (MemorySaver): Checkpoint saver for state management.
            
        Example:
            >>> direct_agent = DirectAgent(
            ...     llm=chat_model,
            ...     checkpoint_saver=memory_saver
            ... )
        """
        self.llm = llm
        self.checkpoint_saver = checkpoint_saver
        self.logger = logging.getLogger(__name__)
    
    async def process_message(self, state: MessagesState) -> MessagesState:
        """Process a general message with the LLM.
        
        Takes an incoming user message and generates a helpful response
        using the language model directly (without tools).
        
        Args:
            state (MessagesState): The current state containing messages.
            
        Returns:
            MessagesState: Updated state with the agent's response appended.
            
        Example:
            >>> updated_state = await direct_agent.process_message(state)
            >>> response = updated_state["messages"][-1].content
        """
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
        response = await chat_chain.ainvoke({
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
        """Build and return a compiled graph for this agent.
        
        Constructs a simple LangGraph with a single node for processing
        general queries.
        
        Returns:
            Graph: A compiled LangGraph workflow ready for execution.
            
        Example:
            >>> workflow = direct_agent.build()
            >>> result = await workflow.ainvoke({"messages": [HumanMessage(content="Tell me a joke")]})
        """
        graph = StateGraph(MessagesState)
        
        graph.set_entry_point("process")

        # Add the process_message node
        graph.add_node("process", self.process_message)
        
        # Set entry point and add edge to END
        graph.add_edge("process", END)
        
        # Compile with checkpointing
        return graph.compile(checkpointer=self.checkpoint_saver) 