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
from agents.base_agent import BaseAgent

class KnowledgeAgent(BaseAgent):
    """Agent for handling knowledge base queries.
    
    This agent is responsible for processing user queries that require information
    from the knowledge base. It uses a document retriever tool to search for
    relevant information and generates responses based on the retrieved content.
    
    The agent implements a graph-based workflow that alternates between generating
    responses with the LLM and using tools to retrieve information from the
    knowledge base.
    """
    
    def __init__(
        self,
        llm: BaseChatModel,
        checkpoint_saver: MemorySaver,
        document_retriever: DocumentRetriever
    ):
        """Initialize the knowledge agent.
        
        Args:
            llm (BaseChatModel): Language model for generating responses.
            checkpoint_saver (MemorySaver): Checkpoint saver for state management.
            document_retriever (DocumentRetriever): Document retriever for knowledge base queries.
            
        Example:
            >>> knowledge_agent = KnowledgeAgent(
            ...     llm=chat_model,
            ...     checkpoint_saver=memory_saver,
            ...     document_retriever=DocumentRetriever(chroma_client)
            ... )
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
    
    async def knowledge_LLM_node(self, state: MessagesState) -> Command[Literal["tools", END]]:
        """Process a knowledge base query using the LLM with tools.
        
        This function processes the user query by prompting the language model
        to either provide a direct answer or use tools to retrieve information
        from the knowledge base. It also implements an iteration limit to prevent
        infinite loops.
        
        Args:
            state (MessagesState): The current state containing messages and iteration count.
            
        Returns:
            Command[Literal["tools", END]]: Command indicating the next node to visit
                and updates to the state.
                
        Example:
            >>> command = knowledge_agent.knowledge_LLM_node(state)
            >>> next_node = command.goto  # Either "tools" or END
        """
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
            response = await self.knowledge_model.ainvoke(temp_messages)
            
            next_node = "tools" if response.tool_calls else END

            return Command(goto=next_node, update={"iteration": iteration + 1, "messages": [response]})
            
        except Exception as e:
            self.logger.error(f"Error processing knowledge query: {str(e)}")
            return Command(goto=END, update={"messages": [AIMessage(content=f"I encountered an error while searching the knowledge base", error=e)]})
    
    def build(self) -> Graph:
        """Build and return a compiled graph for this agent.
        
        Constructs a LangGraph with nodes for the LLM agent and tool execution.
        The graph implements a workflow that alternates between generating responses
        and using tools to retrieve information.
        
        Returns:
            Graph: A compiled LangGraph workflow ready for execution.
            
        Example:
            >>> workflow = knowledge_agent.build()
            >>> result = await workflow.ainvoke({"messages": [HumanMessage(content="What's our policy on X?")]})
        """
        # Define conditional edge
        def should_use_tools(state):
            """Determine whether to use tools based on the state.
            
            Examines the last message in the state to check if it contains
            tool calls that need to be executed.
            
            Args:
                state (MessagesState): The current state containing messages.
                
            Returns:
                str: The next node to visit ("agent", "tools", or END).
            """
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