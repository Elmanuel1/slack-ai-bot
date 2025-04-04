import logging
from typing import Dict, Any, List, Optional
from langgraph.graph import Graph, StateGraph, END, MessagesState, START
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .base_agent import BaseAgent
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver
from config.prompts import ROUTING_PROMPT

class MainAgent(BaseAgent):
    """Implementation of an agent using LangGraph for workflow management.
    
    This agent serves as the main orchestrator that routes messages to specialized
    sub-agents based on message content analysis. It uses a language model to
    determine which specialized agent should handle each message.
    
    The agent builds a graph with conditional routing to registered specialized agents.
    """
    
    def __init__(
        self, 
        llm: BaseChatModel,
        checkpoint_saver: MemorySaver,
        agents: List[BaseAgent],
        default_agent_key: Optional[str] = None
    ):
        """Initialize the MainAgent.
        
        Args:
            llm (BaseChatModel): The language model used for message routing.
            checkpoint_saver (MemorySaver): Checkpoint mechanism for saving state.
            agents (List[BaseAgent]): List of agent instances to be registered.
            default_agent_key (Optional[str]): Key of the default agent (fallback).
            
        Example:
            >>> agents = [
            ...     incident_agent,
            ...     knowledge_agent
            ... ]
            >>> agent = MainAgent(
            ...     llm=chat_model,
            ...     checkpoint_saver=memory_saver,
            ...     agents=agents,
            ...     default_agent_key="knowledge"
            ... )
        """
        self.llm = llm
        self.checkpoint_saver = checkpoint_saver
        self.logger = logging.getLogger(__name__)
        
        # Build workflows from agents
        self.workflows = {}
        for agent in agents:
            self.workflows[agent.key] = agent.build()
        
        # Set default workflow key (use first one if not specified)
        if default_agent_key and default_agent_key in self.workflows:
            self.default_workflow_key = default_agent_key
        elif agents:
            self.default_workflow_key = agents[0].key
        else:
            self.default_workflow_key = None
            self.logger.warning("No agents registered and no default set")
        
        # Initialize the graph
        self.graph = StateGraph(MessagesState)
    
    def get_response(self, state: MessagesState) -> str:
        """Get the response from the current state.
        
        Extracts the latest message content from the provided state.
        
        Args:
            state (MessagesState): The current state containing messages.
            
        Returns:
            str: The content of the last message in the state.
        """
        return state["messages"][-1].content
    
    async def main_llm_node(self, state: MessagesState) -> MessagesState:
        """Main LLM node that routes messages to appropriate agents.
        
        This function analyzes the current message and determines which
        specialized agent should handle it based on content analysis.
        
        Args:
            state (MessagesState): The current state containing messages.
            
        Returns:
            MessagesState: Updated state with routing decision.
        """
        messages = state["messages"]
        current_message = messages[-1].content
        self.logger.debug("Main agent received message: %s", current_message)
        
        # Create routing prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", ROUTING_PROMPT),
            ("human", "{input}")
        ])
        
        # Get routing decision
        chain = prompt | self.llm | StrOutputParser()
        result = await chain.ainvoke({
            "input": current_message
        })
        route = result.strip().lower()
        self.logger.debug("Raw routing decision: '%s'", route)
        
        # Determine next step - use the route if valid, otherwise default
        next_step = route if route in self.workflows else self.default_workflow_key
        self.logger.debug("Next step: %s", next_step)
        
        # Return state and routing decision
        return MessagesState(
            messages=messages,
            goto=next_step
        )
    
    def build(self) -> Graph:
        """Build and return a compiled graph for this agent.
        
        Constructs a LangGraph with nodes for the main routing agent and all
        specialized sub-agents. It configures the connections between nodes
        based on the routing decisions.
        
        Returns:
            Graph: A compiled LangGraph workflow ready for execution.
            
        Example:
            >>> workflow = agent.build()
            >>> result = await workflow.ainvoke({"messages": [HumanMessage(content="Help me")]})
        """
        # Add the main LLM node
        self.graph.add_node("main_agent", self.main_llm_node)
        
        # Add each workflow as a node
        for key, workflow in self.workflows.items():
            self.graph.add_node(key, workflow)
        
        # Add edge from START to main_agent
        self.graph.add_edge(START, "main_agent")
        
        # Add conditional edges from main_agent to each workflow
        # Create a mapping of possible destinations
        destinations = {key: key for key in self.workflows}
        destinations[END] = END  # Add END as a possible destination
        
        self.graph.add_conditional_edges(
            "main_agent",
            lambda x: x.get("goto", END),
            destinations
        )
        
        # Set the entry point
        self.graph.set_entry_point("main_agent")
        
        # Compile the graph with checkpointing
        self.workflow = self.graph.compile(
            checkpointer=self.checkpoint_saver
        )
        
        return self.workflow 