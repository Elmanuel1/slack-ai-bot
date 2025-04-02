import logging
from typing import Dict, Any
from langgraph.graph import Graph, StateGraph, END, MessagesState, START
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .base_agent import BaseAgent
from .incident_agent import IncidentAgent
from .knowledge_agent import KnowledgeAgent
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver
#from .direct_agent import DirectAgent
from config.prompts import ROUTING_PROMPT

class LangGraphAgent(BaseAgent):
    """Implementation of an agent using LangGraph for workflow management."""
    
    def __init__(
        self, 
        llm: BaseChatModel,
        checkpoint_saver: MemorySaver,
        incident_workflow: Graph,
        knowledge_workflow: Graph,
        #direct_workflow: Graph
    ):
        self.llm = llm
        self.checkpoint_saver = checkpoint_saver
        self.incident_workflow = incident_workflow
        self.knowledge_workflow = knowledge_workflow
        #self.direct_workflow = direct_workflow
        self.logger = logging.getLogger(__name__)
        
        # Initialize the graph
        self.graph = StateGraph(MessagesState)
    
    def get_response(self, state: MessagesState) -> str:
        """Get the response from the current state."""
        return state["messages"][-1].content
    
    def build(self) -> Graph:
        """Build and return a compiled graph for this agent."""
        def main_llm_node(state: MessagesState) -> Dict[str, Any]:
            """Main LLM node that routes messages to appropriate agents."""
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
            route = chain.invoke({
                "input": current_message
            }).strip().lower()
            self.logger.debug("Raw routing decision: '%s'", route)
            
            # Parse routing decision - force to knowledge if not exact match
            if route == "incident":
                next_step = "incident"
            elif route == "knowledge":
                next_step = "knowledge"
            else:
                next_step = "knowledge"  # Default everything else to knowledge
            
            self.logger.debug("Next step: %s", next_step)
            
            # Return state and routing decision without adding routing message
            return MessagesState(
                messages=messages,  # Keep original messages without adding routing message
                goto=next_step
            )
        
        # Add the main LLM node
        self.graph.add_node("main_agent", main_llm_node)
        
        # Add compiled graphs from each agent
        self.graph.add_node("incident", self.incident_workflow)
        self.graph.add_node("knowledge", self.knowledge_workflow)
        #self.graph.add_node("direct", self.direct_workflow)
        
        # Add edge from START to main_agent
        self.graph.add_edge(START, "main_agent")
        
        # Add conditional edges from main_agent
        self.graph.add_conditional_edges(
            "main_agent",
            lambda x: x.get("goto", END),
            {
                "incident": "incident",
                "knowledge": "knowledge",
                #"direct": "direct",
                END: END
            }
        )
        
        # Set the entry point
        self.graph.set_entry_point("main_agent")
        
        # Compile the graph with checkpointing and timeout
        self.workflow = self.graph.compile(
            checkpointer=self.checkpoint_saver
        )
        
        return self.workflow 