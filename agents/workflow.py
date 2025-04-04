from langgraph.graph import Graph

class AgentWorkflow:
    """Container for an agent workflow with its routing key.
    
    This class pairs a routing key with a compiled agent workflow graph,
    allowing for dynamic registration and lookup of specialized agents.
    """
    
    def __init__(self, key: str, workflow: Graph):
        """Initialize an agent workflow.
        
        Args:
            key: The routing key for this workflow
            workflow: The compiled graph for this agent
        """
        self.key = key
        self.workflow = workflow 