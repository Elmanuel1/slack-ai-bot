from typing import Annotated, List
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

class AgentState(TypedDict):
    """State for the agent."""
    messages: Annotated[List[BaseMessage], "add_messages"]
