# Agent Development Guidelines (Proposal)

## Executive Summary

This document proposes a standardized approach to developing AI agents within our Slack bot framework. The proposal builds upon our existing architecture while introducing asynchronous programming patterns to significantly improve performance and scalability.

Key recommendations:
- Maintain our current separation of concerns and agent structure
- Transition from synchronous to asynchronous processing
- Standardize a consistent pattern for adding new agent types
- Implement proper error handling and timeout management
- Use a list-based approach for workflow registration to improve modularity

This proposal aims to establish a set of best practices that will make our codebase more maintainable and allow us to handle more concurrent users efficiently.

## Overview

This document outlines our standardized approach to developing AI agents within our Slack bot framework. The architecture uses LangGraph for orchestrating agent workflows, Pydantic for configuration management, and embraces asynchronous programming patterns for optimal performance.

## Core Architecture

Our agent system follows these key design principles:

1. **Separation of Concerns**: Each agent handles a specific type of task
2. **Orchestration**: A central agent routes messages to specialized agents
3. **State Management**: The LangGraph framework manages agent state transitions
4. **Asynchronous Processing**: All handlers process requests concurrently using async/await
5. **Configurability**: Settings are loaded from multiple sources with clear precedence. This can also greatly enhance local testing
6. **Modularity**: Workflows are registered using a list-based approach with keys for identification

## Why Asynchronous Processing

We'll adopt asynchronous processing for several critical reasons:

### Understanding Python's Event Loop

Python's asyncio provides an event loop that allows concurrent execution without threading complexities:

```
┌───────────────────────────────────────────┐
│                Event Loop                 │
│                                           │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐│
│  │ Slack   │    │  LLM    │    │Document ││
│  │ Message │    │  Call   │    │ Lookup  ││
│  │ Handler │    │         │    │         ││
│  └─────────┘    └─────────┘    └─────────┘│
└───────────────────────────────────────────┘
```

When using synchronous code:
- A blocking LLM call stops all processing
- Other users must wait for the current request to complete
- This creates a bottleneck that limits throughput

With asynchronous code:
- When an LLM call is waiting for a response, the event loop handles other tasks
- Multiple user requests can be processed simultaneously
- The application remains responsive even during long-running operations

### Performance Benefits

In our AI application, we regularly encounter operations that involve waiting:
- LLM API calls (100ms-10s)
- Vector database searches (50-500ms)
- Slack API communication (50-200ms)

Async processing allows us to handle multiple user requests efficiently, effectively providing a form of concurrency without the complexity of thread management.

## Agent Architecture

### Base Classes

All agents inherit from the `BaseAgent` abstract class:

```python
class BaseAgent(ABC, GraphBuilder):
    """Base class for all agent implementations."""
    
    @abstractmethod
    def build(self) -> Graph:
        """Build and return a compiled graph."""
        pass
    
    def get_response(self, state: MessagesState) -> str:
        """Get the response from the current state."""
        return state["messages"][-1].content
        
    @property
    def key(self) -> str:
        """Return the unique key for this agent type."""
        # Default implementation uses the class name
        return self.__class__.__name__.lower()
```

### Main Agent Types

1. **MainAgent**: Orchestrates routing to specialized agents
2. **KnowledgeAgent**: Handles knowledge base queries
3. **IncidentAgent**: Processes incident reports

## Creating a New Agent

### Step 1: Create a New Agent Class

Create a new file in the `agents/` directory:

```python
# agents/my_custom_agent.py
from typing import Dict, Any, Literal
import logging
from langchain_core.messages import AIMessage
from langchain_core.language_models import BaseChatModel
from langgraph.graph import Graph, StateGraph, END, MessagesState, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from .base_agent import BaseAgent

class MyCustomAgent(BaseAgent):
    """Agent for handling custom task type."""
    
    def __init__(
        self,
        llm: BaseChatModel,
        checkpoint_saver: MemorySaver,
    ):
        self.llm = llm
        self.checkpoint_saver = checkpoint_saver
        self.logger = logging.getLogger(__name__)
    
    @property
    def key(self) -> str:
        """Return the unique key for this agent."""
        return "custom"
    
    async def process_message(self, state: MessagesState) -> MessagesState:
        """Process a message asynchronously."""
        messages = state["messages"]
        current_message = messages[-1].content
        self.logger.debug("Custom agent processing: %s", current_message)
        
        # Process with LLM asynchronously
        response = await self.llm.ainvoke([
            {"role": "system", "content": "You are a custom agent."},
            {"role": "user", "content": current_message}
        ])
        
        # Return updated state with new message
        return MessagesState(
            messages=[
                *messages,  # Keep existing messages
                AIMessage(content=response.content)  # Add response
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
```

### Step 2: Update the MainAgent

In this step, you don't need to modify the MainAgent class directly since it's designed to accept any list of workflows. Instead, you'll just need to register your new agent when initializing the MainAgent:

```python
# main.py (update)
from agents.main_agent import MainAgent, AgentWorkflow
from agents.incident_agent import IncidentAgent
from agents.knowledge_agent import KnowledgeAgent
from agents.my_custom_agent import MyCustomAgent

# Create individual agent instances
incident_agent = IncidentAgent(llm=llm, checkpoint_saver=memory_saver)
knowledge_agent = KnowledgeAgent(
    llm=llm, 
    checkpoint_saver=memory_saver, 
    document_retriever=DocumentRetriever(chroma_client)
)
custom_agent = MyCustomAgent(llm=llm, checkpoint_saver=memory_saver)

# Build workflows with their routing keys
workflows = [
    AgentWorkflow(key=incident_agent.key, workflow=incident_agent.build()),
    AgentWorkflow(key=knowledge_agent.key, workflow=knowledge_agent.build()),
    AgentWorkflow(key=custom_agent.key, workflow=custom_agent.build())
]

# Initialize the main agent with all workflows
main_workflow = MainAgent(
    llm=llm,
    checkpoint_saver=memory_saver,
    workflows=workflows,
    default_workflow_key="knowledge"  # Set knowledge as the default fallback
).build()

# Use in Slack event handler
slack_event_handler = SlackEventsHandler(
    settings,
    [AppMentionEventHandler(app, settings, main_workflow)],
    app
)
```

### Step 3: Update Application Configuration

After implementing your agent and registering it with the MainAgent, you'll need to make sure your application has the right configuration:

1. **Environment Variables**: Ensure any new agent-specific environment variables are documented and set.

```bash
# Example .env file additions
CUSTOM_AGENT_ENABLED=true
CUSTOM_AGENT_MODEL=gpt-4
```

2. **YAML Configuration**: Update your YAML configuration if needed for the new agent.

```yaml
# config.yaml
agents:
  custom:
    enabled: true
    model: gpt-4
    system_prompt: "You are a specialized custom agent that helps with..."
```

3. **Update Main App Startup**: Make sure your app startup logic includes the new agent correctly.

```python
# In your startup logic (e.g., in main.py)
if settings.agents.custom.enabled:
    # Only add the custom agent if it's enabled in configuration
    custom_agent = MyCustomAgent(llm=llm, checkpoint_saver=memory_saver)
    workflows.append(AgentWorkflow(key=custom_agent.key, workflow=custom_agent.build()))
```

4. **Update Routing Prompts**: If needed, update the routing prompts to include your new agent type.

```python
# In config/prompts.py
ROUTING_PROMPT = """
You are a message router. Your job is to analyze the user's message and determine which specialized agent should handle it.
Routes:
- "incident" for incident reports and issues
- "knowledge" for questions about documentation, processes, or general knowledge
- "custom" for specialized tasks related to <your custom agent purpose>

Return only one word: "incident", "knowledge", or "custom".
"""
```

### Step 4: Update Slack Event Handler

Make the Slack event handler asynchronous:

```python
# events/app_mention_handler.py
@app.event("app_mention")
async def handle_mention(event, say):
    user = event["user"]
    text = event["text"]
    thread_ts = event.get("thread_ts", event.get("ts"))
    
    try:
        # Acknowledge receipt immediately
        await say(text="I'm on it!", thread_ts=thread_ts)
        
        # Create initial state with the message
        initial_state = MessagesState(
            messages=[HumanMessage(content=text)]
        )
        
        # Create config from the event
        config = {
            "configurable": {
                "thread_id": thread_ts
            }
        }
        
        # Invoke the workflow asynchronously
        final_state = await self.workflow.ainvoke(
            initial_state,
            config
        )
        
        # Get the response from the final state
        response = final_state["messages"][-1].content
        
        # Send the response back to Slack in the same thread
        await say(text=response, thread_ts=thread_ts)
        
    except Exception as e:
        self.logger.error("Error processing message: %s", str(e))
        await say(text="I apologize, but I encountered an error processing your message.", thread_ts=thread_ts)
```

## Best Practices

1. **Always Use Async/Await**
   - All handlers and LLM calls should be asynchronous
   - Use `async def` for methods and `await` for async calls

2. **Error Handling**
   - Always wrap async operations in try/except blocks
   - Log errors and provide user-friendly messages

3. **Timeouts and Limits**
   - Set reasonable timeouts for all external API calls
   - Limit token usage and implement conversation length boundaries

4. **State Management**
   - Be deliberate about what you store in state
   - Use MessagesState for conversation history

5. **Agent Registration**
   - Each agent should define a unique `key` property
   - Use the `AgentWorkflow` class to register workflows with the main agent

6. **Testing**
   - Write unit tests for agent logic
   - Use mocks for LLM and API calls in tests

## Conclusion

This agent architecture provides a flexible, maintainable way to build AI applications. The list-based workflow registration approach makes it easy to add, remove, or modify agent capabilities without changing the core orchestration logic.

By following these guidelines, you can create new agents that seamlessly integrate with our existing system while maintaining optimal performance and code quality. The asynchronous approach ensures our application can handle multiple users simultaneously and remain responsive even under high load or when dealing with slow external APIs. 