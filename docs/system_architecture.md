# Slack AI Bot System Architecture Proposal

## 1. Introduction

This document proposes the architecture for a Slack AI bot system that shall process and respond to user messages through specialized AI agents. The system will utilize LangGraph for workflow management and LangChain for language model interactions, implementing an asynchronous processing pattern to ensure scalability and responsiveness.

## 2. System Overview

The Slack AI bot shall consist of the following key components:

1. **Agent Framework**: A hierarchical system of specialized agents
2. **Slack Integration**: Components for interfacing with the Slack API
3. **Knowledge Management**: Tools for document retrieval and knowledge base access
4. **Configuration System**: Settings management for application flexibility

## 3. Component Architecture

### 3.1 Agent Framework

The agent framework shall be implemented as a hierarchical system with the following components:

1. **BaseAgent**: An abstract base class that all specialized agents must implement. It shall:
   - Define the common interface for all agents
   - Provide utility methods for response handling
   - Include methods for graph construction

2. **MainAgent**: The orchestrator that shall:
   - Route incoming messages to appropriate specialized agents
   - Use an LLM to analyze message content and determine routing
   - Maintain a registry of available specialized agents
   - Fall back to a default agent when no suitable route is found

3. **Specialized Agents**:
   - **DirectAgent**: Shall handle general queries using the LLM
   - **IncidentAgent**: Shall process incident related queries with appropriate tone
   - **KnowledgeAgent**: Shall retrieve information from the knowledge base

### 3.2 Slack Integration

The Slack integration layer shall consist of:

1. **SlackEventsHandler**: The main entry point that shall:
   - Initialize the Slack Bolt application
   - Register event handlers
   - Start the application in either Socket Mode or Bolt App Mode

2. **EventHandler**: An abstract base class that shall:
   - Define the contract for event handling
   - Ensure modularity and separation of concerns

3. **AppMentionEventHandler**: A concrete handler that shall:
   - Process mentions of the bot in Slack channels
   - Convert Slack messages to LangChain message format
   - Route messages through the agent workflow
   - Format and return responses to the Slack thread

### 3.3 Knowledge Management

The knowledge management system shall include:

1. **DocumentRetriever**: A tool that shall:
   - Search a vector database for relevant documents
   - Format search results for consumption by agents
   - Integrate with the KnowledgeAgent

2. **Vector Store Integration**: The system shall:
   - Connect to a Chroma vector database
   - Store document embeddings for semantic search
   - Support retrieval of contextually relevant information

3. **Knowledge Base Loading Job**: The system shall include a job runner for loading documents into the vector database:
   - **Job Runner**: Shall orchestrate the document loading process
   - **Document Loaders**: Shall extract documents from various sources (e.g., Confluence)
   - **Document Processors**: Shall clean and format documents
   - **Document Splitters**: Shall split documents into appropriate chunks
   - **Vector Store Writer**: Shall create embeddings and store documents in the vector database

### 3.4 Configuration System

The configuration system shall:

1. Use Pydantic Settings for type-safe configuration
2. Support environment variable overrides
3. Provide defaults for development environments
4. Include settings for:
   - Slack API credentials
   - Language model configuration
   - Knowledge base connection details

## 4. Processing Flow

The message processing flow shall follow these steps:

1. A user mentions the bot in a Slack channel
2. The AppMentionEventHandler receives the event
3. The message is converted to a LangChain message format
4. The message is sent to the MainAgent
5. The MainAgent routes the message to the appropriate specialized agent
6. The specialized agent processes the message and generates a response
7. The response is sent back to the Slack thread

## 5. Asynchronous Processing

The system shall implement asynchronous patterns to ensure responsiveness:

1. All agent processing shall use `async/await` patterns
2. LangModel invocations shall use `ainvoke` instead of `invoke`
3. Slack Bolt shall be configured in async mode
4. Document retrieval shall be asynchronous to prevent blocking

## 6. Testing Strategy

The system shall include comprehensive testing:

1. Unit tests for each agent type
2. Integration tests for agent interactions
3. Mock objects for external dependencies
4. Asynchronous test methods for async components

## 7. Implementation Guidelines

The implementation shall follow these guidelines:

1. All agents shall inherit from BaseAgent
2. Workflows shall be built using LangGraph
3. Document retrieval shall use vector search capabilities
4. The Slack integration shall support both development (Socket Mode) and production (Bolt App Mode) environments
5. Error handling shall be consistent and provide fallback responses

## 8. Conclusion

This architecture provides a flexible, maintainable, and scalable solution for a Slack AI bot system. The agent-based approach allows for specialized handling of different query types, while the asynchronous processing model ensures the system can handle multiple concurrent users without performance degradation.

The proposed architecture will deliver a responsive, intelligent assistant that can answer questions, handle incidents, and retrieve knowledge base information within Slack conversations.

## 9. Sample Code Implementation

### 9.1 BaseAgent Implementation

```python
from abc import ABC, abstractmethod
from typing import Any, Dict
from langgraph.graph import Graph, MessagesState
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver

class BaseAgent(ABC):
    """Base class for all agents in the system.
    
    All specialized agents must inherit from this class and implement
    the required methods. It provides a common interface and utility methods.
    """
    
    @property
    def key(self) -> str:
        """Return the unique identifier for this agent type."""
        return self.__class__.__name__.lower().replace("agent", "")
    
    def get_response(self, state: MessagesState) -> str:
        """Get the response from the current state.
        
        Args:
            state (MessagesState): The current state containing messages.
            
        Returns:
            str: The content of the last message in the state.
        """
        return state["messages"][-1].content
    
    @abstractmethod
    def build(self) -> Graph:
        """Build and return a compiled graph for this agent.
        
        Returns:
            Graph: A compiled LangGraph workflow ready for execution.
        """
        pass
```

### 9.2 MainAgent Implementation

```python
import logging
from typing import List, Optional
from langgraph.graph import Graph, StateGraph, END, MessagesState, START
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .base_agent import BaseAgent

class MainAgent(BaseAgent):
    """Orchestrator that routes messages to specialized agents.
    
    This agent analyzes message content and determines which specialized
    agent should handle each message.
    """
    
    def __init__(
        self, 
        llm,
        checkpoint_saver,
        agents: List[BaseAgent],
        default_agent_key: Optional[str] = None
    ):
        """Initialize the MainAgent.
        
        Args:
            llm: The language model used for message routing.
            checkpoint_saver: Checkpoint mechanism for saving state.
            agents: List of agent instances to be registered.
            default_agent_key: Key of the default agent (fallback).
        """
        self.llm = llm
        self.checkpoint_saver = checkpoint_saver
        self.logger = logging.getLogger(__name__)
        
        # Build workflows from agents
        self.workflows = {}
        for agent in agents:
            self.workflows[agent.key] = agent.build()
        
        # Set default workflow key
        if default_agent_key and default_agent_key in self.workflows:
            self.default_workflow_key = default_agent_key
        elif agents:
            self.default_workflow_key = agents[0].key
        else:
            self.default_workflow_key = None
            
        # Initialize the graph
        self.graph = StateGraph(MessagesState)
    
    async def main_llm_node(self, state: MessagesState) -> MessagesState:
        """Route messages to appropriate agents."""
        messages = state["messages"]
        current_message = messages[-1].content
        
        # Create routing prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a router. Analyze the message and determine which agent should handle it."),
            ("human", "{input}")
        ])
        
        # Get routing decision
        chain = prompt | self.llm | StrOutputParser()
        result = await chain.ainvoke({
            "input": current_message
        })
        route = result.strip().lower()
        
        # Determine next step
        next_step = route if route in self.workflows else self.default_workflow_key
        
        # Return state with routing decision
        return MessagesState(
            messages=messages,
            goto=next_step
        )
    
    def build(self) -> Graph:
        """Build the workflow graph."""
        # Add the main LLM node
        self.graph.add_node("main_agent", self.main_llm_node)
        
        # Add each workflow as a node
        for key, workflow in self.workflows.items():
            self.graph.add_node(key, workflow)
        
        # Add conditional edges
        destinations = {key: key for key in self.workflows}
        destinations[END] = END
        
        self.graph.add_conditional_edges(
            "main_agent",
            lambda x: x.get("goto", END),
            destinations
        )
        
        # Set entry point and compile
        self.graph.set_entry_point("main_agent")
        return self.graph.compile(checkpointer=self.checkpoint_saver)
```

### 9.3 DirectAgent Implementation

```python
import logging
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import Graph, StateGraph, END, MessagesState
from .base_agent import BaseAgent

class DirectAgent(BaseAgent):
    """Agent for handling general queries.
    
    This agent processes general queries by invoking the language model directly
    with the user's message and returning the response.
    """
    
    def __init__(self, llm, checkpoint_saver):
        """Initialize the DirectAgent.
        
        Args:
            llm: The language model to use for generating responses.
            checkpoint_saver: Checkpoint mechanism for saving state.
        """
        self.llm = llm
        self.checkpoint_saver = checkpoint_saver
        self.logger = logging.getLogger(__name__)
    
    async def process_message(self, state: MessagesState) -> MessagesState:
        """Process a message using the language model.
        
        Args:
            state: The current state containing messages.
            
        Returns:
            Updated state with the agent's response appended.
        """
        messages = state["messages"]
        current_message = messages[-1].content
        
        # Create chat prompt for direct responses
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ])
        
        # Get response from LLM with full message history
        chat_chain = chat_prompt | self.llm
        response = await chat_chain.ainvoke({
            "question": current_message,
            "history": messages[:-1]  # Pass all previous messages as history
        })
        
        # Return updated state with new message
        return MessagesState(
            messages=[
                *messages,  # Keep existing messages
                AIMessage(content=response.content)  # Add our response
            ]
        )
    
    def build(self) -> Graph:
        """Build a graph for this agent."""
        graph = StateGraph(MessagesState)
        
        graph.set_entry_point("process")
        graph.add_node("process", self.process_message)
        graph.add_edge("process", END)
        
        return graph.compile(checkpointer=self.checkpoint_saver)
```

### 9.4 KnowledgeAgent Implementation

```python
import logging
from typing import Literal
from langchain_core.messages import AIMessage
from langgraph.graph import Graph, StateGraph, END, MessagesState, START
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import Command
from .base_agent import BaseAgent

class KnowledgeAgent(BaseAgent):
    """Agent for knowledge retrieval.
    
    This agent uses tools to search for information in a knowledge base
    and return relevant answers based on retrieved documents.
    """
    
    def __init__(self, llm, checkpoint_saver, document_retriever):
        """Initialize the KnowledgeAgent.
        
        Args:
            llm: The language model to use for generating responses.
            checkpoint_saver: Checkpoint mechanism for saving state.
            document_retriever: Tool for retrieving documents from the knowledge base.
        """
        self.llm = llm
        self.checkpoint_saver = checkpoint_saver
        self.logger = logging.getLogger(__name__)
        self.document_retriever = document_retriever
        
        # Get the document retrieval tool
        self.knowledge_tools = [self.document_retriever.get_tool()]
        self.knowledge_tool_node = ToolNode(self.knowledge_tools)
        
        # Bind tools to model
        self.knowledge_model = self.llm.bind_tools(self.knowledge_tools)
    
    async def knowledge_LLM_node(self, state: MessagesState) -> Command[Literal["tools", END]]:
        """Process a knowledge base query.
        
        Args:
            state: The current state containing messages.
            
        Returns:
            Command indicating the next node to visit.
        """
        messages = state['messages']
        iteration = state.get('iteration', 0)
        
        if iteration > 2:
            return Command(goto=END, update={"messages": [{"role": "system", "content": "Here's what I found in the knowledge base: "}]})
        
        current_message = messages[-1].content
        
        try:
            # Create system prompt that requires tool usage
            temp_messages = [
                {"role": "system", "content": "You are a knowledge retrieval assistant. Use the retrieve_documents tool to search for information."},
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
        """Build a graph for this agent."""
        graph = StateGraph(MessagesState)
        
        # Add nodes
        graph.add_node("agent", self.knowledge_LLM_node)
        graph.add_node("tools", self.knowledge_tool_node)
        
        # Add edges
        graph.add_edge(START, "agent")
        
        # Define conditional edge
        def should_use_tools(state):
            messages = state["messages"]
            if not messages:
                return "agent"
            
            last_message = messages[-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"
            return END
        
        graph.add_conditional_edges("agent", should_use_tools)
        graph.add_edge("tools", "agent")
        
        # Set entry point and compile
        graph.set_entry_point("agent")
        return graph.compile(checkpointer=self.checkpoint_saver)
```

### 9.5 SlackEventsHandler Implementation

```python
import logging
import asyncio
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from events.event_handler import EventHandler
from config.settings import Settings

class SlackEventsHandler:
    """Manages Slack event handling and bot initialization."""
    
    def __init__(self, settings: Settings, event_handlers: list[EventHandler], app: AsyncApp):
        """Initialize the Slack events handler.
        
        Args:
            settings: Configuration settings for the Slack bot.
            event_handlers: List of specialized event handlers.
            app: The Slack Bolt async app instance.
        """
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        self.app = app
        self.event_handlers = event_handlers

    async def setup_events(self):
        """Define Slack event listeners."""
        for handler in self.event_handlers:
            handler.handle()

    async def _start_socket_mode(self):
        """Start the bot in Socket Mode."""
        self.logger.info("Starting Slack bot in Socket Mode")
        handler = AsyncSocketModeHandler(self.app, self.settings.slack.app_token)
        await handler.start_async()
        
    async def _start_bolt_app(self):
        """Start the bot in Bolt App Mode."""
        self.logger.info("Starting Slack bot in Bolt App Mode")
        await self.app.start_async(port=self.settings.slack.port)

    def start(self):
        """Start the Slack bot."""
        # Create and get the event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Define the main async function
        async def main():
            # Setup events
            await self.setup_events()
            
            # Start the bot in the appropriate mode
            if self.settings.slack.mode == "socket":
                await self._start_socket_mode()
            else:
                await self._start_bolt_app()
        
        # Run the async main function until complete
        loop.run_until_complete(main())
```

### 9.6 AppMentionEventHandler Implementation

```python
import logging
from events.event_handler import EventHandler
from slack_bolt.async_app import AsyncApp
from config.settings import Settings
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState, Graph

class AppMentionEventHandler(EventHandler):
    """Handler for Slack app_mention events."""
    
    def __init__(self, app: AsyncApp, settings: Settings, workflow: Graph):
        """Initialize the app mention event handler.
        
        Args:
            app: The Slack Bolt async app instance.
            settings: Configuration settings.
            workflow: The compiled workflow to process messages.
        """
        self.logger = logging.getLogger(__name__)
        self.app = app
        self.settings = settings
        self.workflow = workflow

    def handle(self):
        """Set up the Slack event handler."""
        @self.app.event("app_mention")
        async def handle_mention(event, say):
            """Process a Slack app_mention event."""
            user = event["user"]
            text = event["text"]
            thread_ts = event.get("thread_ts", event.get("ts"))
            
            try:
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
                await say(text="I apologize, but I encountered an error processing your message. Please try again later.", thread_ts=thread_ts)
```

### 9.7 DocumentRetriever Implementation

```python
from typing import List, Dict, Any
import logging
from langchain.vectorstores.base import VectorStore
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

class RetrieveDocumentsInput(BaseModel):
    """Input schema for document retrieval."""
    query: str = Field(..., description="The question to search for in the knowledge base")

class DocumentRetriever:
    """A class for retrieving relevant documents from a vector store."""

    def __init__(self, vectorstore: VectorStore, k: int = 1):
        """Initialize the DocumentRetriever.

        Args:
            vectorstore: The vector store containing document embeddings.
            k: Number of documents to retrieve for each query.
        """
        self.vectorstore = vectorstore
        self.logger = logging.getLogger(__name__)
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    async def retrieve_documents(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve relevant documents based on a query.

        Args:
            query: The search query string.

        Returns:
            A list of document dictionaries.
        """
        try:
            self.logger.debug(f"Retrieving Query: {query}")

            # Get documents from the retriever
            docs = await self.retriever.ainvoke(query)
            
            # Format the documents for return   
            results = []
            for i, doc in enumerate(docs):
                doc_dict = {
                    "content": doc.page_content,
                    "index": i,
                    "metadata": doc.metadata
                }
                
                # Add metadata if available
                if hasattr(doc, "metadata") and doc.metadata:
                    if "source" in doc.metadata:
                        doc_dict["source"] = doc.metadata["source"]
                    if "title" in doc.metadata:
                        doc_dict["title"] = doc.metadata["title"]
                
                results.append(doc_dict)
                
            return results
        except Exception as e:
            self.logger.error(f"Error retrieving documents: {str(e)}")
            return []
            
    def get_tool(self) -> StructuredTool:
        """Return a structured tool for document retrieval."""
        # Create a sync wrapper for the async retrieve_documents function
        async def retrieve_documents_wrapper(query: str) -> List[Dict[str, Any]]:
            return await self.retrieve_documents(query)
            
        return StructuredTool.from_function(
            func=retrieve_documents_wrapper,
            name="retrieve_documents",
            description="Retrieve relevant documents from the knowledge base",
            args_schema=RetrieveDocumentsInput,
            coroutine=retrieve_documents_wrapper,
            return_direct=False
        )
```

### 9.8 Knowledge Base Loading Components

```python
from typing import List, Optional
import logging
from langchain_core.documents import Document
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel
from config.settings import Settings

class ConfluenceDocument(BaseModel):
    """Model representing a document from Confluence."""
    id: str
    title: str
    content: str
    url: str
    version: Optional[int] = None
    space_key: str

class DocumentProcessor:
    """Processes documents from Confluence.
    
    This class is responsible for processing raw Confluence documents
    and converting them into a format suitable for vector storage.
    """
    
    def __init__(self):
        """Initialize the document processor."""
        self.logger = logging.getLogger(__name__)
    
    def process_document(self, document: dict) -> Optional[ConfluenceDocument]:
        """Process a document from Confluence.
        
        This method extracts the relevant information from a Confluence
        document and creates a structured document object.
        
        Args:
            document: Raw document data from Confluence API.
            
        Returns:
            Processed document object or None if processing fails.
        """
        try:
            # Extract the content from the body
            if "body" not in document or "storage" not in document["body"]:
                self.logger.error(f"Missing body or storage in document: {document['id']}")
                return None
                
            content = document["body"]["storage"]["value"]
            
            # Extract metadata
            doc_id = document["id"]
            title = document["title"]
            space_key = document["space"]["key"]
            url = f"https://confluence.example.com/display/{space_key}/{doc_id}"
            
            # Get version information if available
            version = document.get("version", {}).get("number", None)
            
            # Create structured document
            return ConfluenceDocument(
                id=doc_id,
                title=title,
                content=content,
                url=url,
                version=version,
                space_key=space_key
            )
            
        except Exception as e:
            self.logger.error(f"Error processing document: {str(e)}")
            return None

class TokenDocumentSplitter:
    """Splits documents into chunks based on token count.
    
    This class is responsible for splitting larger documents into
    smaller chunks that can be processed effectively by the language model.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize the document splitter.
        
        Args:
            chunk_size: Maximum size of each chunk in tokens.
            chunk_overlap: Number of tokens to overlap between chunks.
        """
        self.logger = logging.getLogger(__name__)
        self.splitter = TokenTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
    
    def split_document(self, doc: ConfluenceDocument) -> List[Document]:
        """Split a document into chunks.
        
        This method takes a document and splits its content into
        smaller chunks while preserving metadata.
        
        Args:
            doc: Document to split.
            
        Returns:
            List of document chunks with metadata.
        """
        try:
            if not doc.content or doc.content.strip() == "":
                self.logger.warning(f"Empty content in document: {doc.id}")
                return []
                
            # Split the text content
            texts = self.splitter.split_text(doc.content)
            
            # Create LangChain documents with metadata
            docs = []
            for i, text in enumerate(texts):
                metadata = {
                    "source": doc.url,
                    "title": doc.title,
                    "doc_id": doc.id,
                    "chunk": i,
                    "space_key": doc.space_key,
                    "version": doc.version
                }
                docs.append(Document(page_content=text, metadata=metadata))
                
            return docs
            
        except Exception as e:
            self.logger.error(f"Error splitting document: {str(e)}")
            return []

class ConfluenceLoader:
    """Loads documents from Confluence.
    
    This class is responsible for connecting to Confluence,
    retrieving documents, and processing them for storage.
    """
    
    def __init__(self, settings: Settings, processor: DocumentProcessor, batch_size: int = 25):
        """Initialize the Confluence loader.
        
        Args:
            settings: Application settings.
            processor: Document processor instance.
            batch_size: Number of documents to process in each batch.
        """
        self.settings = settings
        self.processor = processor
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)
        
        # Initialize Confluence client
        from atlassian import Confluence
        self.client = Confluence(
            url=settings.confluence.url,
            username=settings.confluence.username,
            password=settings.confluence.api_token
        )
    
    async def load_documents(self, space_key: str = None) -> List[ConfluenceDocument]:
        """Load documents from Confluence.
        
        This method retrieves all documents from a Confluence space,
        processes them, and returns a list of structured documents.
        
        Args:
            space_key: Key of the Confluence space to load documents from.
                If not provided, uses the space key from settings.
                
        Returns:
            List of processed documents.
        """
        try:
            # Use provided space key or default from settings
            space_key = space_key or self.settings.confluence.space_key
            
            if not space_key:
                self.logger.error("No space key provided")
                return []
                
            self.logger.info(f"Loading documents from Confluence space: {space_key}")
            
            # Get all content from the space
            start = 0
            limit = self.batch_size
            all_content = []
            
            # Paginate through results
            while True:
                self.logger.debug(f"Fetching content batch: start={start}, limit={limit}")
                batch = self.client.get_all_pages_from_space(
                    space=space_key,
                    start=start,
                    limit=limit,
                    expand="body.storage,version"
                )
                
                if not batch:
                    break
                    
                all_content.extend(batch)
                
                if len(batch) < limit:
                    break
                    
                start += limit
            
            self.logger.info(f"Retrieved {len(all_content)} documents from Confluence")
            
            # Process all documents
            processed_docs = []
            for content in all_content:
                doc = self.processor.process_document(content)
                if doc:
                    processed_docs.append(doc)
            
            self.logger.info(f"Successfully processed {len(processed_docs)} documents")
            return processed_docs
            
        except Exception as e:
            self.logger.error(f"Error loading documents from Confluence: {str(e)}")
            return []

class ChromaWriter:
    """Writes documents to a Chroma vector database.
    
    This class is responsible for creating embeddings for document chunks
    and storing them in the Chroma vector database.
    """
    
    def __init__(self, settings: Settings, chroma_client: Chroma, splitter: TokenDocumentSplitter):
        """Initialize the Chroma writer.
        
        Args:
            settings: Application settings.
            chroma_client: Chroma database client.
            splitter: Document splitter for chunking.
        """
        self.settings = settings
        self.chroma_client = chroma_client
        self.splitter = splitter
        self.logger = logging.getLogger(__name__)
    
    async def write_documents(self, documents: List[ConfluenceDocument]) -> int:
        """Write documents to Chroma.
        
        This method takes processed documents, splits them into chunks,
        and stores them in the Chroma vector database.
        
        Args:
            documents: List of documents to store.
            
        Returns:
            Number of chunks successfully stored.
        """
        try:
            if not documents:
                self.logger.info("No documents to write")
                return 0
                
            self.logger.info(f"Writing {len(documents)} documents to Chroma")
            
            # Split all documents into chunks
            all_chunks = []
            for doc in documents:
                chunks = self.splitter.split_document(doc)
                all_chunks.extend(chunks)
                
            if not all_chunks:
                self.logger.warning("No chunks generated from documents")
                return 0
                
            self.logger.info(f"Created {len(all_chunks)} chunks for indexing")
            
            # Add documents to Chroma
            self.chroma_client.add_documents(all_chunks)
            
            self.logger.info(f"Successfully wrote {len(all_chunks)} chunks to Chroma")
            return len(all_chunks)
            
        except Exception as e:
            self.logger.error(f"Error writing documents to Chroma: {str(e)}")
            return 0

class KnowledgeBaseJobRunner:
    """Runs knowledge base indexing jobs.
    
    This class orchestrates the process of loading documents from
    sources, processing them, and storing them in the vector database.
    """
    
    def __init__(self, settings: Settings, loader, writer):
        """Initialize the job runner.
        
        Args:
            settings: Application settings.
            loader: Document loader instance.
            writer: Vector store writer instance.
        """
        self.settings = settings
        self.loader = loader
        self.writer = writer
        self.logger = logging.getLogger(__name__)
    
    async def run(self, space_key: str = None) -> bool:
        """Run the knowledge base indexing job.
        
        This method orchestrates the loading, processing, and storing
        of documents in the vector database.
        
        Args:
            space_key: Key of the space to process. If not provided,
                uses the space key from settings.
                
        Returns:
            True if the job completes successfully, False otherwise.
        """
        try:
            self.logger.info("Starting knowledge base indexing job")
            
            # Load documents from source
            documents = await self.loader.load_documents(space_key)
            
            if not documents:
                self.logger.warning("No documents loaded")
                return False
                
            self.logger.info(f"Loaded {len(documents)} documents")
            
            # Write documents to vector store
            chunks_written = await self.writer.write_documents(documents)
            
            self.logger.info(f"Job completed successfully, wrote {chunks_written} chunks")
            return chunks_written > 0
            
        except Exception as e:
            self.logger.error(f"Error running knowledge base job: {str(e)}")
            return False
    
    @classmethod
    def create_confluence_job(cls, settings: Settings, chroma_client: Chroma) -> 'KnowledgeBaseJobRunner':
        """Create a job for indexing Confluence documents.
        
        This factory method creates a complete job runner with all
        necessary components configured for Confluence.
        
        Args:
            settings: Application settings.
            chroma_client: Chroma database client.
            
        Returns:
            Configured job runner instance.
        """
        processor = DocumentProcessor()
        splitter = TokenDocumentSplitter(
            chunk_size=settings.knowledge_base.chunk_size,
            chunk_overlap=settings.knowledge_base.chunk_overlap
        )
        loader = ConfluenceLoader(settings, processor)
        writer = ChromaWriter(settings, chroma_client, splitter)
        
        return cls(settings, loader, writer)

### 9.9 Main Application Entry Point

```python
import logging
import os
import asyncio
from config.settings import Settings
from events.app_mention_handler import AppMentionEventHandler
from slack.slack_events_handlers import SlackEventsHandler
from slack_bolt.async_app import AsyncApp
from utils.llm import init_chat_model
from agents.main_agent import MainAgent
from agents.incident_agent import IncidentAgent
from agents.knowledge_agent import KnowledgeAgent
from agents.direct_agent import DirectAgent
from langgraph.checkpoint.memory import MemorySaver
from documents.document_retriever import DocumentRetriever
from knowledge_base.job_runner import KnowledgeBaseJobRunner
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from config.logging import Logger

async def load_knowledge_base(settings, chroma_client):
    """Load the knowledge base with documents from Confluence."""
    # Create and run a Confluence job
    job = KnowledgeBaseJobRunner.create_confluence_job(settings, chroma_client)
    success = await job.run()
    
    if success:
        logging.info("Knowledge base successfully loaded")
    else:
        logging.error("Failed to load knowledge base")
    
    return success

if __name__ == "__main__":
    settings = Settings()

    Logger(settings).configure_logger()

    os.environ["LANGSMITH_TRACING"] = str(settings.langsmith.tracing)
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith.api_key

    # Initialize Chroma client
    chroma_client = Chroma(
        persist_directory=settings.knowledge_base.persist_directory,
        collection_name=settings.knowledge_base.space_key,
        embedding_function=OpenAIEmbeddings(model=settings.llm.embeddings_model, api_key=settings.llm.api_key),
        create_collection_if_not_exists=True
    )
    
    # Load knowledge base if required
    if settings.knowledge_base.load_on_startup:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(load_knowledge_base(settings, chroma_client))
    
    # Initialize Slack app
    app = AsyncApp(token=settings.slack.bot_token, signing_secret=settings.slack.signing_secret)
    
    # Create LLM
    llm = init_chat_model(settings.llm)

    memory_saver = MemorySaver()

    # Create individual agent instances
    incident_agent = IncidentAgent(llm=llm, checkpoint_saver=memory_saver)
    knowledge_agent = KnowledgeAgent(
        llm=llm, 
        checkpoint_saver=memory_saver, 
        document_retriever=DocumentRetriever(chroma_client)
    )
    direct_agent = DirectAgent(llm=llm, checkpoint_saver=memory_saver)
    
    # Initialize the main agent with all agents
    main_workflow = MainAgent(
        llm=llm,
        checkpoint_saver=memory_saver,
        agents=[incident_agent, knowledge_agent, direct_agent],
        default_agent_key=knowledge_agent.key  # Set knowledge as the default fallback
    ).build()

    # Start the Slack bot
    slack_event_handler = SlackEventsHandler(
         settings, 
         [AppMentionEventHandler(app, settings, main_workflow)], 
         app
    )
    
    slack_event_handler.start() 