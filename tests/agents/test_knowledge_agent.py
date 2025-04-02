import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import StructuredTool, tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END
from documents.document_retriever import DocumentRetriever, RetrieveDocumentsInput
from agents.knowledge_agent import KnowledgeAgent
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

# Create a real tool function for testing instead of a mock
@tool
def mock_retrieve_documents(query: str) -> list:
    """Mock function to retrieve documents."""
    return [
        {
            "content": "This is a test document about Python programming.",
            "index": 0,
            "metadata": {
                "id": "doc-1",
                "title": "Python Guide",
                "url": "https://test.atlassian.net/wiki/spaces/TEST/pages/123"
            }
        }
    ]

@pytest.fixture
def mock_llm():
    mock = MagicMock(spec=BaseChatModel)
    mock.invoke.return_value = AIMessage(content="This is a test response")
    return mock

@pytest.fixture
def mock_checkpoint_saver():
    return MagicMock(spec=MemorySaver)

@pytest.fixture
def mock_document_retriever():
    mock = MagicMock(spec=DocumentRetriever)
    
    # Create a real tool instance for the get_tool method
    doc_tool = StructuredTool.from_function(
        func=mock_retrieve_documents,
        name="retrieve_documents",
        description="Retrieve relevant documents from the knowledge base",
        args_schema=RetrieveDocumentsInput,
        return_direct=False
    )
    mock.get_tool.return_value = doc_tool
    
    return mock

@pytest.fixture
def knowledge_agent(mock_llm, mock_checkpoint_saver, mock_document_retriever):
    with patch('agents.knowledge_agent.ToolNode', autospec=True) as mock_tool_node:
        # Mock the ToolNode to return a callable function
        mock_tool_node.return_value = lambda state: {"messages": [AIMessage(content="Tool response")]}
        
        agent = KnowledgeAgent(mock_llm, mock_checkpoint_saver, mock_document_retriever)
        # Reset mock call counts after initialization
        mock_llm.reset_mock()
        mock_document_retriever.reset_mock()
        return agent

def test_init(knowledge_agent, mock_llm, mock_checkpoint_saver, mock_document_retriever):
    """Test KnowledgeAgent initialization."""
    assert knowledge_agent.llm is mock_llm
    assert knowledge_agent.checkpoint_saver is mock_checkpoint_saver
    assert knowledge_agent.document_retriever is mock_document_retriever
    assert hasattr(knowledge_agent, "knowledge_tools")

def test_build(knowledge_agent):
    """Test that graph can be built."""
    with patch('langgraph.graph.StateGraph'):
        graph = knowledge_agent.build()
        assert graph is not None

def test_knowledge_llm_node(knowledge_agent, mock_llm):
    """Test the LLM node processing."""
    # Create a tool call response
    tool_call_response = AIMessage(
        content="Using tool to search",
        tool_calls=[{
            "name": "retrieve_documents", 
            "args": {"query": "python test"},
            "id": "call_12345"
        }]
    )
    knowledge_agent.knowledge_model.invoke.return_value = tool_call_response
    
    # Setup test state
    state = {
        "messages": [HumanMessage(content="How to write Python tests?")]
    }
    
    # Get the command result
    command = knowledge_agent.knowledge_LLM_node(state)
    
    # Verify tool node is used next
    assert command.goto == "tools"
    assert "messages" in command.update
    assert command.update["messages"][0] == tool_call_response

def test_knowledge_llm_node_no_tool_call(knowledge_agent, mock_llm):
    """Test LLM node with no tool calls."""
    # Create a direct response (no tool calls)
    direct_response = AIMessage(content="Here's an answer without tool use")
    knowledge_agent.knowledge_model.invoke.return_value = direct_response
    
    # Setup test state
    state = {
        "messages": [HumanMessage(content="How to write Python tests?")]
    }
    
    # Get the command result
    command = knowledge_agent.knowledge_LLM_node(state)
    
    # Should go to END since no tool calls
    assert command.goto == END
    assert "messages" in command.update
    assert command.update["messages"][0] == direct_response

def test_knowledge_llm_node_error_handling(knowledge_agent):
    """Test error handling in LLM node."""
    # Setup test state
    state = {
        "messages": [HumanMessage(content="How to write Python tests?")]
    }
    
    # Make the knowledge model raise an exception
    knowledge_agent.knowledge_model.invoke.side_effect = Exception("Test error")
    
    # Get the command result
    command = knowledge_agent.knowledge_LLM_node(state)
    
    # Check that we go to END
    assert command.goto == END
    assert "messages" in command.update
    assert isinstance(command.update["messages"][0], AIMessage)
    assert "I encountered an error" in command.update["messages"][0].content 