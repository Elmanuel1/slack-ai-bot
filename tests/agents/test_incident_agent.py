import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, Graph
from agents.incident_agent import IncidentAgent

@pytest.fixture
def mock_llm():
    mock = MagicMock(spec=BaseChatModel)
    return mock

@pytest.fixture
def mock_checkpoint_saver():
    return MagicMock(spec=MemorySaver)

@pytest.fixture
def incident_agent(mock_llm, mock_checkpoint_saver):
    agent = IncidentAgent(mock_llm, mock_checkpoint_saver)
    return agent

def test_init(incident_agent, mock_llm, mock_checkpoint_saver):
    """Test IncidentAgent initialization."""
    assert incident_agent.llm is mock_llm
    assert incident_agent.checkpoint_saver is mock_checkpoint_saver

def test_process_message(incident_agent):
    """Test the process_message method."""
    # Setup test state
    state = {
        "messages": [HumanMessage(content="We have an outage in the payment system")]
    }
    
    # Process the message
    result = incident_agent.process_message(state)
    
    # Check the result contains our messages and a response
    assert len(result["messages"]) == 2
    assert result["messages"][0] == state["messages"][0]
    assert isinstance(result["messages"][1], AIMessage)
    assert "Another incident to handle" in result["messages"][1].content
    assert "payment system" in result["messages"][1].content

def test_build(incident_agent):
    """Test that graph can be built."""
    with patch('langgraph.graph.StateGraph'):
        graph = incident_agent.build()
        assert graph is not None

def test_get_response(incident_agent):
    """Test get_response method."""
    # Create a test state
    state = {
        "messages": [
            HumanMessage(content="Hello"),
            AIMessage(content="Test response")
        ]
    }
    
    # Get response from state
    response = incident_agent.get_response(state)
    
    # Check the response matches the last message content
    assert response == "Test response"

def test_key_property(incident_agent):
    """Test the key property."""
    assert incident_agent.key == "incidentagent" 