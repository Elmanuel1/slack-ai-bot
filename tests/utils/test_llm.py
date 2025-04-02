import pytest
from unittest.mock import Mock, patch, MagicMock
from utils.llm import ChatModelFactory
from config.settings import LLMSettings
from langchain_openai import ChatOpenAI
from pydantic import SecretStr


@pytest.fixture
def mock_settings():
    """Fixture for creating test settings."""
    return LLMSettings(
        provider="openai",
        model="gpt-4",
        temperature=0.0,
        api_key="test-key"
    )


@pytest.fixture
def mock_model():
    """Fixture for creating a mock model class."""
    return Mock()


def test_create_openai_model(mock_settings):
    """Test creating an OpenAI model with default factory."""
    # Arrange
    factory = ChatModelFactory()
    
    # Act
    model = factory.create(mock_settings)
    
    # Assert
    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "gpt-4"
    assert model.temperature == 0.0
    assert isinstance(model.openai_api_key, SecretStr)
    assert model.openai_api_key.get_secret_value() == "test-key"


def test_unsupported_provider(mock_settings):
    """Test error handling for unsupported provider."""
    # Arrange
    mock_settings.provider = "unsupported"
    factory = ChatModelFactory()
    
    # Act & Assert
    with pytest.raises(ValueError) as exc_info:
        factory.create(mock_settings)
    assert "Unsupported LLM provider" in str(exc_info.value)
    assert "Supported providers are: openai" in str(exc_info.value)


def test_custom_provider(mock_settings, mock_model):
    """Test creating a model with custom provider."""
    # Arrange
    mock_init_kwargs = lambda settings: {"test": "value"}
    
    mock_settings.provider = "custom"
    
    factory = ChatModelFactory(
        providers={"custom": mock_model},
        init_kwargs={"custom": mock_init_kwargs}
    )
    
    # Act
    model = factory.create(mock_settings)
    
    # Assert
    assert model == mock_model.return_value  # Compare with the instance returned by the mock
    mock_model.assert_called_once_with(test="value")


def test_empty_providers():
    """Test factory with empty providers dictionary."""
    # Arrange
    factory = ChatModelFactory(providers={})
    settings = LLMSettings(
        provider="any",
        model="test",
        temperature=0.0,
        api_key="test"
    )
    
    # Act & Assert
    with pytest.raises(ValueError) as exc_info:
        factory.create(settings)
    assert "Unsupported LLM provider" in str(exc_info.value)


def test_provider_case_insensitive(mock_settings):
    """Test that provider names are case insensitive."""
    # Arrange
    factory = ChatModelFactory()
    mock_settings.provider = "OPENAI"  # Uppercase
    
    # Act
    model = factory.create(mock_settings)
    
    # Assert
    assert isinstance(model, ChatOpenAI)


def test_init_kwargs_override(mock_settings):
    """Test that init_kwargs can be overridden."""
    # Arrange
    custom_kwargs = lambda settings: {
        "model_name": "custom-model",
        "temperature": settings.temperature,
        "openai_api_key": settings.api_key
    }
    factory = ChatModelFactory(
        init_kwargs={"openai": custom_kwargs}
    )
    
    # Act
    model = factory.create(mock_settings)
    
    # Assert
    assert model.model_name == "custom-model"
    assert model.temperature == 0.0
    assert isinstance(model.openai_api_key, SecretStr)
    assert model.openai_api_key.get_secret_value() == "test-key"


@patch('utils.llm.ChatOpenAI')
def test_openai_model_initialization(mock_chat_openai, mock_settings):
    """Test OpenAI model initialization with mocked ChatOpenAI."""
    # Arrange
    factory = ChatModelFactory()
    
    # Act
    factory.create(mock_settings)
    
    # Assert
    mock_chat_openai.assert_called_once_with(
        model="gpt-4",
        temperature=0.0,
        openai_api_key="test-key"
    ) 