import pytest
import os
from unittest.mock import Mock
from config.settings import Settings, knowledgeBaseSettings

@pytest.fixture(autouse=True)
def mock_env_vars():
    """Mock environment variables for testing."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("CONFLUENCE_URL", "https://test.atlassian.net")
        mp.setenv("CONFLUENCE_USERNAME", "test@example.com")
        mp.setenv("CONFLUENCE_API_TOKEN", "test-token")
        mp.setenv("CONFLUENCE_SPACE_KEY", "TEST")
        mp.setenv("CHROMA_PERSIST_DIR", "data/test_knowledge_base")
        yield

@pytest.fixture
def test_settings():
    """Create test settings with mocked values."""
    settings = Mock(spec=Settings)
    settings.knowledge_base = Mock(spec=knowledgeBaseSettings)
    settings.knowledge_base.url = "https://test.atlassian.net"
    settings.knowledge_base.username = "test@example.com"
    settings.knowledge_base.api_token = "test-token"
    settings.knowledge_base.space_key = "TEST"
    settings.knowledge_base.max_pages = 100
    settings.knowledge_base.persist_directory = "data/test_knowledge_base"
    settings.knowledge_base.batch_size = 50
    return settings

@pytest.fixture(autouse=True)
def cleanup_test_dir():
    """Clean up test directory before and after tests."""
    test_dir = "data/test_knowledge_base"
    if os.path.exists(test_dir):
        for file in os.listdir(test_dir):
            os.remove(os.path.join(test_dir, file))
    yield
    if os.path.exists(test_dir):
        for file in os.listdir(test_dir):
            os.remove(os.path.join(test_dir, file)) 