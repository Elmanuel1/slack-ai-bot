import pytest
import logging
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document
from knowledge_base.job_runner import KnowledgeBaseJobRunner, create_confluence_job
from knowledge_base.knowledge_base_loader import KnowledgeBaseLoader
from knowledge_base.knowledge_base_writer import KnowledgeBaseWriter
from config.settings import KnowledgeBaseSettings, Settings

@pytest.fixture
def mock_loader():
    loader = Mock(spec=KnowledgeBaseLoader)
    loader.load_documents.return_value = [
        Document(page_content="Document 1", metadata={"id": "doc1"}),
        Document(page_content="Document 2", metadata={"id": "doc2"}),
        Document(page_content="Document 3", metadata={"id": "doc3"}),
        Document(page_content="Document 4", metadata={"id": "doc4"}),
        Document(page_content="Document 5", metadata={"id": "doc5"})
    ]
    return loader

@pytest.fixture
def mock_writer():
    writer = Mock(spec=KnowledgeBaseWriter)
    writer.write_documents.side_effect = lambda docs: [f"id-{doc.metadata['id']}" for doc in docs]
    return writer

@pytest.fixture
def mock_settings():
    settings = Mock(spec=KnowledgeBaseSettings)
    settings.space_key = "TEST"
    settings.batch_size = 2
    return settings

@pytest.fixture
def job_runner(mock_loader, mock_writer, mock_settings):
    return KnowledgeBaseJobRunner(
        loader=mock_loader,
        writer=mock_writer,
        settings=mock_settings
    )

@pytest.fixture
def test_settings():
    """Create complete mock settings with llm attribute"""
    settings = Mock(spec=Settings)
    settings.knowledge_base = Mock(spec=KnowledgeBaseSettings)
    settings.knowledge_base.space_key = "TEST"
    settings.knowledge_base.batch_size = 50
    settings.knowledge_base.host = "test.atlassian.net"
    settings.knowledge_base.path = "wiki"
    settings.knowledge_base.username = "test@example.com"
    settings.knowledge_base.api_token = "test-token"
    settings.knowledge_base.persist_directory = "data/test_knowledge_base"
    
    # Add llm attribute
    settings.llm = Mock()
    settings.llm.embeddings_model = "text-embedding-ada-002"
    settings.llm.api_key = "test-api-key"
    
    return settings

def test_init(job_runner, mock_loader, mock_writer, mock_settings):
    """Test KnowledgeBaseJobRunner initialization."""
    assert job_runner.loader is mock_loader
    assert job_runner.writer is mock_writer
    assert job_runner.settings is mock_settings
    assert job_runner.logger is not None

def test_run_success(job_runner, mock_loader, mock_writer, mock_settings, caplog):
    """Test successful job run."""
    # Set log level to capture info logs
    caplog.set_level(logging.INFO)
    
    job_runner.run()
    
    # Verify loader was called
    mock_loader.load_documents.assert_called_once_with("TEST")
    
    # Verify writer was called for each batch
    assert mock_writer.write_documents.call_count == 3  # 5 docs with batch size 2 = 3 batches
    
    # Verify logs
    assert "Starting knowledge base loading job" in caplog.text
    assert "Loaded 5 documents from source" in caplog.text
    assert "Successfully completed knowledge base loading job" in caplog.text

def test_run_empty_documents(job_runner, mock_loader, mock_writer, caplog):
    """Test job run with no documents."""
    # Set log level to capture info logs
    caplog.set_level(logging.INFO)
    
    mock_loader.load_documents.return_value = []
    
    job_runner.run()
    
    mock_loader.load_documents.assert_called_once()
    mock_writer.write_documents.assert_not_called()
    assert "Loaded 0 documents from source" in caplog.text

def test_run_loader_error(job_runner, mock_loader, caplog):
    """Test error handling for loader errors."""
    # Set log level to capture error logs
    caplog.set_level(logging.ERROR)
    
    mock_loader.load_documents.side_effect = Exception("Loader error")
    
    with pytest.raises(Exception, match="Loader error"):
        job_runner.run()
    
    assert "Error in knowledge base loading job: Loader error" in caplog.text

def test_run_writer_error(job_runner, mock_writer, caplog):
    """Test error handling for writer errors."""
    # Set log level to capture error logs
    caplog.set_level(logging.ERROR)
    
    mock_writer.write_documents.side_effect = Exception("Writer error")
    
    with pytest.raises(Exception, match="Writer error"):
        job_runner.run()
    
    assert "Error in knowledge base loading job: Writer error" in caplog.text

@patch('knowledge_base.job_runner.Chroma')
@patch('knowledge_base.job_runner.OpenAIEmbeddings')
@patch('knowledge_base.job_runner.Confluence')
@patch('knowledge_base.job_runner.ConfluenceURLBuilder')
@patch('knowledge_base.job_runner.TokenDocumentSplitter')
@patch('knowledge_base.job_runner.ChromaWriter')
@patch('knowledge_base.job_runner.ConfluenceLoader')
@patch('knowledge_base.job_runner.ConfluenceDocumentProcessor')
def test_create_confluence_job(mock_conf_processor, mock_conf_loader, mock_chroma_writer, 
                             mock_token_splitter, mock_url_builder, mock_confluence, 
                             mock_embeddings, mock_chroma, test_settings):
    """Test the factory function to create a Confluence job."""
    # Setup mocks
    mock_url_builder.return_value.build.return_value = "https://test.atlassian.net/wiki"
    
    # Call the factory function
    job = create_confluence_job(test_settings)
    
    # Verify the job was created correctly
    assert isinstance(job, KnowledgeBaseJobRunner)
    
    # Verify components were created
    mock_chroma.assert_called_once()
    mock_embeddings.assert_called_once()
    mock_confluence.assert_called_once()
    mock_token_splitter.assert_called_once()
    mock_chroma_writer.assert_called_once()
    mock_conf_loader.assert_called_once()
    mock_conf_processor.assert_called_once() 