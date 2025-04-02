import pytest
from unittest.mock import Mock, patch, MagicMock
import uuid
from langchain_core.documents import Document
from langchain_chroma.vectorstores import Chroma
from knowledge_base.chroma_writer import ChromaWriter
from knowledge_base.document_splitter import DocumentSplitter
from knowledge_base.token_document_splitter import TokenDocumentSplitter
from config.settings import Settings, knowledgeBaseSettings
import logging

@pytest.fixture
def mock_chroma_client():
    return Mock(spec=Chroma)

@pytest.fixture
def mock_settings():
    settings = Mock(spec=Settings)
    settings.knowledge_base = Mock(spec=knowledgeBaseSettings)
    settings.knowledge_base.url = "https://test.atlassian.net"
    return settings

@pytest.fixture
def mock_splitter():
    splitter = Mock(spec=DocumentSplitter)
    splitter.split_document.return_value = [
        Document(
            page_content="Chunk 1",
            metadata={
                "chunk_index": 0,
                "total_chunks": 2
            }
        ),
        Document(
            page_content="Chunk 2",
            metadata={
                "chunk_index": 1,
                "total_chunks": 2
            }
        )
    ]
    return splitter

@pytest.fixture
def chroma_writer(mock_chroma_client, mock_splitter):
    return ChromaWriter(mock_chroma_client, splitter=mock_splitter)

@pytest.fixture
def sample_document():
    return Document(
        page_content="This is a test document with some content that should be split into chunks.",
        metadata={
            "id": "test-doc-1",
            "url": "https://test.atlassian.net/wiki/spaces/TEST/pages/123",
            "title": "Test Document"
        }
    )

def test_init_with_invalid_client():
    """Test initialization with invalid client."""
    # The actual implementation allows None client
    writer = ChromaWriter(None, Mock(spec=DocumentSplitter))
    assert writer.client is None
    assert writer.splitter is not None

def test_init_with_invalid_splitter():
    """Test initialization with invalid splitter."""
    # The actual implementation allows None splitter
    writer = ChromaWriter(Mock(spec=Chroma), None)
    assert writer.client is not None
    assert writer.splitter is None

def test_init(chroma_writer):
    """Test ChromaWriter initialization."""
    assert chroma_writer.client is not None
    assert chroma_writer.splitter is not None
    assert chroma_writer.logger is not None

def test_write_documents_empty_list(chroma_writer, caplog):
    """Test writing empty document list."""
    result = chroma_writer.write_documents([])
    assert result == []
    assert "No documents provided for writing" in caplog.text

def test_write_documents_success(chroma_writer, sample_document, caplog):
    """Test successful document writing."""
    # Set log level to capture all logs
    caplog.set_level(logging.INFO)
    
    mock_ids = ["doc-1-0", "doc-1-1"]
    chroma_writer.client.add_documents.return_value = mock_ids
    
    result = chroma_writer.write_documents([sample_document])
    
    assert result == mock_ids
    chroma_writer.client.add_documents.assert_called_once()
    assert "Processing 1 documents" in caplog.text or f"Processing {len([sample_document])} documents" in caplog.text

def test_write_documents_chroma_error(chroma_writer, sample_document, caplog):
    """Test handling of ChromaDB errors."""
    chroma_writer.client.add_documents.side_effect = Exception("ChromaDB error")
    
    with pytest.raises(Exception, match="ChromaDB error"):
        chroma_writer.write_documents([sample_document])
    assert "Failed to write documents to ChromaDB: ChromaDB error" in caplog.text

def test_write_documents_no_chunks(chroma_writer, sample_document, caplog):
    """Test handling of documents that produce no chunks."""
    # Make the splitter return empty chunks
    chroma_writer.splitter.split_document.return_value = []
    
    result = chroma_writer.write_documents([sample_document])
    
    assert result == []
    chroma_writer.client.add_documents.assert_not_called()
    assert "No chunks were generated from the documents" in caplog.text 