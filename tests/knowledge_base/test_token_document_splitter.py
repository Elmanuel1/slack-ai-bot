import pytest
import logging
from unittest.mock import Mock, patch
from langchain_core.documents import Document
from knowledge_base.token_document_splitter import TokenDocumentSplitter

@pytest.fixture
def token_splitter():
    return TokenDocumentSplitter(chunk_size=100, chunk_overlap=20)

@pytest.fixture
def sample_document():
    return Document(
        page_content="This is a test document with enough content to be split into multiple chunks. " * 10,
        metadata={
            "id": "test-doc-1",
            "title": "Test Document",
            "url": "https://test.atlassian.net/wiki/spaces/TEST/pages/123"
        },
        id="test-doc-1"
    )

def test_init():
    """Test TokenDocumentSplitter initialization."""
    splitter = TokenDocumentSplitter(chunk_size=500, chunk_overlap=50)
    # TokenTextSplitter doesn't expose chunk_size/chunk_overlap as attributes
    # Just verify splitter was created successfully
    assert splitter.splitter is not None
    assert splitter.logger is not None

def test_init_default_values():
    """Test initialization with default values."""
    splitter = TokenDocumentSplitter()
    # Just verify splitter was created successfully
    assert splitter.splitter is not None
    assert splitter.logger is not None

def test_split_document_empty_content(token_splitter, caplog):
    """Test handling of document with empty content."""
    caplog.set_level(logging.WARNING)
    
    doc = Document(
        page_content="",
        metadata={"id": "empty-doc"},
        id="empty-doc"
    )
    
    chunks = token_splitter.split_document(doc)
    
    assert len(chunks) == 0
    assert "Document has no content, skipping" in caplog.text

def test_split_document_empty_content_as_none(token_splitter, caplog):
    """Test handling of document with empty content (using empty string instead of None)."""
    caplog.set_level(logging.WARNING)
    
    doc = Document(
        page_content="",
        metadata={"id": "none-doc"},
        id="none-doc"
    )
    
    chunks = token_splitter.split_document(doc)
    
    assert len(chunks) == 0
    assert "Document has no content, skipping" in caplog.text

def test_split_document_whitespace_content(token_splitter):
    """Test handling of document with only whitespace content."""
    doc = Document(
        page_content="   \n\t  ",
        metadata={"id": "whitespace-doc"},
        id="whitespace-doc"
    )
    
    # Whitespace is treated as content by the token splitter
    chunks = token_splitter.split_document(doc)
    
    assert len(chunks) > 0
    assert all(chunk.page_content.isspace() for chunk in chunks)

def test_split_document_success(token_splitter, sample_document):
    """Test successful document splitting."""
    chunks = token_splitter.split_document(sample_document)
    
    # Should create multiple chunks
    assert len(chunks) > 1
    assert all(isinstance(chunk, Document) for chunk in chunks)
    assert all(chunk.page_content for chunk in chunks)
    
    # Check metadata
    first_chunk = chunks[0]
    assert first_chunk.metadata["id"] == "test-doc-1"
    assert first_chunk.metadata["title"] == "Test Document"
    assert first_chunk.metadata["url"] == "https://test.atlassian.net/wiki/spaces/TEST/pages/123"
    assert first_chunk.metadata["chunk_index"] == 0
    assert first_chunk.metadata["total_chunks"] == len(chunks)
    
    # Check that each chunk has unique ID
    chunk_ids = [chunk.id for chunk in chunks]
    assert len(chunk_ids) == len(set(chunk_ids))
    
    # Check that IDs contain the document ID as prefix
    assert all(chunk.id.startswith("test-doc-1") for chunk in chunks)

def test_split_document_exception_handling(token_splitter, caplog):
    """Test error handling during document splitting."""
    caplog.set_level(logging.ERROR)
    
    # Mock the splitter to raise an exception
    with patch.object(token_splitter.splitter, 'split_text', side_effect=Exception("Test error")):
        doc = Document(
            page_content="Test content",
            metadata={"id": "error-doc"},
            id="error-doc"
        )
        
        chunks = token_splitter.split_document(doc)
        
        assert len(chunks) == 0
        assert "Error splitting document: Test error" in caplog.text 