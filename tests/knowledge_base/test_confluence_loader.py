import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document
from atlassian import Confluence
from knowledge_base.confluence_loader import ConfluenceLoader
from knowledge_base.document_processor import DocumentProcessor
from knowledge_base.confluence_document_processor import ConfluenceDocumentProcessor

@pytest.fixture
def mock_confluence_client():
    return Mock(spec=Confluence)

@pytest.fixture
def mock_document_processor():
    processor = Mock(spec=DocumentProcessor)
    processor.process_document.return_value = Document(
        page_content="Processed content",
        metadata={
            "id": "123",
            "title": "Test Page",
            "url": "https://test.atlassian.net/123",
            "version": 1
        }
    )
    return processor

@pytest.fixture
def confluence_loader(mock_confluence_client, mock_document_processor):
    return ConfluenceLoader(mock_confluence_client, mock_document_processor, batch_size=10)

@pytest.fixture
def mock_pages():
    return [
        {
            "id": "123",
            "title": "Test Page 1",
            "type": "page",
            "status": "current",
            "version": {"number": 1},
            "body": {
                "storage": {
                    "value": "<p>Test content 1</p>"
                }
            }
        },
        {
            "id": "124",
            "title": "Test Page 2",
            "type": "page",
            "status": "current",
            "version": {"number": 1},
            "body": {
                "storage": {
                    "value": "<p>Test content 2</p>"
                }
            }
        }
    ]

def test_init(confluence_loader):
    """Test ConfluenceLoader initialization."""
    assert confluence_loader.confluence is not None
    assert confluence_loader.document_processor is not None
    assert confluence_loader.batch_size == 10
    assert confluence_loader.logger is not None

def test_init_missing_batch_size():
    """Test initialization without required batch_size."""
    client = Mock(spec=Confluence)
    processor = Mock(spec=DocumentProcessor)
    with pytest.raises(TypeError):
        ConfluenceLoader(client, processor)

def test_load_documents_empty_space(confluence_loader, mock_confluence_client):
    """Test loading documents from empty space."""
    mock_confluence_client.get_all_pages_from_space.return_value = []
    
    documents = confluence_loader.load_documents("TEST")
    
    assert len(documents) == 0
    mock_confluence_client.get_all_pages_from_space.assert_called_once_with(
        "TEST", start=0, limit=10, status='current', expand='body.storage,version,space'
    )

def test_load_documents_success(confluence_loader, mock_confluence_client, mock_document_processor, mock_pages):
    """Test successful document loading."""
    # Configure the mock to return pages
    mock_confluence_client.get_all_pages_from_space.side_effect = [
        mock_pages,  # First batch
        []  # Empty second batch to end pagination
    ]
    
    # Call the method under test
    documents = confluence_loader.load_documents("TEST")
    
    # Validate results
    assert len(documents) == 2
    assert all(isinstance(doc, Document) for doc in documents)
    
    # Verify the document processor was called for each page
    assert mock_document_processor.process_document.call_count == 2
    
    # Verify proper API calls
    mock_confluence_client.get_all_pages_from_space.assert_called_with(
        "TEST", start=10, limit=10, status='current', expand='body.storage,version,space'
    )

def test_load_documents_pagination(confluence_loader, mock_confluence_client, mock_document_processor, mock_pages):
    """Test pagination handling."""
    # Configure the mock to return multiple batches
    mock_confluence_client.get_all_pages_from_space.side_effect = [
        [mock_pages[0]],  # First batch - one document
        [mock_pages[1]],  # Second batch - one document
        []  # Empty third batch to end pagination
    ]
    
    # Call the method under test
    documents = confluence_loader.load_documents("TEST")
    
    # Validate results
    assert len(documents) == 2
    
    # Verify API calls
    calls = mock_confluence_client.get_all_pages_from_space.call_args_list
    assert len(calls) == 3
    
    # First call with start=0
    assert calls[0][1]["start"] == 0
    # Second call with start=10 (after processing the first batch)
    assert calls[1][1]["start"] == 10
    # Third call with start=20 (after processing the second batch)
    assert calls[2][1]["start"] == 20

def test_load_documents_processor_error(confluence_loader, mock_confluence_client, mock_document_processor, mock_pages):
    """Test handling of document processor errors."""
    # Configure mocks
    mock_confluence_client.get_all_pages_from_space.side_effect = [
        mock_pages,  # First batch with the two pages
        []  # Empty second batch to break the loop
    ]
    
    # Set up the processor to succeed on first doc and fail on second
    def side_effect(page):
        if page["id"] == "123":
            return Document(page_content="Good document", metadata={"id": "123"})
        else:
            raise Exception("Processing error")
            
    mock_document_processor.process_document.side_effect = side_effect
    
    # Call the method under test
    documents = confluence_loader.load_documents("TEST")
    
    # Should still get the first document 
    assert len(documents) == 1
    assert documents[0].page_content == "Good document"
    
    # Verify the logger was called for the error
    mock_document_processor.process_document.assert_called_with(mock_pages[1])

def test_load_documents_space_key(confluence_loader, mock_confluence_client):
    """Test using space_key parameter."""
    mock_confluence_client.get_all_pages_from_space.return_value = []
    
    # Call with explicit space key
    confluence_loader.load_documents("CUSTOM")
    
    # Default space key should not be used
    mock_confluence_client.get_all_pages_from_space.assert_called_once_with(
        "CUSTOM", start=0, limit=10, status='current', expand='body.storage,version,space'
    ) 