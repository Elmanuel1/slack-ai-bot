import pytest
import logging
from unittest.mock import Mock, patch
from langchain_core.documents import Document
from knowledge_base.confluence_document_processor import ConfluenceDocumentProcessor

@pytest.fixture
def processor():
    return ConfluenceDocumentProcessor(base_url="https://test.atlassian.net")

@pytest.fixture
def sample_confluence_page():
    return {
        "id": "123456",
        "title": "Test Confluence Page",
        "type": "page",
        "status": "current",
        "version": {"number": 5},
        "body": {
            "storage": {
                "value": "<p>This is a test Confluence page content.</p>"
            }
        }
    }

def test_init():
    """Test ConfluenceDocumentProcessor initialization."""
    processor = ConfluenceDocumentProcessor(base_url="https://test.atlassian.net")
    assert processor.base_url == "https://test.atlassian.net"
    assert processor.logger is not None

def test_process_document_success(processor, sample_confluence_page):
    """Test successful document processing."""
    doc = processor.process_document(sample_confluence_page)
    
    assert doc is not None
    assert isinstance(doc, Document)
    assert doc.page_content == "<p>This is a test Confluence page content.</p>"
    assert doc.metadata["id"] == "123456"
    assert doc.metadata["title"] == "Test Confluence Page"
    assert doc.metadata["url"] == "https://test.atlassian.net/123456"
    assert doc.metadata["version"] == 5
    assert doc.id == "123456"

def test_process_document_no_content(processor, caplog, sample_confluence_page):
    """Test handling of document with no content."""
    # Set log level to capture info logs
    caplog.set_level(logging.INFO)
    
    # Remove content
    sample_confluence_page["body"]["storage"]["value"] = ""
    
    doc = processor.process_document(sample_confluence_page)
    
    assert doc is None
    assert "Skipping document ID: 123456" in caplog.text

def test_process_document_missing_body(processor, caplog, sample_confluence_page):
    """Test handling of document with missing body."""
    # Set log level to capture info logs
    caplog.set_level(logging.INFO)
    
    # Remove body structure
    del sample_confluence_page["body"]
    
    doc = processor.process_document(sample_confluence_page)
    
    assert doc is None
    assert "Skipping document ID: 123456" in caplog.text

def test_process_document_exception_handling(processor, caplog, sample_confluence_page):
    """Test error handling during document processing."""
    # Set log level to capture error logs
    caplog.set_level(logging.ERROR)
    
    # Make the document invalid to trigger an error
    del sample_confluence_page["id"]
    
    with pytest.raises(Exception):
        processor.process_document(sample_confluence_page)
    
    assert "Error processing document unknown" in caplog.text

def test_process_document_missing_version(processor, sample_confluence_page):
    """Test handling of document with missing version."""
    # Remove version
    del sample_confluence_page["version"]
    
    doc = processor.process_document(sample_confluence_page)
    
    assert doc is not None
    assert doc.metadata["version"] == 1  # Default to version 1 