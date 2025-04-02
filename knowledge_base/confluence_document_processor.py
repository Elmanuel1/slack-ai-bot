import logging
from typing import Dict, Any
from langchain_core.documents import Document
from .document_processor import DocumentProcessor


class ConfluenceDocumentProcessor(DocumentProcessor):
    """Processes Confluence documents into Document objects."""
    
    def __init__(self, base_url: str):
        """Initialize the processor."""
        self.logger = logging.getLogger(__name__)
        self.base_url = base_url
    
    def process_document(self, raw_doc: Dict[str, Any]) -> Document:
        """Process a raw Confluence document into a Document object.
        
        Args:
            raw_doc: Raw Confluence document data
            
        Returns:
            Processed Document object
            
        Raises:
            Exception: If document processing fails
        """
        try:
            
            # Extract content
            content = raw_doc.get('body', {}).get('storage', {}).get('value', '')

            if not content:
                self.logger.info(f"Skipping document ID: {raw_doc.get('id')}, Title: {raw_doc.get('title')}")
                self.logger.info(f"Processing content:  {content}")
                return None
            
            # Extract metadata
            metadata = {
                'id': raw_doc['id'],
                'title': raw_doc['title'],
                'url': f"{self.base_url}/{raw_doc['id']}",
                'version': raw_doc.get('version', {}).get('number', 1)  # Default to version 1 if not present
            }
            
            doc = Document(
                page_content=content,
                metadata=metadata,
                id=raw_doc['id']
            )
            return doc
            
        except Exception as e:
            self.logger.error(f"Error processing document {raw_doc.get('id', 'unknown')}: {str(e)}")
            raise 