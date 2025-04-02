from typing import List, Dict, Any
import logging
from atlassian import Confluence
from knowledge_base.url_builder import URLBuilder
from langchain_core.documents import Document
from .knowledge_base_loader import KnowledgeBaseLoader
from .document_processor import DocumentProcessor

class ConfluenceLoader(KnowledgeBaseLoader):
    """Loads documents from Confluence."""
    
    def __init__(
        self,
        confluence: Confluence,
        document_processor: DocumentProcessor,
        batch_size: int
    ):
        self.confluence = confluence
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)
        self.document_processor = document_processor
        
    def load_documents(self, space_key: str) -> List[Document]:
        """Load all pages from the Confluence space with pagination."""
        self.logger.info(f"Loading documents from Confluence space: {space_key}")
        
        documents = []
        start = 0
        
        while True:
            self.logger.info(f"Fetching pages {start} to {start + self.batch_size}")
            
            # Get pages for current batch with full content
            pages = self.confluence.get_all_pages_from_space(
                space_key,
                start=start,
                limit=self.batch_size,
                status='current',
                expand='body.storage,version,space'  
            )
            
            if not pages:
                break
                
            # Process pages in current batch
            for page in pages:
                try:
                    doc = self.document_processor.process_document(page)
                    if doc:
                        documents.append(doc)
                except Exception as e:
                    self.logger.error(f"Error processing page {page['id']}: {str(e)}")
                    continue
                
            # Move to next batch
            start += self.batch_size
            self.logger.info(f"Processed {len(documents)} documents so far")
        
        self.logger.info(f"Completed loading {len(documents)} documents from Confluence")
        return documents