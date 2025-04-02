from abc import ABC, abstractmethod
from typing import List
import uuid
from langchain_core.documents import Document
from langchain.text_splitter import TokenTextSplitter
import logging
from knowledge_base.document_splitter import DocumentSplitter


class TokenDocumentSplitter(DocumentSplitter):
    """Document splitter using token-based splitting."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        """Initialize the splitter.
        
        Args:
            chunk_size: Size of each chunk in tokens
            chunk_overlap: Number of tokens to overlap between chunks
        """
        self.splitter = TokenTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.logger = logging.getLogger(__name__)

    def split_document(self, doc: Document) -> List[Document]:
        """Split a document into chunks using token-based splitting.
        
        Args:
            doc: Document to split
            
        Returns:
            List of chunked documents
            
        Note:
            Returns empty list if document has no content
        """
        try:
            if not doc.page_content:
                self.logger.warning(f"Document has no content, skipping")
                return []
                
            chunks = self.splitter.split_text(doc.page_content)
            
            return [
                Document(
                    id=doc.id + str(uuid.uuid4()),
                    page_content=chunk,
                    metadata={
                        **doc.metadata,
                        'chunk_index': i,
                        'total_chunks': len(chunks)
                    }
                )
                for i, chunk in enumerate(chunks)
            ]
            
        except Exception as e:
            self.logger.error(f"Error splitting document: {str(e)}")
            return [] 