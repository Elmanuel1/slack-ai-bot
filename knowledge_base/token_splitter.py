from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import TokenTextSplitter
from .document_splitter import DocumentSplitter


class TokenDocumentSplitter(DocumentSplitter):
    """Token-based document splitter implementation."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        """Initialize the splitter.
        
        Args:
            chunk_size: Size of each chunk in tokens
            chunk_overlap: Number of tokens to overlap between chunks
        """
        self.splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks.
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of split documents
        """
        return self.splitter.split_documents(documents)
    
    def split_text(self, text: str) -> List[str]:
        """Split text into smaller chunks.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        return self.splitter.split_text(text) 