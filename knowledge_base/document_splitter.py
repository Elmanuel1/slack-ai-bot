from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document


class DocumentSplitter(ABC):
    """Interface for document splitting strategies."""
    
    @abstractmethod
    def split_document(self, document: Document) -> List[Document]:
        """Split document into smaller chunks.
        
        Args:
            document: Document to split
            
        Returns:
            List of split documents
        """
        pass