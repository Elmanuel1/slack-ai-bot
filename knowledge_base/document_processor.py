from abc import ABC, abstractmethod
from typing import Dict, Any
from langchain_core.documents import Document


class DocumentProcessor(ABC):
    """Interface for document processing strategies."""
    
    @abstractmethod
    def process_document(self, raw_doc: Dict[str, Any]) -> Document:
        """Process a raw document into a Document object.
        
        Args:
            raw_doc: Raw document data
            
        Returns:
            Processed Document object
            
        Raises:
            Exception: If document processing fails
        """
        pass 