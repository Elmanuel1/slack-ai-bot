from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document

class KnowledgeBaseWriter(ABC):
    """Interface for knowledge base writers."""
    
    @abstractmethod
    def write_documents(self, documents: List[Document]) -> None:
        """Write documents to the knowledge base."""
        pass