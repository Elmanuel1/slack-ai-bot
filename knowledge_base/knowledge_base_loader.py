from abc import ABC, abstractmethod
from typing import List, Dict, Any
from langchain_core.documents import Document


class KnowledgeBaseLoader(ABC):
    """Interface for knowledge base loaders."""
    
    @abstractmethod
    def load_documents(self, space_key: str) -> List[Document]:
        """Load documents from the source."""
        pass
