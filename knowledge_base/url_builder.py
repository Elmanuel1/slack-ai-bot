from abc import ABC, abstractmethod
from typing import Optional

class URLBuilder(ABC):
    """Abstract base class for URL construction strategies."""
    
    @abstractmethod
    def build(self) -> str:
        """Build a complete URL.
        
        Args:
            relative_url: Optional relative URL to append
            
        Returns:
            Complete URL
        """
        pass
