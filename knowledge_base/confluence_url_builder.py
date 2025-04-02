from .url_builder import URLBuilder
class ConfluenceURLBuilder(URLBuilder):
    """URL builder for Confluence URLs."""
    
    def __init__(self, host: str, path: str):
        """Initialize the URL builder.
        
        Args:
            host: Host of the Confluence instance
            path: Path of the Confluence instance
        """
        self.host = host
        self.path = path
    
    def build(self) -> str:
        """Build a complete Confluence URL.
        
        Args:
            relative_url: Optional relative URL to append
            
        Returns:
            Complete URL
        """
        return f"{self.host.rstrip('/')}"