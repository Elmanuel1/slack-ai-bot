from typing import List, Dict, Any
import logging
from langchain_chroma.vectorstores import Chroma
from knowledge_base.knowledge_base_writer import KnowledgeBaseWriter
from langchain_core.documents import Document
from knowledge_base.document_splitter import DocumentSplitter


class ChromaWriter(KnowledgeBaseWriter):
    """Writes documents to ChromaDB."""
    
    def __init__(
        self,
        client: Chroma,
        splitter: DocumentSplitter
    ):
        """Initialize the writer.
        
        Args:
            client: ChromaDB client
            splitter: Document splitter to use
        """
        self.client = client
        self.splitter = splitter
        self.logger = logging.getLogger(__name__)
    
    def write_documents(self, documents: List[Document]) -> List[str]:
        """Write documents to ChromaDB.
        
        Args:
            documents: List of documents to write
            
        Returns:
            List of document IDs
            
        Raises:
            ValueError: If no documents are provided
            Exception: If document processing or writing fails
        """
        if not documents:
            self.logger.warning("No documents provided for writing")
            return []

        self.logger.info(f"Processing {len(documents)} documents")
        try:
            # Split all documents into chunks
            all_chunks = [chunk for doc in documents for chunk in self.splitter.split_document(doc)]
            
            if not all_chunks:
                self.logger.warning("No chunks were generated from the documents")
                return []
            
            # Insert all chunks in a single call
            all_ids = self.client.add_documents(all_chunks)
            
            self.logger.info(f"Successfully wrote {len(all_ids)} total chunks to ChromaDB")
            return all_ids
      
        except Exception as e:
            self.logger.error(f"Failed to write documents to ChromaDB: {str(e)}")
            raise
