from typing import List, Dict, Any, Optional
import logging
from langchain.vectorstores.base import VectorStore
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

class RetrieveDocumentsInput(BaseModel):
    """Input schema for document retrieval."""
    query: str = Field(..., description="The question to search for in the knowledge base")

class DocumentRetriever:
    """
    A class for retrieving relevant documents from a vector store,
    designed to work with KnowledgeAgent in LangGraph.
    """

    def __init__(self, vectorstore: VectorStore, k: int = 1):
        """
        Initialize the DocumentRetriever.

        Args:
            vectorstore: The vector store (FAISS, ChromaDB, etc.)
        """
        self.vectorstore = vectorstore
        self.logger = logging.getLogger(__name__)
        
        # Convert the vector store to a retriever with configurable k
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        

    def retrieve_documents(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents based on a query.

        Args:
            query: The search query
            k: Number of documents to retrieve

        Returns:
            List of document dictionaries with content and metadata
        """
        try:
       
            self.logger.debug   (f"Retrieving Query: {query}")

            # Get documents from the retriever
            docs = self.retriever.invoke(query)
            # Format the documents for return   
            results = []
            for i, doc in enumerate(docs):
                doc_dict = {
                    "content": doc.page_content,
                    "index": i,
                    "metadata": doc.metadata
                }
                
                # Add metadata if available
                if hasattr(doc, "metadata") and doc.metadata:
                    doc_dict["metadata"] = doc.metadata
                    
                    # Extract common metadata fields for convenience
                    if "source" in doc.metadata:
                        doc_dict["source"] = doc.metadata["source"]
                    if "title" in doc.metadata:
                        doc_dict["title"] = doc.metadata["title"]
                
                results.append(doc_dict)
                
            return results
        except Exception as e:
            self.logger.error(f"Error retrieving documents: ", e)
            return []
            
    def get_tool(self) -> StructuredTool:
        """
        Return a structured tool that can be used with the KnowledgeAgent.
        
        Returns:
            StructuredTool: A tool for document retrieval with schema validation
        """
        return StructuredTool.from_function(
            func=self.retrieve_documents,
            name="retrieve_documents",
            description="Retrieve relevant documents from the knowledge base",
            args_schema=RetrieveDocumentsInput,
            return_direct=False
        )