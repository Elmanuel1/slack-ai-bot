from typing import List, Dict, Any, Optional
import logging
from langchain.vectorstores.base import VectorStore
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

class RetrieveDocumentsInput(BaseModel):
    """Input schema for document retrieval.
    
    This Pydantic model defines the expected input format for the document
    retrieval tool. It ensures that a query string is provided for searching
    the knowledge base.
    
    Attributes:
        query (str): The question or search query to look for in the knowledge base.
    """
    query: str = Field(..., description="The question to search for in the knowledge base")

class DocumentRetriever:
    """A class for retrieving relevant documents from a vector store.
    
    This class provides functionality to search a vector database for documents
    relevant to a given query. It's designed to work with the KnowledgeAgent
    in the LangGraph workflow system as a tool for retrieving information from
    the knowledge base.
    
    The retriever converts search results into a standardized format with content
    and metadata that can be easily processed by language models.
    """

    def __init__(self, vectorstore: VectorStore, k: int = 1):
        """Initialize the DocumentRetriever.

        Args:
            vectorstore (VectorStore): The vector store (FAISS, ChromaDB, etc.) containing
                document embeddings.
            k (int, optional): Number of documents to retrieve for each query. Defaults to 1.
            
        Example:
            >>> from langchain_chroma import Chroma
            >>> from langchain_openai import OpenAIEmbeddings
            >>> 
            >>> chroma_db = Chroma(
            ...     persist_directory="./knowledge_base",
            ...     embedding_function=OpenAIEmbeddings(),
            ... )
            >>> retriever = DocumentRetriever(vectorstore=chroma_db, k=3)
        """
        self.vectorstore = vectorstore
        self.logger = logging.getLogger(__name__)
        
        # Convert the vector store to a retriever with configurable k
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        

    async def retrieve_documents(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve relevant documents based on a query.

        This method searches the vector store for documents that are semantically
        similar to the provided query. It formats the results into a standardized
        dictionary format with content and metadata.

        Args:
            query (str): The search query string.

        Returns:
            List[Dict[str, Any]]: A list of document dictionaries, each containing:
                - content: The document's text content
                - index: The position in the result set
                - metadata: Additional document metadata if available
                - source: The document source (if available in metadata)
                - title: The document title (if available in metadata)
                
        Example:
            >>> results = retriever.retrieve_documents("What is our refund policy?")
            >>> for doc in results:
            ...     print(f"Title: {doc.get('title', 'No title')}")
            ...     print(f"Content: {doc['content'][:100]}...")
        """
        try:
       
            self.logger.debug(f"Retrieving Query: {query}")

            # Get documents from the retriever
            docs = await self.retriever.ainvoke(query)
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
        """Return a structured tool for document retrieval.
        
        Creates and returns a LangChain StructuredTool that can be used with
        language models in the KnowledgeAgent. This tool enables language models
        to search the knowledge base by calling the retrieve_documents method
        with proper input validation.
        
        Returns:
            StructuredTool: A tool for document retrieval with schema validation.
            
        Example:
            >>> tool = retriever.get_tool()
            >>> # The tool can then be provided to a language model
            >>> tools = [tool]
            >>> llm_with_tools = llm.bind_tools(tools)
        """
        # Create a sync wrapper for the async retrieve_documents function
        async def retrieve_documents_wrapper(query: str) -> List[Dict[str, Any]]:
            return await self.retrieve_documents(query)
            
        return StructuredTool.from_function(
            func=retrieve_documents_wrapper,
            name="retrieve_documents",
            description="Retrieve relevant documents from the knowledge base",
            args_schema=RetrieveDocumentsInput,
            coroutine=retrieve_documents_wrapper,
            return_direct=False
        )