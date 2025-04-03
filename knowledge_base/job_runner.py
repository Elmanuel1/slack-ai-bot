import logging
from typing import Optional

from knowledge_base.confluence_document_processor import ConfluenceDocumentProcessor
from knowledge_base.confluence_url_builder import ConfluenceURLBuilder
from knowledge_base.token_document_splitter import TokenDocumentSplitter
from knowledge_base.knowledge_base_loader import KnowledgeBaseLoader
from knowledge_base.knowledge_base_writer import KnowledgeBaseWriter
from knowledge_base.confluence_loader import ConfluenceLoader
from knowledge_base.chroma_writer import ChromaWriter
from config.settings import Settings, KnowledgeBaseSettings
from langchain_chroma.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from atlassian import Confluence
class KnowledgeBaseJobRunner:
    """Runs the knowledge base loading job."""
    
    def __init__(
        self,
        loader: KnowledgeBaseLoader,
        writer: KnowledgeBaseWriter,
        settings: KnowledgeBaseSettings,
    ):
        self.loader = loader
        self.writer = writer
        self.settings = settings
        self.logger = logging.getLogger(__name__)
    
    def run(self) -> None:
        """Run the knowledge base loading job."""
        self.logger.info("Starting knowledge base loading job")
        
        try:
            # Load documents from source
            documents = self.loader.load_documents(self.settings.space_key)
            self.logger.info(f"Loaded {len(documents)} documents from source")
            
            # Process documents in batches
            total_processed = 0
            for i in range(0, len(documents), self.settings.batch_size):
                batch = documents[i:i + self.settings.batch_size]
                batch_ids = self.writer.write_documents(batch)
                total_processed += len(batch_ids)
                self.logger.info(f"Processed batch {i//self.settings.batch_size + 1} ({len(batch_ids)} documents)")
            
            self.logger.info(f"Successfully completed knowledge base loading job. Total documents processed: {total_processed}")
            
        except Exception as e:
            self.logger.error(f"Error in knowledge base loading job: {str(e)}")
            raise


def create_confluence_job(settings: Settings) -> KnowledgeBaseJobRunner:
    """Factory function to create a Confluence knowledge base loading job."""    
    
    chroma_client = Chroma(
            persist_directory=settings.knowledge_base.persist_directory,
            collection_name=settings.knowledge_base.space_key,
            embedding_function=OpenAIEmbeddings(model=settings.llm.embeddings_model, api_key=settings.llm.api_key),
            create_collection_if_not_exists=True
         )
    
    confluence_client = Confluence(
            url= ConfluenceURLBuilder(settings.knowledge_base.host, settings.knowledge_base.path).build(),
            username=settings.knowledge_base.username,
            password=settings.knowledge_base.api_token,
            cloud=True
        )
    
    return KnowledgeBaseJobRunner(
        writer=ChromaWriter(chroma_client, splitter=TokenDocumentSplitter()),
        loader = ConfluenceLoader(confluence_client, ConfluenceDocumentProcessor(settings.knowledge_base.host), settings.knowledge_base.batch_size),
        settings=settings.knowledge_base
    )