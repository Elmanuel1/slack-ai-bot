import logging
import os
import sys
from config.settings import Settings
from events.app_mention_handler import AppMentionEventHandler
from slack.slack_events_handlers import SlackEventsHandler
from slack_bolt import App
from utils.llm import init_chat_model
from agents.langgraph_agent import LangGraphAgent
from agents.incident_agent import IncidentAgent
from agents.knowledge_agent import KnowledgeAgent
from agents.direct_agent import DirectAgent
from langgraph.checkpoint.memory import MemorySaver
from documents.document_retriever import DocumentRetriever
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from config.logging import Logger
if __name__ == "__main__":

    settings = Settings()

    Logger(settings).configure_logger()

    os.environ["LANGSMITH_TRACING"] = str(settings.langsmith.tracing)
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith.api_key

    app = App(token=settings.slack.bot_token, signing_secret=settings.slack.signing_secret)
     # Create LLM
    llm = init_chat_model(settings.llm)

    memory_saver = MemorySaver()

    chroma_client = Chroma(
            persist_directory=settings.knowledge_base.persist_directory,
    
            collection_name=settings.knowledge_base.space_key,

            embedding_function=OpenAIEmbeddings(model=settings.llm.embeddings_model, api_key=settings.llm.api_key),

            create_collection_if_not_exists=True
         )
     
     # Initialize event handlers with LLM
    langgraph_workflow = LangGraphAgent(
          llm=llm, 

          checkpoint_saver=memory_saver, 

          incident_workflow=IncidentAgent(llm=llm, checkpoint_saver=memory_saver).build(), 

          knowledge_workflow=KnowledgeAgent(
               llm=llm, 
               checkpoint_saver=memory_saver, 
               document_retriever=DocumentRetriever(chroma_client)
          ).build()

     ).build()

    slack_event_handler = SlackEventsHandler(
         settings, 
         [AppMentionEventHandler(app, settings, langgraph_workflow)], 
         app
     )
    
    slack_event_handler.start()
