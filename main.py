import os
from config.settings import Settings
from events.app_mention_handler import AppMentionEventHandler
from slack.slack_events_handlers import SlackEventsHandler
from slack_bolt.async_app import AsyncApp
from utils.llm import init_chat_model
from agents.main_agent import MainAgent
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

    app = AsyncApp(token=settings.slack.bot_token, signing_secret=settings.slack.signing_secret)
    
    # Create LLM
    llm = init_chat_model(settings.llm)

    memory_saver = MemorySaver()

    chroma_client = Chroma(
            persist_directory=settings.knowledge_base.persist_directory,
            collection_name=settings.knowledge_base.space_key,
            embedding_function=OpenAIEmbeddings(model=settings.llm.embeddings_model, api_key=settings.llm.api_key),
            create_collection_if_not_exists=True
    )

    # Create individual agent instances
    incident_agent = IncidentAgent(llm=llm, checkpoint_saver=memory_saver)
    knowledge_agent = KnowledgeAgent(
        llm=llm, 
        checkpoint_saver=memory_saver, 
        document_retriever=DocumentRetriever(chroma_client)
    )
    direct_agent = DirectAgent(llm=llm, checkpoint_saver=memory_saver)
    
    # Initialize the main agent with all workflows
    main_workflow = MainAgent(
        llm=llm,
        checkpoint_saver=memory_saver,
        agents=[incident_agent, knowledge_agent, direct_agent],
        default_agent_key=knowledge_agent.key  # Set knowledge as the default fallback
    ).build()

    slack_event_handler = SlackEventsHandler(
         settings, 
         [AppMentionEventHandler(app, settings, main_workflow)], 
         app
    )
    
    slack_event_handler.start()
