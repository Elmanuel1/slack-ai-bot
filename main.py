import logging
from config.settings import Settings
from events.app_mention_handler import AppMentionEventHandler
from slack.slack_events_handlers import SlackEventsHandler
from slack_bolt import App
from utils.llm import ChatModelFactory

if __name__ == "__main__":
     settings = Settings()
     app = App(token=settings.slack.bot_token, signing_secret=settings.slack.signing_secret)
     
     # Create LLM
     llm = ChatModelFactory.create(settings.chat)
     # Initialize event handlers with LLM
     slack_event_handler = SlackEventsHandler(
         settings, 
         [AppMentionEventHandler(app, settings, llm)], 
         app
     )
     slack_event_handler.start()
