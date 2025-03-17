import logging

from config.settings import Settings
from events.app_mention_handler import AppMentionEventHandler
from slack.slack_events_handlers import SlackEventsHandler
from slack_bolt import App

if __name__ == "__main__":
     settings = Settings()
     app = App(token=settings.slack.bot_token, signing_secret=settings.slack.signing_secret)

     slack_event_handler = SlackEventsHandler(settings, [AppMentionEventHandler(app)], app)
     slack_event_handler.start()
