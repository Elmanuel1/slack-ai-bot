from events.event_handler import EventHandler

import logging
from slack_bolt import App

class AppMentionEventHandler(EventHandler):
    def __init__(self,  app: App):
        # Use __name__ for logger to get the module name
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.app = app


    def handle(self):
        """Set up the Slack event handler."""
        @self.app.event("app_mention")
        def handle_mention(event, say):
            user = event["user"]
            text = event["text"]
            self.logger.info(f"Received mention from {user}: {text}")
            say(f"Hi there, <@{user}>! How can I help you today?")
