import logging

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from config.settings import Settings
from events.event_handler import EventHandler


class SlackEventsHandler:
    def __init__(self, settings: Settings, event_handlers: list[EventHandler], app: App):
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        self.app = app
        self.event_handlers = event_handlers

    def setup_events(self):
        """Define Slack event listeners."""
        for handler in self.event_handlers:
            handler.handle()

    def start(self):
        """Start the Slack bot."""
        self.setup_events()

        if self.settings.slack.mode == "socket":
            self.logger.info("Starting Slack bot in Socket Mode")
            # Start the Slack bot in Socket Mode
            handler = SocketModeHandler(self.app, self.settings.slack.app_token)
            handler.start()
            return

        # Start the Slack bot in Bolt App Mode
        self.app.start(self.settings.slack.port)