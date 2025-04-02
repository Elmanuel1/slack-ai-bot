import logging

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from config.settings import Settings
from events.event_handler import EventHandler


class SlackEventsHandler:
    """Manages Slack event handling and bot initialization.
    
    This class serves as the main entry point for the Slack bot, handling
    initialization, event registration, and starting the bot in either Socket Mode
    or Bolt App Mode. It delegates specific event handling to specialized
    event handler classes.
    """
    
    def __init__(self, settings: Settings, event_handlers: list[EventHandler], app: App):
        """Initialize the Slack events handler.
        
        Args:
            settings (Settings): Configuration settings for the Slack bot.
            event_handlers (list[EventHandler]): List of specialized event handlers.
            app (App): The Slack Bolt app instance.
            
        Example:
            >>> app = App(token=settings.slack.bot_token, signing_secret=settings.slack.signing_secret)
            >>> handlers = [AppMentionEventHandler(app, settings, agent)]
            >>> slack_handler = SlackEventsHandler(settings, handlers, app)
            >>> slack_handler.start()
        """
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        self.app = app
        self.event_handlers = event_handlers

    def setup_events(self):
        """Define Slack event listeners.
        
        Registers all event handlers with the Slack app by calling each handler's
        handle method. This allows different components to register their own
        event listeners without the main handler needing to know the details.
        """
        for handler in self.event_handlers:
            handler.handle()

    def start(self):
        """Start the Slack bot.
        
        Sets up event handlers and starts the Slack bot in either Socket Mode
        or Bolt App Mode based on configuration settings.
        
        Socket Mode is used for development and doesn't require public endpoints,
        while Bolt App Mode is used for production with webhook endpoints.
        """
        self.setup_events()

        if self.settings.slack.mode == "socket":
            self.logger.info("Starting Slack bot in Socket Mode")
            # Start the Slack bot in Socket Mode
            handler = SocketModeHandler(self.app, self.settings.slack.app_token)
            handler.start()
            return

        # Start the Slack bot in Bolt App Mode
        self.app.start(self.settings.slack.port)