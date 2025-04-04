import logging
import asyncio

from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler

from config.settings import Settings
from events.event_handler import EventHandler


class SlackEventsHandler:
    """Manages Slack event handling and bot initialization.
    
    This class serves as the main entry point for the Slack bot, handling
    initialization, event registration, and starting the bot in either Socket Mode
    or Bolt App Mode. It delegates specific event handling to specialized
    event handler classes.
    """
    
    def __init__(self, settings: Settings, event_handlers: list[EventHandler], app: AsyncApp):
        """Initialize the Slack events handler.
        
        Args:
            settings (Settings): Configuration settings for the Slack bot.
            event_handlers (list[EventHandler]): List of specialized event handlers.
            app (AsyncApp): The Slack Bolt async app instance.
            
        Example:
            >>> app = AsyncApp(token=settings.slack.bot_token, signing_secret=settings.slack.signing_secret)
            >>> handlers = [AppMentionEventHandler(app, settings, agent)]
            >>> slack_handler = SlackEventsHandler(settings, handlers, app)
            >>> slack_handler.start()
        """
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        self.app = app
        self.event_handlers = event_handlers

    async def setup_events(self):
        """Define Slack event listeners.
        
        Registers all event handlers with the Slack app by calling each handler's
        handle method. This allows different components to register their own
        event listeners without the main handler needing to know the details.
        """
        for handler in self.event_handlers:
            handler.handle()

    async def _start_socket_mode(self):
        """Start the bot in Socket Mode.
        
        Creates and starts the AsyncSocketModeHandler for Socket Mode operation.
        """
        self.logger.info("Starting Slack bot in Socket Mode")
        handler = AsyncSocketModeHandler(self.app, self.settings.slack.app_token)
        await handler.start_async()
        
    async def _start_bolt_app(self):
        """Start the bot in Bolt App Mode.
        
        Starts the Slack Bolt app on the configured port.
        """
        self.logger.info("Starting Slack bot in Bolt App Mode")
        await self.app.start_async(port=self.settings.slack.port)

    def start(self):
        """Start the Slack bot.
        
        Sets up event handlers and starts the Slack bot in either Socket Mode
        or Bolt App Mode based on configuration settings.
        
        Socket Mode is used for development and doesn't require public endpoints,
        while Bolt App Mode is used for production with webhook endpoints.
        """
        # Create and get the event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Define the main async function
        async def main():
            # Setup events
            await self.setup_events()
            
            # Start the bot in the appropriate mode
            if self.settings.slack.mode == "socket":
                await self._start_socket_mode()
            else:
                await self._start_bolt_app()
        
        # Run the async main function until complete
        loop.run_until_complete(main())