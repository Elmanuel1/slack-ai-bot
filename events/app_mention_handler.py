from events.event_handler import EventHandler
import logging
from slack_bolt import App
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver
from config.settings import Settings
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState
from langgraph.graph import Graph

class AppMentionEventHandler(EventHandler):
    """Handler for Slack app_mention events.
    
    This handler processes mentions of the bot in Slack channels and responds using
    the configured LangGraph workflow. It converts Slack messages to LangChain
    message format, processes them through the workflow, and sends responses back
    to the appropriate thread in Slack.
    """
    
    def __init__(self, app: App, settings: Settings, langgraph_workflow: Graph):
        """Initialize the app mention event handler.
        
        Args:
            app (App): The Slack Bolt app instance.
            settings (Settings): Configuration settings.
            langgraph_workflow (Graph): The compiled LangGraph workflow to process messages.
            
        Example:
            >>> app = App(token=settings.slack.bot_token, signing_secret=settings.slack.signing_secret)
            >>> workflow = agent.build()
            >>> handler = AppMentionEventHandler(app, settings, workflow)
            >>> handler.handle()
        """
        # Use __name__ for logger to get the module name
        self.logger = logging.getLogger(__name__)
        self.app = app
        self.settings = settings
        
        # Initialize the LangGraph agent and build its graph
        self.workflow = langgraph_workflow

    def handle(self):
        """Set up the Slack event handler.
        
        Registers a callback for the app_mention event that processes
        mentions of the bot in Slack channels using the LangGraph workflow.
        """
        @self.app.event("app_mention")
        def handle_mention(event, say):
            """Process a Slack app_mention event.
            
            This function extracts the user, message text, and thread information
            from the event, processes the message through the LangGraph workflow,
            and sends the response back to the Slack thread.
            
            Args:
                event (dict): The Slack event data.
                say (callable): Function to send messages back to Slack.
            """
            user = event["user"]
            text = event["text"]
            thread_ts = event.get("thread_ts", event.get("ts"))  # Get thread_ts or fallback to message ts
            self.logger.debug("Received mention from %s: %s", user, text)
            
            try:
                # Create initial state with the message
                initial_state = MessagesState(
                    messages=[HumanMessage(content=text)]
                )
                self.logger.debug("Initial state: %s", initial_state)
                
                # Create config from the event
                config = {
                    "configurable": {
                        "thread_id": thread_ts
                    }
                }
                
                # Invoke the workflow
                final_state = self.workflow.invoke(
                    initial_state,
                    config
                )
                self.logger.debug("Final state: %s", final_state)
                
                # Get the response from the final state
                response = final_state["messages"][-1].content
                self.logger.debug("Response: %s", response)
                
                # Send the response back to Slack in the same thread
                say(text=response, thread_ts=thread_ts)
                
            except Exception as e:
                self.logger.error("Error processing message: %s", str(e))
                say(text="I apologize, but I encountered an error processing your message. Please try again later.", thread_ts=thread_ts)
