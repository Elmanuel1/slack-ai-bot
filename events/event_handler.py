# Event Handler Interface
from abc import abstractmethod, ABC


class EventHandler(ABC):
    """Abstract base class for event handlers.
    
    This interface defines the contract that all event handlers must implement.
    Event handlers are responsible for registering callbacks with external systems
    (like Slack) and processing events from those systems.
    
    Different event types should have their own dedicated handler classes that
    implement this interface. This allows for a clean separation of concerns
    and makes the system more modular and maintainable.
    """
    
    @abstractmethod
    def handle(self):
        """Set up and register event handlers.
        
        This method should be implemented by concrete event handler classes to
        register callbacks or listeners for specific events. For example, a Slack
        event handler might register callbacks for message events or app_mention events.
        
        Returns:
            None
            
        Example:
            >>> class ConcreteEventHandler(EventHandler):
            ...     def handle(self):
            ...         @app.event("app_mention")
            ...         def process_mention(event, say):
            ...             # Process the event
            ...             say("I received your mention!")
        """
        pass