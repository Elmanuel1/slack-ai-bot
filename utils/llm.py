from langchain_core.language_models import BaseChatModel
from config.settings import LLMSettings
from langchain_openai import ChatOpenAI
from typing import Dict, Type, Callable, Optional


class ChatModelFactory:
    """Factory class for creating chat model instances."""
    
    def __init__(
        self,
        providers: Optional[Dict[str, Type[BaseChatModel]]] = None,
        init_kwargs: Optional[Dict[str, Callable[[LLMSettings], Dict[str, str]]]] = None
    ):
        """
        Initialize the factory with optional provider and initialization configurations.
        
        Args:
            providers: Dictionary mapping provider names to their model classes
            init_kwargs: Dictionary mapping provider names to their initialization functions
        """
        self._providers = providers or {
            "openai": ChatOpenAI,
        }
        
        self._init_kwargs = init_kwargs or {
            "openai": self._get_openai_kwargs
        }
    
    def _get_openai_kwargs(self, settings: LLMSettings) -> Dict[str, str]:
        """Get initialization kwargs for OpenAI provider."""
        return {
            "model": settings.model,
            "temperature": settings.temperature,
            "openai_api_key": settings.api_key
        }
    
    def create(self, settings: LLMSettings) -> BaseChatModel:
        """
        Create a LLM instance based on the provider specified in settings.
        
        Args:
            settings: LLMSettings object containing provider and configuration
            
        Returns:
            An instance of BaseChatModel
            
        Raises:
            ValueError: If the provider is not supported
        """
        provider = settings.provider.lower()
        
        if provider not in self._providers:
            raise ValueError(
                f"Unsupported LLM provider: {provider}. "
                f"Supported providers are: {', '.join(self._providers.keys())}"
            )
            
        model_class = self._providers[provider]
        init_kwargs = self._init_kwargs[provider](settings)
        return model_class(**init_kwargs)


def init_chat_model(settings: LLMSettings) -> BaseChatModel:
    """Initialize the chat model using the factory."""
    factory = ChatModelFactory()
    return factory.create(settings)