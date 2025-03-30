from langchain_core.language_models import BaseChatModel
from config.settings import ChatSettings
from langchain_openai import ChatOpenAI
from typing import Dict, Type, Callable


class ChatModelFactory:
    """Factory class for creating chat model instances."""
    
    _providers: Dict[str, Type[BaseChatModel]] = {
        "openai": ChatOpenAI,
    }
    
    _init_kwargs: Dict[str, Callable[[ChatSettings], Dict[str, str]]] = {
        "openai": lambda settings: {
            "model": settings.model,
            "temperature": settings.temperature,
            "openai_api_key": settings.api_key
        }
    }
    
    @classmethod
    def create(cls, settings: ChatSettings) -> BaseChatModel:
        """
        Create a chat model instance based on the provider specified in settings.
        
        Args:
            settings: ChatSettings object containing provider and configuration
            
        Returns:
            An instance of BaseChatModel
            
        Raises:
            ValueError: If the provider is not supported
        """
        provider = settings.provider.lower()
        
        if provider not in cls._providers:
            raise ValueError(
                f"Unsupported chat provider: {provider}. "
                f"Supported providers are: {', '.join(cls._providers.keys())}"
            )
            
        model_class = cls._providers[provider]
        init_kwargs = cls._init_kwargs[provider](settings)
        return model_class(**init_kwargs)