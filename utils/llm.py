from langchain_core.language_models import BaseChatModel
from config.settings import LLMSettings
from langchain_openai import ChatOpenAI
from typing import Dict, Type, Callable, Optional


class ChatModelFactory:
    """Factory class for creating chat model instances.
    
    This factory provides a flexible way to create different language model
    instances based on configuration settings. It supports multiple providers
    and handles the specific initialization requirements for each.
    
    The factory pattern allows the application to work with different LLM
    providers without needing to know the specific implementation details
    of each provider.
    """
    
    def __init__(
        self,
        providers: Optional[Dict[str, Type[BaseChatModel]]] = None,
        init_kwargs: Optional[Dict[str, Callable[[LLMSettings], Dict[str, str]]]] = None
    ):
        """Initialize the factory with optional provider and initialization configurations.
        
        Args:
            providers (Optional[Dict[str, Type[BaseChatModel]]]): Dictionary mapping provider 
                names to their model classes. Defaults to {'openai': ChatOpenAI}.
            init_kwargs (Optional[Dict[str, Callable]]): Dictionary mapping provider names 
                to functions that return initialization parameters. Defaults to using internal 
                OpenAI initialization.
                
        Example:
            >>> factory = ChatModelFactory()
            >>> # Or with custom providers
            >>> factory = ChatModelFactory(
            ...     providers={"openai": ChatOpenAI, "anthropic": ChatAnthropic},
            ...     init_kwargs={
            ...         "openai": lambda s: {"model": s.model, "api_key": s.api_key},
            ...         "anthropic": lambda s: {"model": s.model, "api_key": s.anthropic_key}
            ...     }
            ... )
        """
        self._providers = providers or {
            "openai": ChatOpenAI,
        }
        
        self._init_kwargs = init_kwargs or {
            "openai": self._get_openai_kwargs
        }
    
    def _get_openai_kwargs(self, settings: LLMSettings) -> Dict[str, str]:
        """Get initialization kwargs for OpenAI provider.
        
        Extracts the relevant parameters from LLMSettings and formats them
        for use with the OpenAI chat model.
        
        Args:
            settings (LLMSettings): Configuration settings for the language model.
            
        Returns:
            Dict[str, str]: Dictionary of keyword arguments for ChatOpenAI initialization.
        """
        return {
            "model": settings.model,
            "temperature": settings.temperature,
            "openai_api_key": settings.api_key
        }
    
    def create(self, settings: LLMSettings) -> BaseChatModel:
        """Create a LLM instance based on the provider specified in settings.
        
        This method selects the appropriate model class based on the provider
        specified in the settings and initializes it with the correct parameters.
        
        Args:
            settings (LLMSettings): Configuration settings for the language model.
            
        Returns:
            BaseChatModel: An initialized language model instance.
            
        Raises:
            ValueError: If the specified provider is not supported.
            
        Example:
            >>> llm_settings = LLMSettings(provider="openai", model="gpt-4", temperature=0.7)
            >>> llm = factory.create(llm_settings)
            >>> response = llm.invoke("Hello, how are you?")
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
    """Initialize the chat model using the factory.
    
    Convenience function to create a language model instance using
    the ChatModelFactory with the provided settings.
    
    Args:
        settings (LLMSettings): Configuration settings for the language model.
        
    Returns:
        BaseChatModel: An initialized language model instance.
        
    Example:
        >>> from config.settings import Settings
        >>> settings = Settings()
        >>> llm = init_chat_model(settings.llm)
        >>> response = llm.invoke("What can you help me with?")
    """
    factory = ChatModelFactory()
    return factory.create(settings)