from typing import Optional, Type, Tuple
from pydantic import Field, ConfigDict
from pydantic_settings import BaseSettings
from pydantic_settings.sources import YamlConfigSettingsSource, PydanticBaseSettingsSource

class BaseAppSettings(BaseSettings):
    """Base settings class with common functionality for all settings classes.
    This class is used to define the common settings for all settings classes.
    It is also used to define the custom sources for the settings.
    Environment variables have the highest priority.
    .env file has the next priority.
    config.yaml file has the next priority. This will be used for local development and properties that do not need to be changed frequently
    """
    
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        yaml_source = YamlConfigSettingsSource(
            settings_cls=settings_cls, 
            yaml_file="config.yaml"
        )
        
        return (
            env_settings,
            init_settings,
            file_secret_settings,
            dotenv_settings,
            yaml_source
        )
    
    @classmethod
    def model_config_with_prefix(cls, prefix: str) -> ConfigDict:
        return ConfigDict(
            extra="ignore",
            env_nested_delimiter="_",
            env_file=".env",
            env_file_encoding="utf-8",
            case_sensitive=False,
            env_prefix=prefix,
        )

class AppSettings(BaseAppSettings):
    name: str = Field(description="The name of the application")  
    version: str = Field(description="The version of the application")
    port: int = Field(description="The port to run the application on")
    host: str = Field(description="The host to run the application on")
    log_level: str = Field(description="The log level to use")
    
    model_config = BaseAppSettings.model_config_with_prefix("app_")

class LLMSettings(BaseAppSettings):
    provider: str = Field(description="The chat provider to use")
    model: str = Field(description="The model to use")
    temperature: float = Field(description="The temperature for model outputs")
    api_key: str = Field(description="The API key for the chat provider")
    embeddings_model: str = Field(description="The embeddings model to use")
    model_config = BaseAppSettings.model_config_with_prefix("llm_")

class SlackSettings(BaseAppSettings):
    bot_token: str = Field(description="Slack bot token")
    app_token: str = Field(description="Slack app token")
    mode: str = Field(description="Slack mode. socket or webhook")
    signing_secret: str = Field(description="Slack signing secret")
    port: int = Field(description="Slack port")
    
    model_config = BaseAppSettings.model_config_with_prefix("slack_")

class KnowledgeBaseSettings(BaseAppSettings):
    persist_directory: str = Field(description="Directory to store the knowledge base. Useful for local development.")
    host: str = Field(description="Host of the Knowledge Base")
    path: str = Field(description="Base path for Knowledge Base")
    username: str = Field(description="Username for the Knowledge Base")
    api_token: str = Field(description="API token for the Knowledge Base")
    space_key: str = Field(description="Space key for the Knowledge Base")
    batch_size: int = Field(description="Batch size for the Knowledge Base")
    
    model_config = BaseAppSettings.model_config_with_prefix("knowledge_base_")

class LangsmithSettings(BaseAppSettings):
    tracing: Optional[bool] = Field(default=True)
    api_key: str = Field(description="API key for Langsmith")
    
    model_config = BaseAppSettings.model_config_with_prefix("langsmith_")

class Settings(BaseAppSettings):
    app: AppSettings = Field(default_factory=AppSettings)
    slack: SlackSettings = Field(default_factory=SlackSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    langsmith: LangsmithSettings = Field(default_factory=LangsmithSettings)
    knowledge_base: KnowledgeBaseSettings = Field(default_factory=KnowledgeBaseSettings)
