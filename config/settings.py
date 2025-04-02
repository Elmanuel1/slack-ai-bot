import logging
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
import yaml
from pydantic_settings import BaseSettings

class AppSettings(BaseSettings):
    name: str
    version: str = "1.0.0"
    port: int = 8080
    host: str = "0.0.0.0"
    log_level: str = "INFO"
    
    model_config = ConfigDict(
        env_prefix="APP_",
        env_file=".env",
        env_file_encoding="utf-8"
    )

class LLMSettings(BaseSettings):
    """Settings for chat model configuration."""
    provider: str = Field(default="openai", description="The chat provider to use")
    model: str = Field(default="gpt-4", description="The model to use")
    temperature: float = Field(default=0.0, description="The temperature for model outputs")
    api_key: str = Field(description="The API key for the chat provider")
    embeddings_model: str = Field(default="text-embedding-3-large", description="The embeddings model to use")

    model_config = ConfigDict(
        env_prefix="LLM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

class SlackSettings(BaseSettings):
    bot_token: str = Field(description="Slack bot token")
    app_token: str = Field(description="Slack app token")
    mode: str = Field(description="Slack mode. socket or webhook")
    signing_secret: str = Field(description="Slack signing secret")
    
    model_config = ConfigDict(
        env_prefix="SLACK_",
        env_file=".env",
        env_file_encoding="utf-8"
    )

class knowledgeBaseSettings(BaseSettings):
    """Knowledge base settings."""
    persist_directory: str = "data/knowledge_base"
    host: str = Field(description="Host of the Knowledge Base (e.g., https://your-domain.atlassian.net)")
    path: str = Field( description="Base path for Knowledge Base")
    username: str = Field(description="Username for the Knowledge Base")
    api_token: str = Field(description="API token for the Knowledge Base")
    space_key: str = Field(description="Space key for the Knowledge Base")
    max_pages: int = Field(default=1000, description="Maximum number of pages to load")
    batch_size: int = Field(default=100, description="Batch size for the Knowledge Base")
    model_config = ConfigDict(
        env_prefix="KNOWLEDGE_BASE_",
        env_file=".env",
        env_file_encoding="utf-8"
    )

class LangsmithSettings(BaseSettings):
    """Langsmith settings."""
    tracing: bool = Field(default=True, description="Enable tracing")
    api_key: str = Field(description="API key for Langsmith")
    model_config = ConfigDict(
        env_prefix="LANGSMITH_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


class Settings(BaseSettings):
    """Main application settings."""
    app: AppSettings = Field(default_factory=AppSettings)
    slack: SlackSettings = Field(default_factory=SlackSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    langsmith: LangsmithSettings = Field(default_factory=LangsmithSettings)
    knowledge_base: knowledgeBaseSettings = Field(default_factory=knowledgeBaseSettings)

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def __init__(self, **data: Any) -> None:
        # Load YAML configuration
        yaml_path = "config.yaml"
        with open(yaml_path, "r") as file:
            yaml_config = yaml.safe_load(file)

        # Merge YAML config with provided data
        # This ensures YAML is the baseline but can be overridden
        merged_data: Dict[str, Any] = {}
        merged_data.update(yaml_config)
        merged_data.update(data)

        # Let Pydantic handle everything, including env vars
        super().__init__(**merged_data)