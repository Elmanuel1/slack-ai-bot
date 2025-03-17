import logging

from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml

class AppSettings(BaseSettings):
    name: str
    version: str = "1.0.0"
    port: int = 8080
    host: str = "0.0.0.0"
    log_level: str = "INFO"
    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_file=".env",
        env_file_encoding="utf-8"
    )

class SlackSettings(BaseSettings):
    bot_token: str
    app_token: str
    mode: str
    signing_secret: str# Required for Socket Mode
    model_config = SettingsConfigDict(
        env_prefix="SLACK_",
        env_file=".env",
        env_file_encoding="utf-8"
    )

class Settings(BaseSettings):
    app: AppSettings
    slack: SlackSettings
    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        extra="ignore",
        env_file=".env",
        env_file_encoding="utf-8"
    )

    def __init__(self, **data):
        # Load YAML configuration
        logging.info("Data: " + str(data))
        yaml_path = "config.yaml"
        with open(yaml_path, "r") as file:
            yaml_config = yaml.safe_load(file)

        # Merge YAML config with provided data
        # This ensures YAML is the baseline but can be overridden
        merged_data = {}
        merged_data.update(yaml_config)
        merged_data.update(data)

        # Let Pydantic handle everything, including env vars
        super().__init__(**merged_data)