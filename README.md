# Slack AI Bot

A Slack bot that uses LangGraph and LLM capabilities to process messages and provide intelligent responses. The bot can handle different types of queries, retrieve information from a knowledge base, and manage incidents.

## Features

- **Multi-Agent Architecture**: Uses LangGraph to coordinate between different specialized agents
- **Knowledge Base Integration**: Connects to Confluence to retrieve relevant information
- **Incident Management**: Specialized workflow for handling incident reports
- **Extensible Design**: Easy to add new capabilities and agents

## Requirements

- Python 3.9+
- Slack workspace with admin privileges
- Access to an LLM provider (default: OpenAI)
- Confluence workspace (optional, for knowledge base)
- LangSmith account (optional, for tracing)

## Installation

1. Clone this repository
2. Install dependencies:
```bash
make install 
make activate
```
Run the source command generated from the activate command to activate the vritual environment

3. Configure your environment variables or update the `config.yaml` file (see Configuration section)
4. Run the bot:
```bash
python main.py
```

## Configuration

The bot can be configured through environment variables or by editing the `config.yaml` file. Environment variables will override values in the config file.

### Environment Variables

#### App Configuration
- `APP__NAME`: Name of the application (default: "passport_ai")
- `APP__VERSION`: Version of the application (default: "1.0.0")
- `APP__PORT`: Port to run the service on (default: 3000)
- `APP__HOST`: Host to run the service on (default: "0.0.0.0")
- `APP__LOG_LEVEL`: Logging level (default: "INFO")

#### Slack Configuration
- `SLACK__MODE`: Mode for Slack connection - "socket" for local development, "webhook" for production (default: "socket")
- `SLACK__BOT_TOKEN`: Bot token from Slack (`xoxb-` prefix)
- `SLACK__APP_TOKEN`: App token from Slack (`xapp-` prefix) - required for Socket Mode
- `SLACK__SIGNING_SECRET`: Signing secret from Slack app

#### LLM Provider Configuration. Only OpenAPI is supported for now
- `LLM__PROVIDER`: LLM provider to use (default: "openai")
- `LLM__MODEL`: Model to use (default: "gpt-4o-mini")
- `LLM__TEMPERATURE`: Temperature for generation (default: 0.0)
- `LLM__API_KEY`: API key for the LLM provider
- `LLM__EMBEDDINGS_MODEL`: Embeddings model to use (default: "text-embedding-3-large")

#### LangSmith Tracing
- `LANGSMITH__TRACING`: Enable LangSmith tracing (default: true)
- `LANGSMITH__API_KEY`: API key for LangSmith

#### Knowledge Base Configuration (Confluence)
- `KNOWLEDGE__BASE_PERSIST_DIRECTORY`: Directory to store vector embeddings (default: "data/knowledge_base")
- `KNOWLEDGE__BASE_HOST`: Confluence host URL (e.g., "https://your-domain.atlassian.net")
- `KNOWLEDGE__BASE_USERNAME`: Confluence username (email)
- `KNOWLEDGE__BASE_API_TOKEN`: Confluence API token
- `KNOWLEDGE__BASE_SPACE_KEY`: Confluence space key (default: "PROPWISE")
- `KNOWLEDGE__BASE_BATCH_SIZE`: Batch size for processing documents (default: 100)
- `KNOWLEDGE__BASE_MAX_PAGES`: Maximum number of pages to load (default: 1000)

### Slack Configuration Steps

1. Create a new Slack app at https://api.slack.com/apps
2. Under "OAuth & Permissions", add the following bot token scopes:
   - `app_mentions:read`
   - `chat:write`
   - `commands`
3. Install the app to your workspace
4. Copy the Bot Token, App Token, and Signing Secret to your config or environment variables

### LLM Provider Configuration

The default LLM provider is OpenAI. To use it:

1. Get an API key from OpenAI
2. Set `LLM__API_KEY` to your OpenAI API key
3. Optionally change `LLM__MODEL` to use a different model (default: "gpt-4o-mini")

For other providers, update the `LLM__PROVIDER` setting and the appropriate API key.

### Knowledge Base Configuration

The bot uses Confluence as its knowledge source. To set it up:

1. Generate an API token in Confluence
2. Configure the following settings:
   - `KNOWLEDGE__BASE_HOST`: Your Confluence URL
   - `KNOWLEDGE__BASE_USERNAME`: Your Confluence username
   - `KNOWLEDGE__BASE_API_TOKEN`: Your Confluence API token
   - `KNOWLEDGE__BASE_SPACE_KEY`: Space key for your knowledge base

### Logging

The default log level is `INFO`. You can change it by setting `APP__LOG_LEVEL` to one of:
- `DEBUG`
- `INFO`
- `WARNING`
- `ERROR`
- `CRITICAL`

## Usage

Once configured and running, tag the bot in any Slack channel it's been invited to:

```
@your-bot-name Tell me about our refund policy
```

The bot will automatically:
1. Analyze your message
2. Route it to the appropriate agent
3. Process the request (search knowledge base, handle incident, etc.)
4. Respond in the thread

## Project Structure

- `agents/`: Contains different agent implementations
  - `base_agent.py`: Base class for all agents
  - `langgraph_agent.py`: Main orchestrator agent
  - `knowledge_agent.py`: Agent for knowledge base queries
  - `incident_agent.py`: Agent for incident management
  - `direct_agent.py`: Agent for general queries
- `slack/`: Slack integration code
- `events/`: Event handlers for different message types
- `documents/`: Knowledge base integration
- `config/`: Configuration utilities
- `utils/`: Utility functions 