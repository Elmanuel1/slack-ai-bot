# Slack AI Bot

A Slack bot that uses LangGraph and LLM capabilities to process messages and provide intelligent responses. The bot operates in two main components:

1. **Knowledge Base Loader**: Loads documents from Confluence into a vector database
2. **Slack Bot**: Processes and responds to messages in Slack using the knowledge base

## Features

- **Multi-Agent Architecture**: Uses LangGraph to coordinate between different specialized agents
- **Knowledge Base Integration**: Connects to Confluence to retrieve relevant information
- **Incident Management**: Specialized workflow for handling incident reports
- **Extensible Design**: Easy to add new capabilities and agents

## Requirements

- Python 3.9+
- Slack app credentials that has the correct priviledges to receive app mentions
- OpenAI API key (for embeddings and chat models)
- Confluence workspace (for knowledge base)
- LangSmith account (optional, for tracing)

## Installation

1. Clone this repository
2. Install dependencies and active a virtual environment:
```bash
make activate
```
3. Configure your environment variables or update the `config.yaml` file
```bash
export SLACK_BOT_TOKEN=xxxx
export SLACK_APP_TOKEN=xxxx
export SLACK_SIGNING_SECRET=xxx
export LLM_API_KEY=xxx
export LANGSMITH_API_KEY=xxx
export KNOWLEDGE_BASE_USERNAME=a@b.com
export KNOWLEDGE_BASE_API_TOKEN=xxx
export LLM_API_KEY=xxx
 
```
4. Run the knowledge base loader (one-time setup):
```bash
make load_knowledge_base

```
5. Start the Slack bot:
```bash
make run
```

## Configuration

The application uses nested configuration with two configuration methods:
- YAML file: `config.yaml` (base configuration)
- Environment variables (override YAML settings)

## Component 1: Knowledge Base Loader

This component loads documents from Confluence into a vector database for semantic search.

### Configuration

Knowledge base environment variables use the format: `KNOWLEDGE_BASE_SETTING_NAME`

Key settings:
- `KNOWLEDGE_BASE_HOST`: Your Confluence instance URL (e.g. "https://company.atlassian.net")
- `KNOWLEDGE_BASE_USERNAME`: Confluence username (email)
- `KNOWLEDGE_BASE_API_TOKEN`: Confluence API token
- `KNOWLEDGE_BASE_SPACE_KEY`: Confluence space to index (e.g. "TEAM")
- `KNOWLEDGE_BASE_PERSIST_DIRECTORY`: Where to store vector DB (default: "data/knowledge_base")
- `KNOWLEDGE_BASE_BATCH_SIZE`: Batch size for processing (default: 100)
- `KNOWLEDGE_BASE_MAX_PAGES`: Maximum pages to index (default: 1000)

### Running the Loader

To load content from Confluence:

```bash
make load_knowledge_base
```

The loader will:
1. Connect to Confluence and fetch pages
2. Process and split content into chunks
3. Generate embeddings using OpenAI
4. Store everything in a local vector database

This needs to be run once initially, and then whenever you want to update the knowledge base.

## Component 2: Slack Bot

This component handles Slack interactions, using the knowledge base to answer questions.

### Configuration

Key settings:
- `SLACK_BOT_TOKEN`: Bot token from Slack (`xoxb-` prefix)
- `SLACK_APP_TOKEN`: App token for Socket Mode (`xapp-` prefix)
- `SLACK_SIGNING_SECRET`: Signing secret from Slack app
- `SLACK_MODE`: "socket" (local development) or "webhook" (production)
- `LLM_API_KEY`: OpenAI API key
- `LLM_MODEL`: Model to use (default: "gpt-4o-mini")
- `APP_LOG_LEVEL`: Logging level (default: "INFO")

## Usage

Once configured and running, tag the bot in any Slack channel:

```
@your-bot-name Tell me about our refund policy
```

The bot will:
1. Analyze your message
2. Search the knowledge base
3. Format and return relevant information

## Project Structure

- `agents/`: Agent implementations (orchestrator, knowledge, incident)
- `knowledge_base/`: Confluence loader and vector database components
- `slack/`: Slack connection and event handling
- `events/`: Event handler implementations
- `documents/`: Document retrieval
- `config/`: Configuration
- `utils/`: Utility functions 