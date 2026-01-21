# Moda Python SDK

LLM Observability with Automatic Conversation Threading

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Installation

```bash
pip install moda-ai
```

## Quick Start

```python
import moda
from openai import OpenAI

# Initialize Moda with your API key
moda.init("moda_xxx")

# Use OpenAI as normal - all calls are automatically tracked
client = OpenAI()

# Turn 1
messages = [{"role": "user", "content": "Hello!"}]
response = client.chat.completions.create(
    model="gpt-4",
    messages=messages
)

# Turn 2 - automatically tracked with same conversation_id
messages.append({"role": "assistant", "content": response.choices[0].message.content})
messages.append({"role": "user", "content": "How are you?"})
response = client.chat.completions.create(
    model="gpt-4",
    messages=messages
)

# Ensure all traces are sent
moda.flush()
```

## Features

### Automatic Conversation Threading

Moda automatically computes a stable `conversation_id` for each conversation based on:
- The first user message
- The system prompt (if present)

This means multi-turn conversations automatically get the same conversation ID, enabling powerful analytics and debugging.

### User Attribution

Track which user is making LLM calls:

```python
with moda.set_user_id("user-123"):
    # All LLM calls here are attributed to user-123
    client.chat.completions.create(...)
```

### Explicit Conversation IDs

Override automatic conversation ID computation when needed:

```python
with moda.set_conversation_id("my-custom-id"):
    # All LLM calls here use "my-custom-id"
    client.chat.completions.create(...)
```

## Supported Providers

- **OpenAI** (via `moda-openai`)
- **Anthropic** (via `moda-anthropic`)

## Environment Variables

| Variable | Description |
|----------|-------------|
| `MODA_API_KEY` | Your Moda API key |
| `MODA_BASE_URL` | Custom endpoint (default: `https://ingest.moda.so/v1/traces`) |

## API Reference

### `moda.init(api_key, **kwargs)`

Initialize the Moda SDK.

**Parameters:**
- `api_key` (str): Your Moda API key
- `app_name` (str): Name of your application
- `endpoint` (str): Custom ingest endpoint
- `enabled` (bool): Enable/disable instrumentation

### `moda.set_conversation_id(id)`

Context manager to set explicit conversation ID.

```python
with moda.set_conversation_id("conv-123"):
    # LLM calls here use "conv-123"
```

### `moda.set_user_id(id)`

Context manager to set user ID for attribution.

```python
with moda.set_user_id("user-456"):
    # LLM calls here are attributed to "user-456"
```

### `moda.flush()`

Force flush all pending spans.

```python
moda.flush()  # Ensure traces are sent before exit
```

## Span Attributes

Moda adds these attributes to LLM spans:

| Attribute | Description |
|-----------|-------------|
| `moda.conversation_id` | Stable conversation identifier |
| `moda.user_id` | User identifier (if set) |

## Development

### Setup

```bash
# Clone the repository
git clone https://github.com/ModaLabs/moda-python.git
cd moda-python

# Install dependencies
cd packages/moda
uv sync --all-extras --dev
```

### Running Tests

```bash
# Unit tests
uv run pytest tests/test_conversation.py -v

# Integration tests
uv run pytest tests/test_integration.py -v
```

### Linting

```bash
uv run ruff check traceloop/
```

## Based on OpenLLMetry

This SDK is based on [OpenLLMetry](https://github.com/traceloop/openllmetry), an open-source observability framework built on OpenTelemetry. We've extended it with automatic conversation threading capabilities.

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.
