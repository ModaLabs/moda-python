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

# Set conversation ID for your session (recommended)
moda.conversation_id = "session_" + session_id

# Use OpenAI as normal - all calls are automatically tracked
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)

# Ensure all traces are sent
moda.flush()
```

## Features

### Conversation Tracking

#### Setting Conversation ID (Recommended)

For production use, explicitly set a conversation ID to group related LLM calls:

```python
# Property-style (recommended)
moda.conversation_id = "support_ticket_123"
client.chat.completions.create(...)
moda.conversation_id = None  # clear when done

# Or use the setter function
moda.set_conversation_id_value("support_ticket_123")

# Or use context manager (scoped)
with moda.set_conversation_id("support_ticket_123"):
    client.chat.completions.create(...)
```

#### Setting User ID

Associate LLM calls with specific users:

```python
moda.user_id = "user_12345"
client.chat.completions.create(...)
moda.user_id = None  # clear
```

### Automatic Fallback

If you don't set a conversation ID, the SDK automatically computes a stable one from:
- The first user message
- The system prompt (if present)

This works for simple use cases but explicit IDs are recommended for production.

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

### `moda.conversation_id`

Property to get/set conversation ID (recommended).

```python
moda.conversation_id = "conv-123"  # set
print(moda.conversation_id)        # get
moda.conversation_id = None        # clear
```

### `moda.user_id`

Property to get/set user ID.

```python
moda.user_id = "user-456"  # set
print(moda.user_id)        # get
moda.user_id = None        # clear
```

### `moda.set_conversation_id(id)`

Context manager to set scoped conversation ID.

```python
with moda.set_conversation_id("conv-123"):
    # LLM calls here use "conv-123"
```

### `moda.set_user_id(id)`

Context manager to set scoped user ID.

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
