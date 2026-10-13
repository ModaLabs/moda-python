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

### User and Account Properties

Attach traits to the end users and accounts behind your conversations so you can segment analytics by plan, revenue, lifecycle, and so on. Calls are non-blocking: arguments are validated immediately (invalid arguments raise `ValueError` / `TypeError`), then payloads are queued and sent in batches by a background thread. Delivery failures are logged, never raised. `moda.flush()` waits for queued payloads, and the queue is also drained at interpreter exit.

```python
import moda

moda.init("YOUR_MODA_API_KEY")

# Traits for an end user (None unsets a trait; `unset` removes keys)
moda.identify("u_123", {"plan": "pro", "mrr": 49, "signup_at": "2026-01-02"}, account_id="acct_9")
moda.identify("u_123", unset=["legacy_flag"])

# Traits for an account, optionally recording a user's membership
moda.group("acct_9", {"name": "Acme", "mrr": 1200, "currency": "USD"}, user_id="u_123")
moda.group("acct_9", user_id="u_456", membership="removed")

# Subscription lifecycle events (event_id is your idempotency key, e.g. the billing provider's event id)
moda.track_subscription(
    "upgraded",
    event_id="evt_1",
    account_id="acct_9",  # or user_id=...; exactly one is required
    plan="enterprise",
    mrr=1200,
    currency="USD",
)

moda.flush()
```

Rules (checked client-side):
- Ids are non-empty strings up to 256 characters.
- Trait values must be flat scalars: string (up to 1024 chars), finite number, boolean, or `None`. Nested dicts/lists are rejected. At most 100 keys.
- Trait keys match `^[A-Za-z_][A-Za-z0-9_.-]{0,127}$`; keys starting with `$` are reserved.
- `timestamp` / `occurred_at` accept a `datetime` or ISO-8601 string (naive values are treated as UTC) and default to the call time.
- Subscription `type` is one of `trial_started`, `trial_converted`, `started`, `renewed`, `upgraded`, `downgraded`, `cancelled`, `reactivated`, `churned`.
- Calls made before `moda.init()` (or without an API key) are ignored.

**PII:** send opaque internal ids rather than emails or names as `user_id` / `account_id`. Trait keys that look like secrets (tokens, passwords, API keys) and values shaped like card numbers or SSNs are dropped server-side. Traits such as email, name, and phone are stored and treated as PII.

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
| `MODA_BASE_URL` | Custom endpoint (default: `https://moda-ingest.modas.workers.dev/v1/traces`) |

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

Force flush all pending spans and queued `identify` / `group` / `track_subscription` payloads.

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
