# moda-ai

Moda's Python SDK for LLM observability with automatic conversation threading. Built on OpenTelemetry.

## Installation

```bash
pip install moda-ai
```

## Quick Start

```python
import moda
from openai import OpenAI

moda.init("YOUR_MODA_API_KEY")

# Set conversation ID for your session (recommended)
moda.conversation_id = "session_" + session_id

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)

moda.flush()
```

## Conversation Tracking

### Setting Conversation ID (Recommended)

For production use, explicitly set a conversation ID to group related LLM calls:

```python
# Property-style (recommended)
moda.conversation_id = "support_ticket_123"
client.chat.completions.create(...)
moda.conversation_id = None  # clear when done

# Or use the setter function
moda.set_conversation_id_value("support_ticket_123")
moda.set_conversation_id_value(None)  # clear

# Or use context manager (scoped)
with moda.set_conversation_id("support_ticket_123"):
    client.chat.completions.create(...)
```

### Setting User ID

Associate LLM calls with specific users:

```python
moda.user_id = "user_12345"
client.chat.completions.create(...)
moda.user_id = None  # clear
```

## Automatic Fallback

If you don't set a conversation ID, the SDK automatically computes one from the first user message and system prompt. This works for simple use cases but explicit IDs are recommended for production.

## Full Documentation

See the [main repository README](https://github.com/ModaLabs/moda-python) for complete documentation.
