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

## User and Account Properties

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

## OpenClaw Integration

OpenClaw can export OpenTelemetry data via its `diagnostics-otel` plugin.
Moda provides helpers to generate OpenClaw config/env and trace OpenClaw runtime operations.

```python
import moda
from moda.openclaw import (
    get_openclaw_otel_config,
    get_openclaw_env,
    run_openclaw_cli,
)

moda.init("YOUR_MODA_API_KEY")

config = get_openclaw_otel_config(service_name="openclaw-gateway")
env = get_openclaw_env()

# Optional traced CLI invocation
result = run_openclaw_cli(["--help"], check=False)

# If OpenClaw is not installed globally, run via npx:
result = run_openclaw_cli(
    ["agent", "--help"],
    command_prefix=["npx", "-y", "openclaw@latest"],
    check=False,
)
print(result.returncode)
```

## Automatic Fallback

If you don't set a conversation ID, the SDK automatically computes one from the first user message and system prompt. This works for simple use cases but explicit IDs are recommended for production.

## Full Documentation

See the [main repository README](https://github.com/ModaLabs/moda-python) for complete documentation.
