# Moda Python SDK Implementation Plan

## Overview

Build the official Moda Python SDK by **forking OpenLLMetry** and adding conversation threading - the key feature that OpenLLMetry lacks.

**Target User Experience:**
```python
import moda
moda.init("moda_xxx")

# All OpenAI/Anthropic calls automatically tracked with conversation threading
```

---

## Phase 1: Fork & Setup Repository

### 1.1 Clone OpenLLMetry
```bash
cd /Users/pranavbedi/Moda-Full/moda-python
git remote add upstream https://github.com/traceloop/openllmetry.git
git fetch upstream
git checkout -b main upstream/main
```

### 1.2 Understand OpenLLMetry Structure
OpenLLMetry uses a monorepo with multiple packages:
```
openllmetry/
├── packages/
│   ├── traceloop-sdk/           # Core SDK → becomes "moda"
│   ├── opentelemetry-instrumentation-openai/    # → moda-openai
│   ├── opentelemetry-instrumentation-anthropic/ # → moda-anthropic
│   ├── opentelemetry-instrumentation-cohere/
│   ├── opentelemetry-instrumentation-bedrock/
│   └── ...
├── pyproject.toml
└── README.md
```

### 1.3 Initial Cleanup
- Remove packages we don't need initially (keep OpenAI + Anthropic)
- Update all branding references
- Update LICENSE and attribution

---

## Phase 2: Core SDK Development

### 2.1 Create Conversation ID Module
**New file: `packages/moda/src/moda/conversation.py`**

```python
import hashlib
import json
import uuid
from contextvars import ContextVar

# Context variables for explicit overrides
_conversation_id_var: ContextVar[str | None] = ContextVar('conversation_id', default=None)
_user_id_var: ContextVar[str | None] = ContextVar('user_id', default=None)

def compute_conversation_id(messages: list[dict], explicit_id: str | None = None) -> str:
    """
    Compute stable conversation_id from message history.

    Algorithm:
    1. If explicit_id provided (via context), use it
    2. Otherwise, hash first user message + system prompt
    3. Fallback to random UUID if no user message
    """
    # Check context variable first
    ctx_id = _conversation_id_var.get()
    if ctx_id:
        return ctx_id

    if explicit_id:
        return explicit_id

    # Find first user message
    first_user = next((m for m in messages if m.get("role") == "user"), None)
    if not first_user:
        return f"conv_{uuid.uuid4().hex[:16]}"

    # Include system prompt if present
    system = next((m for m in messages if m.get("role") == "system"), None)

    seed = json.dumps({
        "system": system.get("content") if system else None,
        "first_user": first_user.get("content")
    }, sort_keys=True)

    return f"conv_{hashlib.sha256(seed.encode()).hexdigest()[:16]}"

def get_user_id() -> str | None:
    """Get current user_id from context."""
    return _user_id_var.get()
```

### 2.2 Create Context Managers
**Add to `packages/moda/src/moda/context.py`**

```python
from contextlib import contextmanager
from .conversation import _conversation_id_var, _user_id_var

@contextmanager
def set_conversation_id(conversation_id: str):
    """Context manager to explicitly set conversation_id."""
    token = _conversation_id_var.set(conversation_id)
    try:
        yield
    finally:
        _conversation_id_var.reset(token)

@contextmanager
def set_user_id(user_id: str):
    """Context manager to set user_id for attribution."""
    token = _user_id_var.set(user_id)
    try:
        yield
    finally:
        _user_id_var.reset(token)
```

### 2.3 Update Main SDK Entry Point
**Modify `packages/moda/src/moda/__init__.py`**

```python
from typing import Literal

DEFAULT_ENDPOINT = "https://ingest.moda.so/v1/traces"

_initialized = False

def init(
    api_key: str,
    *,
    endpoint: str = DEFAULT_ENDPOINT,
    environment: Literal["development", "staging", "production"] = "production",
    enabled: bool = True,
    disable_batch: bool = False,
) -> None:
    """
    Initialize Moda SDK. Auto-patches OpenAI and Anthropic clients.

    Args:
        api_key: Your Moda API key (starts with "moda_")
        endpoint: Moda ingest endpoint (default: https://ingest.moda.so/v1/traces)
        environment: Environment tag for filtering
        enabled: Set to False to disable tracking
        disable_batch: Disable batching (for debugging)

    Usage:
        import moda
        moda.init("moda_xxx")
    """
    global _initialized

    if not enabled:
        return

    if _initialized:
        return

    # Initialize OpenTelemetry with Moda endpoint
    # (Adapt from traceloop-sdk's initialization)
    ...

    _initialized = True

def flush() -> None:
    """Force flush pending spans. Call before process exit."""
    ...

def shutdown() -> None:
    """Graceful shutdown. Flushes and stops background threads."""
    ...

# Re-export context managers
from .context import set_conversation_id, set_user_id

__all__ = [
    "init",
    "flush",
    "shutdown",
    "set_conversation_id",
    "set_user_id",
]
```

---

## Phase 3: Modify Instrumentors

### 3.1 OpenAI Instrumentor Changes
**Modify `packages/moda-openai/src/moda_openai/instrumentor.py`**

Key changes:
1. Import conversation module
2. In `_set_span_attributes()`, extract messages and compute conversation_id
3. Set `moda.conversation_id` attribute on span

```python
from moda.conversation import compute_conversation_id, get_user_id

def _set_span_attributes(span, kwargs, response):
    # ... existing attribute setting ...

    # NEW: Compute and set conversation_id
    messages = kwargs.get("messages", [])
    conversation_id = compute_conversation_id(messages)
    span.set_attribute("moda.conversation_id", conversation_id)

    # NEW: Set user_id if available
    user_id = get_user_id()
    if user_id:
        span.set_attribute("moda.user_id", user_id)
```

### 3.2 Anthropic Instrumentor Changes
**Modify `packages/moda-anthropic/src/moda_anthropic/instrumentor.py`**

Same pattern - add conversation_id computation:

```python
from moda.conversation import compute_conversation_id, get_user_id

def _set_span_attributes(span, kwargs, response):
    # Anthropic uses different message format
    messages = kwargs.get("messages", [])
    system = kwargs.get("system")  # Anthropic has separate system param

    # Normalize to common format for hashing
    normalized = []
    if system:
        normalized.append({"role": "system", "content": system})
    normalized.extend(messages)

    conversation_id = compute_conversation_id(normalized)
    span.set_attribute("moda.conversation_id", conversation_id)

    user_id = get_user_id()
    if user_id:
        span.set_attribute("moda.user_id", user_id)
```

---

## Phase 4: Package Configuration

### 4.1 Rename Packages
```
traceloop-sdk                              → moda
opentelemetry-instrumentation-openai       → moda-openai
opentelemetry-instrumentation-anthropic    → moda-anthropic
opentelemetry-instrumentation-cohere       → moda-cohere (optional)
opentelemetry-instrumentation-bedrock      → moda-bedrock (optional)
```

### 4.2 Update pyproject.toml Files

**`packages/moda/pyproject.toml`**
```toml
[project]
name = "moda"
version = "0.1.0"
description = "Moda SDK for LLM observability with automatic conversation threading"
readme = "README.md"
license = {text = "Apache-2.0"}
requires-python = ">=3.9"
dependencies = [
    "opentelemetry-api>=1.20.0",
    "opentelemetry-sdk>=1.20.0",
    "opentelemetry-exporter-otlp-proto-http>=1.20.0",
]

[project.optional-dependencies]
openai = ["moda-openai"]
anthropic = ["moda-anthropic"]
all = ["moda-openai", "moda-anthropic"]

[project.urls]
Homepage = "https://moda.so"
Documentation = "https://docs.moda.so"
Repository = "https://github.com/ModaLabs/moda-python"
```

**`packages/moda-openai/pyproject.toml`**
```toml
[project]
name = "moda-openai"
version = "0.1.0"
description = "Moda OpenAI instrumentation"
requires-python = ">=3.9"
dependencies = [
    "moda>=0.1.0",
    "openai>=1.0.0",
    "opentelemetry-instrumentation>=0.40b0",
]
```

### 4.3 Create Root pyproject.toml (Monorepo)
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["packages/moda/src/moda"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"

[tool.ruff]
line-length = 100
target-version = "py39"
```

---

## Phase 5: Testing

### 5.1 Unit Tests
**`tests/test_conversation.py`**
```python
import pytest
from moda.conversation import compute_conversation_id

def test_same_conversation_same_id():
    """Same first message should produce same conversation_id."""
    messages1 = [{"role": "user", "content": "Hello"}]
    messages2 = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
        {"role": "user", "content": "How are you?"}
    ]

    id1 = compute_conversation_id(messages1)
    id2 = compute_conversation_id(messages2)

    assert id1 == id2

def test_different_conversation_different_id():
    """Different first message should produce different conversation_id."""
    messages1 = [{"role": "user", "content": "Hello"}]
    messages2 = [{"role": "user", "content": "Goodbye"}]

    id1 = compute_conversation_id(messages1)
    id2 = compute_conversation_id(messages2)

    assert id1 != id2

def test_system_prompt_affects_id():
    """System prompt should affect conversation_id."""
    messages1 = [{"role": "user", "content": "Hello"}]
    messages2 = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello"}
    ]

    id1 = compute_conversation_id(messages1)
    id2 = compute_conversation_id(messages2)

    assert id1 != id2

def test_explicit_id_override():
    """Explicit ID should override computed ID."""
    messages = [{"role": "user", "content": "Hello"}]

    id1 = compute_conversation_id(messages, explicit_id="custom_123")

    assert id1 == "custom_123"

def test_no_user_message_generates_uuid():
    """No user message should generate random UUID."""
    messages = [{"role": "system", "content": "You are helpful"}]

    id1 = compute_conversation_id(messages)
    id2 = compute_conversation_id(messages)

    assert id1.startswith("conv_")
    assert id1 != id2  # Each call generates new UUID
```

### 5.2 Integration Tests
**`tests/test_integration.py`**
```python
import pytest
from unittest.mock import MagicMock, patch

def test_openai_span_has_conversation_id():
    """OpenAI calls should include moda.conversation_id attribute."""
    ...

def test_anthropic_span_has_conversation_id():
    """Anthropic calls should include moda.conversation_id attribute."""
    ...

def test_context_manager_override():
    """set_conversation_id() context manager should override computed ID."""
    ...
```

### 5.3 Test Commands
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=moda --cov-report=html

# Run specific test file
pytest tests/test_conversation.py -v
```

---

## Phase 6: CI/CD Setup

### 6.1 GitHub Actions for Testing
**`.github/workflows/test.yml`**
```yaml
name: Test

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11', '3.12']

    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          pip install -e packages/moda[all]
          pip install -e packages/moda-openai
          pip install -e packages/moda-anthropic
          pip install pytest pytest-asyncio pytest-cov

      - name: Run tests
        run: pytest --cov=moda --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

### 6.2 GitHub Actions for Publishing
**`.github/workflows/publish.yml`**
```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install build tools
        run: pip install build twine

      - name: Build packages
        run: |
          cd packages/moda && python -m build
          cd ../moda-openai && python -m build
          cd ../moda-anthropic && python -m build

      - name: Publish to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
        run: |
          twine upload packages/moda/dist/*
          twine upload packages/moda-openai/dist/*
          twine upload packages/moda-anthropic/dist/*
```

---

## Phase 7: Documentation

### 7.1 README.md
```markdown
# Moda Python SDK

Official Python SDK for [Moda](https://moda.so) - LLM observability with automatic conversation threading.

## Installation

```bash
pip install moda
```

Or with specific provider support:
```bash
pip install moda[openai]      # OpenAI support
pip install moda[anthropic]   # Anthropic support
pip install moda[all]         # All providers
```

## Quick Start

```python
import moda
from openai import OpenAI

# Initialize (2 lines!)
moda.init("moda_xxx")
client = OpenAI()

# All calls automatically tracked with conversation threading
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Features

- **Automatic instrumentation** - No code changes to your LLM calls
- **Conversation threading** - Multi-turn conversations automatically grouped
- **All major providers** - OpenAI, Anthropic, and more
- **Zero config** - Works out of the box

## Advanced Usage

### Explicit Conversation ID

```python
with moda.set_conversation_id("my_session_123"):
    response = client.chat.completions.create(...)
```

### User Attribution

```python
with moda.set_user_id("user_456"):
    response = client.chat.completions.create(...)
```

### Graceful Shutdown

```python
# Before process exit
moda.flush()
moda.shutdown()
```
```

---

## Implementation Checklist

### Phase 1: Setup
- [ ] Fetch OpenLLMetry upstream into moda-python repo
- [ ] Analyze OpenLLMetry codebase structure
- [ ] Identify files to modify vs. keep as-is
- [ ] Set up development environment

### Phase 2: Core SDK
- [ ] Create `moda/conversation.py` with conversation_id computation
- [ ] Create `moda/context.py` with context managers
- [ ] Update main `moda/__init__.py` entry point
- [ ] Change default endpoint to `https://ingest.moda.so/v1/traces`

### Phase 3: Instrumentors
- [ ] Modify OpenAI instrumentor to add `moda.conversation_id`
- [ ] Modify Anthropic instrumentor to add `moda.conversation_id`
- [ ] Handle Anthropic's different message format (separate system param)
- [ ] Add `moda.user_id` attribute support

### Phase 4: Packaging
- [ ] Rename packages (traceloop-sdk → moda, etc.)
- [ ] Update all pyproject.toml files
- [ ] Update import paths throughout codebase
- [ ] Create distribution packages

### Phase 5: Testing
- [ ] Write unit tests for conversation_id computation
- [ ] Write integration tests for instrumentors
- [ ] Test with real OpenAI/Anthropic calls
- [ ] Test streaming responses
- [ ] Test async support

### Phase 6: CI/CD
- [ ] Set up GitHub Actions for testing
- [ ] Set up GitHub Actions for publishing
- [ ] Configure PyPI credentials
- [ ] Test publish to TestPyPI first

### Phase 7: Documentation
- [ ] Write comprehensive README
- [ ] Add inline code documentation
- [ ] Create examples directory
- [ ] Add CHANGELOG.md

---

## File Changes Summary

| Original (OpenLLMetry) | New (Moda) | Changes |
|------------------------|------------|---------|
| `packages/traceloop-sdk/` | `packages/moda/` | + conversation.py, + context.py, update init |
| `packages/opentelemetry-instrumentation-openai/` | `packages/moda-openai/` | + conversation_id attribute |
| `packages/opentelemetry-instrumentation-anthropic/` | `packages/moda-anthropic/` | + conversation_id attribute |

**New files (~150 lines total):**
- `moda/conversation.py` (~50 lines)
- `moda/context.py` (~25 lines)
- Updates to `__init__.py` (~25 lines)
- Instrumentor modifications (~25 lines each)

---

## Dependencies

### Runtime
- `opentelemetry-api>=1.20.0`
- `opentelemetry-sdk>=1.20.0`
- `opentelemetry-exporter-otlp-proto-http>=1.20.0`
- `openai>=1.0.0` (optional)
- `anthropic>=0.18.0` (optional)

### Development
- `pytest>=7.0.0`
- `pytest-asyncio>=0.21.0`
- `pytest-cov>=4.0.0`
- `ruff>=0.1.0`
- `mypy>=1.0.0`

---

## Timeline Estimate

1. **Phase 1-2** (Setup + Core): Fork repo, create conversation module
2. **Phase 3** (Instrumentors): Modify OpenAI/Anthropic instrumentors
3. **Phase 4** (Packaging): Rename and configure packages
4. **Phase 5** (Testing): Write and run tests
5. **Phase 6** (CI/CD): Set up automation
6. **Phase 7** (Docs): Write documentation

---

## Backend Changes Required (ModaObservability)

After SDK is complete, update `apps/moda-ingest-worker/src/otlp.ts`:

```typescript
// Read moda.conversation_id from span attributes
const conversationId =
  findAttribute(attributes, 'moda.conversation_id') ||  // NEW
  findAttribute(attributes, 'traceloop.association.properties.conversation_id') ||
  span.traceId ||
  crypto.randomUUID();
```

This change is backwards compatible - existing clients without the SDK continue working.
