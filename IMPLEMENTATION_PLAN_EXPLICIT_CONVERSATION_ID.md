# Update moda-python SDK: Explicit Conversation ID as Primary Pattern

## Summary of Node SDK Changes (Already Completed)

We updated the Node.js SDK (`moda-node`) to make **explicit conversation IDs** the recommended approach, while keeping auto-compute as a fallback. Here's what was changed:

### Node SDK Changes Made

1. **Added property-style API** (`src/index.ts`):
   ```typescript
   // New property-style API
   Moda.conversationId = 'session_123';   // set
   Moda.conversationId;                    // get (returns string | null)
   Moda.conversationId = null;             // clear

   Moda.userId = 'user_456';               // set
   Moda.userId;                            // get
   Moda.userId = null;                     // clear
   ```

2. **Added `getGlobalContext()` function** (`src/context.ts`):
   - Exposes the global context for the getter properties

3. **Updated documentation** (`README.md`, `moda-sdk-node.mdx`):
   - Quick Start now shows explicit `conversationId` setting
   - "Conversation Tracking" section leads with explicit IDs (recommended)
   - "Automatic Fallback" section explains auto-compute as secondary option

---

## Python SDK Current State

The Python SDK (`moda-python`) currently has:

### Existing API (in `traceloop/sdk/__init__.py`, `context.py`, `conversation.py`)

```python
# Context managers (scoped)
with moda.set_conversation_id("conv_123"):
    client.chat.completions.create(...)

with moda.set_user_id("user_456"):
    client.chat.completions.create(...)

# Value setters (global)
moda.set_conversation_id_value("conv_123")  # set
moda.set_conversation_id_value(None)        # clear

moda.set_user_id_value("user_456")          # set
moda.set_user_id_value(None)                # clear

# Getters
moda.get_conversation_id()  # returns Optional[str]
moda.get_user_id()          # returns Optional[str]
```

### What's Missing

The Python SDK lacks:
1. A simpler property-style API like `moda.conversation_id = "value"`
2. Documentation emphasis on explicit IDs as the primary pattern

---

## Changes Required for Python SDK

### 1. Add Module-Level Properties (`packages/moda/traceloop/sdk/__init__.py`)

Python modules don't natively support properties, but we can achieve this using a module wrapper class. Add this pattern:

```python
import sys
from types import ModuleType

# ... existing code ...

class _ModaModule(ModuleType):
    """Module wrapper that adds property-style access to conversation/user IDs."""

    @property
    def conversation_id(self) -> Optional[str]:
        """Get the current conversation ID.

        Returns:
            The conversation ID if set, None otherwise.
        """
        return get_conversation_id()

    @conversation_id.setter
    def conversation_id(self, value: Optional[str]) -> None:
        """Set the conversation ID.

        Args:
            value: The conversation ID to set, or None to clear.

        Example:
            import moda
            moda.conversation_id = 'session_123'
            client.chat.completions.create(...)
            moda.conversation_id = None  # clear
        """
        set_conversation_id_value(value)

    @property
    def user_id(self) -> Optional[str]:
        """Get the current user ID.

        Returns:
            The user ID if set, None otherwise.
        """
        return get_user_id()

    @user_id.setter
    def user_id(self, value: Optional[str]) -> None:
        """Set the user ID.

        Args:
            value: The user ID to set, or None to clear.

        Example:
            import moda
            moda.user_id = 'user_456'
            client.chat.completions.create(...)
            moda.user_id = None  # clear
        """
        set_user_id_value(value)


# At the end of the module, replace the module with our wrapper
_original_module = sys.modules[__name__]
_new_module = _ModaModule(__name__)
_new_module.__dict__.update(_original_module.__dict__)
sys.modules[__name__] = _new_module
```

### 2. Alternative: Add Properties to Moda Class

If the module wrapper approach is too complex, add the properties to the existing `Moda` class:

```python
class Moda:
    """Moda SDK for LLM observability with automatic conversation threading."""

    # ... existing code ...

    # Property-style API for conversation ID
    @staticmethod
    def _get_conversation_id() -> Optional[str]:
        return get_conversation_id()

    @staticmethod
    def _set_conversation_id_prop(value: Optional[str]) -> None:
        set_conversation_id_value(value)

    # Note: Python doesn't support properties on class objects directly,
    # so we use a descriptor or classproperties pattern
```

**Recommended approach**: Use the module wrapper since it provides the cleanest API (`moda.conversation_id = 'x'`).

### 3. Update Documentation (`packages/moda/README.md`)

Restructure to emphasize explicit IDs:

```markdown
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

moda.flush()
```

## Conversation Tracking

### Setting Conversation ID (Recommended)

For production use, explicitly set a conversation ID:

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

```python
moda.user_id = "user_12345"
client.chat.completions.create(...)
moda.user_id = None  # clear
```

## Automatic Fallback

If you don't set a conversation ID, the SDK automatically computes one from
the first user message and system prompt. This works for simple use cases
but explicit IDs are recommended for production.
```

---

## Files to Modify

| File | Change |
|------|--------|
| `packages/moda/traceloop/sdk/__init__.py` | Add module wrapper with `conversation_id` and `user_id` properties |
| `packages/moda/README.md` | Restructure to emphasize explicit IDs |
| `packages/moda/tests/test_conversation.py` | Add tests for property-style API |

---

## Implementation Details

### Step 1: Update `packages/moda/traceloop/sdk/__init__.py`

At the end of the file, add the module wrapper:

```python
# ============================================================
# Module property wrapper for cleaner API
# ============================================================

class _ModaModule(ModuleType):
    """Module wrapper that adds property-style access to conversation/user IDs.

    This allows the cleaner API:
        moda.conversation_id = 'session_123'
        moda.user_id = 'user_456'

    Instead of:
        moda.set_conversation_id_value('session_123')
        moda.set_user_id_value('user_456')
    """

    @property
    def conversation_id(self) -> Optional[str]:
        """Get or set the current conversation ID.

        Example:
            moda.conversation_id = 'session_123'
            print(moda.conversation_id)  # 'session_123'
            moda.conversation_id = None  # clear
        """
        return get_conversation_id()

    @conversation_id.setter
    def conversation_id(self, value: Optional[str]) -> None:
        set_conversation_id_value(value)

    @property
    def user_id(self) -> Optional[str]:
        """Get or set the current user ID.

        Example:
            moda.user_id = 'user_456'
            print(moda.user_id)  # 'user_456'
            moda.user_id = None  # clear
        """
        return get_user_id()

    @user_id.setter
    def user_id(self, value: Optional[str]) -> None:
        set_user_id_value(value)


# Replace this module with our property-enabled wrapper
# This is a standard Python pattern for adding properties to modules
import sys
from types import ModuleType

_original_module = sys.modules[__name__]
_wrapped_module = _ModaModule(__name__)
_wrapped_module.__dict__.update(_original_module.__dict__)
sys.modules[__name__] = _wrapped_module
```

### Step 2: Add Tests

Add to `packages/moda/tests/test_conversation.py`:

```python
def test_conversation_id_property():
    """Test property-style conversation ID access."""
    import moda

    # Initially None
    assert moda.conversation_id is None

    # Set via property
    moda.conversation_id = "test_conv_123"
    assert moda.conversation_id == "test_conv_123"

    # Clear via None
    moda.conversation_id = None
    assert moda.conversation_id is None


def test_user_id_property():
    """Test property-style user ID access."""
    import moda

    # Initially None
    assert moda.user_id is None

    # Set via property
    moda.user_id = "user_456"
    assert moda.user_id == "user_456"

    # Clear via None
    moda.user_id = None
    assert moda.user_id is None


def test_property_and_context_manager_consistency():
    """Test that property and context manager APIs are consistent."""
    import moda

    # Set via property
    moda.conversation_id = "prop_conv"

    # Context manager should see it
    assert moda.get_conversation_id() == "prop_conv"

    # Context manager overrides temporarily
    with moda.set_conversation_id("scoped_conv"):
        assert moda.conversation_id == "scoped_conv"

    # Back to property value
    assert moda.conversation_id == "prop_conv"

    # Clean up
    moda.conversation_id = None
```

---

## Verification

1. **Tests pass**: `cd packages/moda && uv run pytest tests/ -v`
2. **Property API works**:
   ```python
   import moda
   moda.conversation_id = 'test_123'
   assert moda.conversation_id == 'test_123'
   moda.conversation_id = None
   assert moda.conversation_id is None
   ```
3. **Backward compatibility**: Existing code using `set_conversation_id_value()` still works

---

## API Summary (After Changes)

### New Primary API (Recommended)

```python
import moda

moda.init("moda_xxx")

# Property-style (recommended for production)
moda.conversation_id = "session_123"
moda.user_id = "user_456"

client.chat.completions.create(...)

moda.conversation_id = None  # clear
moda.user_id = None
```

### Secondary APIs (Still Supported)

```python
# Context managers (scoped)
with moda.set_conversation_id("scoped_123"):
    client.chat.completions.create(...)

# Value setters (global)
moda.set_conversation_id_value("conv_123")
moda.set_user_id_value("user_456")

# Getters
moda.get_conversation_id()
moda.get_user_id()
```

### Auto-Compute Fallback

If no conversation ID is set, the SDK automatically computes one from the first user message and system prompt.
