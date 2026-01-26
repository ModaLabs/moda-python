"""Moda SDK - LLM Observability with Automatic Conversation Threading.

Usage:
    import moda

    moda.init("YOUR_MODA_API_KEY")

    # Set conversation ID for your session (recommended)
    moda.conversation_id = "session_" + session_id

    # Your LLM calls are now automatically tracked
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(...)

    moda.flush()
"""

import sys
from types import ModuleType
from typing import Optional

from traceloop.sdk import Instruments, Moda
from traceloop.sdk.context import (
    set_conversation_id,
    set_user_id,
    set_conversation_id_value,
    set_user_id_value,
)
from traceloop.sdk.conversation import (
    compute_conversation_id,
    get_conversation_id,
    get_user_id,
)

# Module-level instance for convenience
_moda_instance: Moda | None = None


def init(
    api_key: str | None = None,
    app_name: str | None = None,
    endpoint: str | None = None,
    exporter=None,
    **kwargs,
):
    """Initialize Moda SDK.

    Args:
        api_key: Your Moda API key. Can also be set via MODA_API_KEY env var.
        app_name: Optional name for your application.
        endpoint: Custom ingest endpoint. Defaults to Moda's ingest endpoint.
        exporter: Custom OpenTelemetry exporter (for testing/debugging).
        **kwargs: Additional arguments passed to Moda.init()
    """
    global _moda_instance
    _moda_instance = Moda()
    _moda_instance.init(
        api_key=api_key,
        app_name=app_name,
        api_endpoint=endpoint,
        exporter=exporter,
        **kwargs,
    )


def flush():
    """Flush all pending telemetry data."""
    if _moda_instance:
        _moda_instance.flush()


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
_original_module = sys.modules[__name__]
_wrapped_module = _ModaModule(__name__)
_wrapped_module.__dict__.update(_original_module.__dict__)
sys.modules[__name__] = _wrapped_module


__all__ = [
    "init",
    "flush",
    "conversation_id",
    "user_id",
    "set_conversation_id",
    "set_user_id",
    "set_conversation_id_value",
    "set_user_id_value",
    "get_conversation_id",
    "get_user_id",
    "compute_conversation_id",
    "Instruments",
    "Moda",
]
