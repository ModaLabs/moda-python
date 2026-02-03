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

# Vapi integration
from moda.vapi import (
    process_vapi_end_of_call_report,
    is_end_of_call_report,
)

# Module-level instance for convenience
_moda_instance: Moda | None = None


def init(
    api_key: str | None = None,
    app_name: str | None = None,
    endpoint: str | None = None,
    exporter=None,
    debug: bool = False,
    **kwargs,
):
    """Initialize Moda SDK.

    Args:
        api_key: Your Moda API key. Can also be set via MODA_API_KEY env var.
        app_name: Optional name for your application.
        endpoint: Custom ingest endpoint. Defaults to Moda's ingest endpoint.
        exporter: Custom OpenTelemetry exporter (for testing/debugging).
        debug: Enable debug mode - disables batching, enables verbose logging.
        **kwargs: Additional arguments passed to Moda.init()
    """
    import logging

    if debug:
        # Enable verbose logging for debugging
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger("opentelemetry").setLevel(logging.DEBUG)
        logging.getLogger("opentelemetry.exporter").setLevel(logging.DEBUG)
        logging.getLogger("urllib3").setLevel(logging.DEBUG)

        # Disable batching so spans are sent immediately
        kwargs["disable_batch"] = True

        print(f"[Moda Debug] Initializing with endpoint: {endpoint or 'default'}")
        print(f"[Moda Debug] API key: {api_key[:10]}..." if api_key else "[Moda Debug] No API key provided")

    global _moda_instance
    _moda_instance = Moda()

    # Only pass api_endpoint if explicitly provided (don't override default with None)
    init_kwargs = {
        "api_key": api_key,
        "exporter": exporter,
        **kwargs,
    }
    if app_name is not None:
        init_kwargs["app_name"] = app_name
    if endpoint is not None:
        init_kwargs["api_endpoint"] = endpoint

    _moda_instance.init(**init_kwargs)

    if debug:
        print("[Moda Debug] Initialization complete")


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
    # Vapi integration
    "process_vapi_end_of_call_report",
    "is_end_of_call_report",
]
