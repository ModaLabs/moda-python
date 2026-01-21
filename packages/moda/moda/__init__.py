"""Moda SDK - LLM Observability with Automatic Conversation Threading.

Usage:
    import moda
    moda.init("YOUR_MODA_API_KEY")

    # Your LLM calls are now automatically tracked
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(...)
"""

from traceloop.sdk import Moda
from traceloop.sdk.context import set_conversation_id, set_user_id
from traceloop.sdk.conversation import compute_conversation_id, get_user_id

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


__all__ = [
    "init",
    "flush",
    "set_conversation_id",
    "set_user_id",
    "compute_conversation_id",
    "get_user_id",
    "Moda",
]
