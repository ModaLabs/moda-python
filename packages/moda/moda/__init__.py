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

import os
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
try:
    from moda.openclaw import (
        get_openclaw_env,
        get_openclaw_otel_config,
        run_openclaw_cli,
        trace_openclaw_operation,
    )
except ImportError:
    def _openclaw_unavailable(*_args, **_kwargs):
        raise ImportError(
            "OpenClaw helpers are unavailable. Install/update the moda package "
            "that includes `moda.openclaw`."
        )

    get_openclaw_env = _openclaw_unavailable
    get_openclaw_otel_config = _openclaw_unavailable
    run_openclaw_cli = _openclaw_unavailable
    trace_openclaw_operation = _openclaw_unavailable

# Vapi integration
from moda.vapi import (
    process_vapi_end_of_call_report,
    is_end_of_call_report,
)

# Module-level instance for convenience
_moda_instance: Moda | None = None

_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}
_FALSE_ENV_VALUES = {"0", "false", "no", "off", ""}


def _resolve_debug_enabled(explicit_debug: bool | None) -> tuple[bool, str]:
    """Resolve debug mode from explicit argument first, then MODA_DEBUG env var."""
    if explicit_debug is not None:
        return explicit_debug, "argument"

    raw_env = os.environ.get("MODA_DEBUG")
    if raw_env is not None:
        normalized = raw_env.strip().lower()
        if normalized in _TRUE_ENV_VALUES:
            return True, "MODA_DEBUG"
        if normalized in _FALSE_ENV_VALUES:
            return False, "MODA_DEBUG"

    return False, "default"


def _moda_config_path() -> str:
    """Absolute path to the CLI-persisted config, honoring MODA_CONFIG_HOME.

    Mirrors the moda CLI, which writes the provisioned API key to
    ``~/.moda/config.json`` (mode 0600) during ``moda init``.
    """
    base = os.environ.get("MODA_CONFIG_HOME") or os.path.join(
        os.path.expanduser("~"), ".moda"
    )
    return os.path.join(base, "config.json")


def _api_key_from_config_file() -> str | None:
    """Read ``api_key`` from ``~/.moda/config.json`` if present.

    Best-effort: a missing/unreadable/malformed file yields ``None`` so the
    caller controls the loud-fail message.
    """
    import json

    try:
        with open(_moda_config_path(), encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    key = data.get("api_key") if isinstance(data, dict) else None
    return key if isinstance(key, str) and key else None


def _resolve_api_key(explicit: str | None) -> str | None:
    """Resolve the Moda API key with CLI-matching precedence.

    explicit argument -> MODA_API_KEY env -> ~/.moda/config.json.
    Returns ``None`` when no source yields a non-empty string.
    """
    if isinstance(explicit, str) and explicit:
        return explicit
    env = os.environ.get("MODA_API_KEY")
    if env:
        return env
    return _api_key_from_config_file()


def init(
    api_key: str | None = None,
    app_name: str | None = None,
    endpoint: str | None = None,
    exporter=None,
    debug: bool | None = None,
    **kwargs,
):
    """Initialize Moda SDK.

    Args:
        api_key: Your Moda API key. When omitted, it is resolved from the
            MODA_API_KEY env var and then from ``~/.moda/config.json`` (written
            by ``moda init``), matching the CLI's precedence. This lets an app
            authenticate even when MODA_API_KEY was never exported.
        app_name: Optional name for your application.
        endpoint: Custom ingest endpoint. Defaults to Moda's ingest endpoint.
        exporter: Custom OpenTelemetry exporter (for testing/debugging).
        debug: Enable debug mode. Precedence: explicit argument > MODA_DEBUG env var > False.
        **kwargs: Additional arguments passed to Moda.init()

    Raises:
        ValueError: when no API key can be resolved from any source and no
            custom exporter was supplied (loud-fail contract).
    """
    import logging

    # Resolve the key with CLI-matching precedence before anything else so the
    # debug output and the loud-fail message both reflect the real source.
    api_key = _resolve_api_key(api_key)
    if not api_key and exporter is None:
        raise ValueError(
            "[Moda] API key is required, but none was found. Tried, in order: "
            "the api_key argument, the MODA_API_KEY environment variable, and "
            f"{_moda_config_path()} (written by `moda init`). "
            "Run `moda init`, or export MODA_API_KEY, or pass api_key to moda.init()."
        )

    debug_enabled, debug_source = _resolve_debug_enabled(debug)

    if debug_enabled:
        # Enable verbose logging for debugging
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger("opentelemetry").setLevel(logging.DEBUG)
        logging.getLogger("opentelemetry.exporter").setLevel(logging.DEBUG)
        logging.getLogger("urllib3").setLevel(logging.DEBUG)

        # Disable batching so spans are sent immediately
        kwargs["disable_batch"] = True

        print(f"[Moda Debug] Enabled via {debug_source}")
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

    if debug_enabled:
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
    "get_openclaw_otel_config",
    "get_openclaw_env",
    "trace_openclaw_operation",
    "run_openclaw_cli",
    "Instruments",
    "Moda",
    # Vapi integration
    "process_vapi_end_of_call_report",
    "is_end_of_call_report",
]
