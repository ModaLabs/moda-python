"""OpenTelemetry Claude Agent SDK instrumentation.

Instruments the Claude Agent SDK (claude-agent-sdk) to capture agent execution
spans. Since the Claude Agent SDK spawns Claude Code as a subprocess, standard
Anthropic API instrumentation doesn't capture any calls. This instrumentor wraps
the SDK's own methods to extract conversation data from the messages it yields.
"""

import logging
from contextvars import ContextVar
from typing import Collection

from opentelemetry.instrumentation.claude_agent_sdk.streaming import (
    WrappedAgentStream,
    _set_span_attribute,
)
from opentelemetry.instrumentation.claude_agent_sdk.version import __version__
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.trace import SpanKind, get_tracer
from wrapt import wrap_function_wrapper

logger = logging.getLogger(__name__)

_instruments = ("claude-agent-sdk",)

# Async-safe storage for the prompt between query() and receive_response()
_current_prompt: ContextVar[str] = ContextVar("_moda_current_prompt", default="")

WRAPPED_METHODS = [
    {
        "package": "claude_agent_sdk",
        "object": "ClaudeSDKClient",
        "method": "query",
    },
    {
        "package": "claude_agent_sdk",
        "object": "ClaudeSDKClient",
        "method": "receive_response",
    },
]


class ClaudeAgentSDKInstrumentor(BaseInstrumentor):
    """OpenTelemetry instrumentor for the Claude Agent SDK."""

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        # Wrap query() — async method that sends the prompt
        try:
            wrap_function_wrapper(
                "claude_agent_sdk",
                "ClaudeSDKClient.query",
                _wrap_query(tracer),
            )
        except Exception as e:
            logger.debug(f"Failed to wrap ClaudeSDKClient.query: {e}")

        # Wrap receive_response() — async generator method
        try:
            wrap_function_wrapper(
                "claude_agent_sdk",
                "ClaudeSDKClient.receive_response",
                _wrap_receive_response(tracer),
            )
        except Exception as e:
            logger.debug(f"Failed to wrap ClaudeSDKClient.receive_response: {e}")

    def _uninstrument(self, **kwargs):
        import claude_agent_sdk

        for method_info in WRAPPED_METHODS:
            try:
                obj = getattr(claude_agent_sdk, method_info["object"])
                unwrap(obj, method_info["method"])
            except Exception as e:
                logger.debug(f"Failed to unwrap {method_info['object']}.{method_info['method']}: {e}")


def _wrap_query(tracer):
    """Wrap ClaudeSDKClient.query() to capture the prompt on the instance."""

    async def wrapper(wrapped, instance, args, kwargs):
        # Store the prompt on the instance so receive_response wrapper can access it
        prompt = args[0] if args else kwargs.get("prompt", "")
        _current_prompt.set(str(prompt))
        return await wrapped(*args, **kwargs)

    return wrapper


def _wrap_receive_response(tracer):
    """Wrap ClaudeSDKClient.receive_response() to create a span and wrap the stream.

    receive_response() is an async generator function. Our wrapper must also be
    an async generator — we yield each message through a WrappedAgentStream that
    accumulates token usage and finalizes the span when the stream is exhausted.
    """

    async def wrapper(wrapped, instance, args, kwargs):
        # Get the model from instance options (defensive access)
        model = _get_model(instance)

        # Get conversation/user context from Moda context vars
        conversation_id = None
        user_id = None
        try:
            from traceloop.sdk.conversation import get_conversation_id, get_user_id

            conversation_id = get_conversation_id()
            user_id = get_user_id()
        except ImportError:
            pass

        # Create span for this agent run
        span = tracer.start_span(
            "claude_agent.chat",
            kind=SpanKind.CLIENT,
            attributes={
                "gen_ai.system": "Anthropic",
                "llm.request_type": "chat",
            },
        )

        # Set known attributes upfront
        if model:
            _set_span_attribute(span, "gen_ai.request.model", model)
        if conversation_id:
            _set_span_attribute(span, "moda.conversation_id", conversation_id)
        if user_id:
            _set_span_attribute(span, "moda.user_id", user_id)

        # Capture the prompt if stored by query() wrapper
        prompt = _current_prompt.get("")
        if prompt:
            prompt_text = str(prompt)
            _set_span_attribute(span, "gen_ai.prompt", prompt_text[:1000])
            # Emit OpenLLMetry-style prompt fields so Moda ingest parser can extract conversation rows.
            _set_span_attribute(span, "llm.prompts.0.role", "user")
            _set_span_attribute(span, "llm.prompts.0.content", prompt_text[:4000])

        # Call original async generator and wrap it
        async_gen = wrapped(*args, **kwargs)
        stream = WrappedAgentStream(async_gen, instance, span)

        # Yield through — try/finally ensures span is finalized on early exit (break, cancel)
        try:
            async for msg in stream:
                yield msg
        finally:
            stream._finalize()

    return wrapper


def _get_model(instance) -> str:
    """Defensively extract the model name from a ClaudeSDKClient instance."""
    # Try instance.options.model
    options = getattr(instance, "options", None)
    if options:
        model = getattr(options, "model", None)
        if model:
            return str(model)

    # Try instance.model
    model = getattr(instance, "model", None)
    if model:
        return str(model)

    # Try instance._model
    model = getattr(instance, "_model", None)
    if model:
        return str(model)

    return ""
