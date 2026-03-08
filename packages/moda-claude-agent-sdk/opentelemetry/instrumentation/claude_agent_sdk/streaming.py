"""Wrapped async generator for Claude Agent SDK receive_response() streams."""

import logging

from opentelemetry.trace import Span
from opentelemetry.trace.status import Status, StatusCode

logger = logging.getLogger(__name__)


def _set_span_attribute(span: Span, name: str, value):
    """Set a span attribute only if value is not None."""
    if value is not None and value != "":
        try:
            span.set_attribute(name, value)
        except Exception:
            pass


def _get(obj, key, default=None):
    """Get a value from an object that may be a dict or an object with attributes."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class WrappedAgentStream:
    """Async iterator proxy that wraps the Claude Agent SDK's receive_response() async generator.

    Yields each message through while accumulating token usage, tool call counts,
    and agent metadata. Finalizes the span when the stream is exhausted.

    Token data comes from two possible sources:
      1. ResultMessage.usage dict (primary — always present at end of run)
      2. StreamEvent with message_start/message_delta (when include_partial_messages=True)
    """

    def __init__(self, async_gen, instance, span: Span):
        self._async_gen = async_gen
        self._instance = instance
        self._span = span
        self._finalized = False

        # Accumulated data
        self._input_tokens = 0
        self._output_tokens = 0
        self._tool_call_count = 0
        self._num_turns = None
        self._session_id = None
        self._model = None
        # Keep completion candidates with turn/source metadata so we can dedupe
        # only stream+assistant duplicates within the same turn.
        self._completion_entries = []
        self._stream_text_buffer = []
        self._turn_counter = 0
        self._current_turn_id = None

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            msg = await self._async_gen.__anext__()
        except StopAsyncIteration:
            self._finalize()
            raise
        except Exception as e:
            self._finalize(error=e)
            raise

        self._process_message(msg)
        return msg

    async def aclose(self):
        """Finalize span when consumer exits the loop early (break, cancel, etc.)."""
        self._finalize()
        if hasattr(self._async_gen, 'aclose'):
            await self._async_gen.aclose()

    def _process_message(self, msg):
        """Extract data from each message yielded by the agent stream."""
        try:
            msg_type = type(msg).__name__

            # Type-name matching with duck-typing fallbacks for robustness
            if msg_type == "ResultMessage" or (
                hasattr(msg, "num_turns") and hasattr(msg, "usage")
            ):
                self._handle_result_message(msg)
            elif msg_type == "AssistantMessage" or (
                hasattr(msg, "model") and hasattr(msg, "content")
            ):
                self._handle_assistant_message(msg)
            elif msg_type == "StreamEvent" or hasattr(msg, "event"):
                self._handle_stream_event(msg)
            # SystemMessage, UserMessage, etc. — ignored
        except Exception as e:
            logger.debug(f"Error processing agent message: {e}")

    def _next_turn_id(self):
        self._turn_counter += 1
        self._current_turn_id = self._turn_counter
        return self._current_turn_id

    def _get_or_create_turn_id(self):
        if self._current_turn_id is None:
            return self._next_turn_id()
        return self._current_turn_id

    def _add_completion_candidate(self, completion_text: str, source: str):
        if not isinstance(completion_text, str) or not completion_text.strip():
            return

        turn_id = self._get_or_create_turn_id()
        normalized = completion_text.strip()

        # Dedupe only within the same turn and only across different sources
        # (stream vs assistant). Identical completions across turns must survive.
        for existing_turn, existing_source, existing_text in self._completion_entries:
            if existing_turn != turn_id:
                continue
            if existing_source == source:
                continue
            if existing_text.strip() == normalized:
                return

        self._completion_entries.append((turn_id, source, completion_text))

    def _flush_stream_text_buffer(self):
        if self._stream_text_buffer:
            self._add_completion_candidate("".join(self._stream_text_buffer), source="stream")
            self._stream_text_buffer = []

    def _handle_result_message(self, msg):
        """Extract token usage and metadata from the final ResultMessage.

        ResultMessage.usage is a dict like:
            {'input_tokens': 10, 'output_tokens': 91,
             'cache_read_input_tokens': 1952, 'cache_creation_input_tokens': 0, ...}
        """
        self._num_turns = getattr(msg, "num_turns", None)
        self._session_id = getattr(msg, "session_id", None)

        # Token usage from ResultMessage.usage dict (primary source)
        usage = getattr(msg, "usage", None)
        if usage and isinstance(usage, dict):
            input_tokens = usage.get("input_tokens", 0) or 0
            cache_read = usage.get("cache_read_input_tokens", 0) or 0
            cache_create = usage.get("cache_creation_input_tokens", 0) or 0
            output_tokens = usage.get("output_tokens", 0) or 0

            # Total input includes cache tokens
            total_input = input_tokens + cache_read + cache_create

            # Only override if we got real data (don't clobber StreamEvent accumulation)
            if total_input > 0 or output_tokens > 0:
                self._input_tokens = total_input
                self._output_tokens = output_tokens

    def _handle_assistant_message(self, msg):
        """Extract model name and count tool calls from assistant messages."""
        # Model name
        model = getattr(msg, "model", None)
        if model:
            self._model = str(model)

        # Count tool use blocks
        content = getattr(msg, "content", None)
        if not content:
            return

        if isinstance(content, list):
            text_chunks = []
            for block in content:
                block_type = type(block).__name__
                if block_type == "ToolUseBlock":
                    self._tool_call_count += 1
                elif hasattr(block, "type"):
                    attr_type = getattr(block, "type", None)
                    if attr_type == "tool_use":
                        self._tool_call_count += 1
                    elif attr_type == "text":
                        block_text = getattr(block, "text", None)
                        if isinstance(block_text, str) and block_text:
                            text_chunks.append(block_text)

            if text_chunks:
                self._add_completion_candidate("".join(text_chunks), source="assistant")

    def _handle_stream_event(self, msg):
        """Extract token usage from raw Anthropic streaming events.

        Only present when include_partial_messages=True. The event field is a dict.
        These are accumulated across turns; ResultMessage.usage takes precedence
        if present (see _handle_result_message).
        """
        event = getattr(msg, "event", None)
        if event is None:
            return

        event_type = _get(event, "type")

        if event_type == "message_start":
            # Start of a new model message; flush any text from the previous one.
            self._flush_stream_text_buffer()
            self._next_turn_id()
            message = _get(event, "message")
            if message:
                usage = _get(message, "usage")
                if usage:
                    input_tokens = _get(usage, "input_tokens", 0)
                    if input_tokens:
                        self._input_tokens += input_tokens
                model = _get(message, "model")
                if model:
                    self._model = model

        elif event_type == "content_block_delta":
            delta = _get(event, "delta")
            delta_type = _get(delta, "type")
            if delta_type == "text_delta":
                chunk = _get(delta, "text", "")
                if isinstance(chunk, str) and chunk:
                    self._stream_text_buffer.append(chunk)

        elif event_type == "message_stop":
            self._flush_stream_text_buffer()

        elif event_type == "message_delta":
            usage = _get(event, "usage")
            if usage:
                output_tokens = _get(usage, "output_tokens", 0)
                if output_tokens:
                    self._output_tokens += output_tokens

    def _finalize(self, error=None):
        """Set accumulated attributes on the span and end it."""
        if self._finalized:
            return
        self._finalized = True

        try:
            # Final defensive flush when streams end without an explicit message_stop.
            self._flush_stream_text_buffer()
            if error:
                self._span.set_status(Status(StatusCode.ERROR, str(error)))
                _set_span_attribute(self._span, "error.type", type(error).__name__)
            else:
                self._span.set_status(Status(StatusCode.OK))

            # Model — only update response model; request model was set upfront
            if self._model:
                _set_span_attribute(self._span, "gen_ai.response.model", self._model)

            # Token usage
            _set_span_attribute(self._span, "gen_ai.usage.input_tokens", self._input_tokens)
            _set_span_attribute(self._span, "gen_ai.usage.output_tokens", self._output_tokens)
            _set_span_attribute(
                self._span, "llm.usage.total_tokens", self._input_tokens + self._output_tokens
            )
            if self._completion_entries:
                # Emit indexed OpenLLMetry-style completions for multi-turn agent runs.
                for index, (_, _, completion_text) in enumerate(self._completion_entries):
                    _set_span_attribute(self._span, f"llm.completions.{index}.role", "assistant")
                    _set_span_attribute(
                        self._span,
                        f"llm.completions.{index}.content",
                        completion_text[:8000],
                    )

            # Agent-specific attributes
            _set_span_attribute(self._span, "claude_agent.num_turns", self._num_turns)
            _set_span_attribute(self._span, "claude_agent.session_id", self._session_id)
            _set_span_attribute(self._span, "claude_agent.tool_call_count", self._tool_call_count)

        except Exception as e:
            logger.debug(f"Error finalizing agent span: {e}")
        finally:
            self._span.end()
