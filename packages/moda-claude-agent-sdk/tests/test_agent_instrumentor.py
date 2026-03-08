"""End-to-end tests for the Claude Agent SDK instrumentor.

Mocks the claude_agent_sdk module (which spawns Claude Code as a subprocess)
and verifies that the instrumentor correctly creates spans with accumulated
token usage, tool call counts, and agent metadata.
"""

import sys
import types
from unittest.mock import MagicMock

import pytest

class ErrorSentinel:
    """Sentinel: when the mock stream encounters this, it raises the error."""

    def __init__(self, error):
        self.error = error


# ---------------------------------------------------------------------------
# Mock claude_agent_sdk module — must be installed before importing instrumentor
# ---------------------------------------------------------------------------

def _build_mock_module():
    """Build a fake claude_agent_sdk module with a real ClaudeSDKClient class."""

    mod = types.ModuleType("claude_agent_sdk")

    class ClaudeSDKClient:
        def __init__(self, options=None):
            self.options = options or MagicMock(model="claude-sonnet-4-20250514")
            self._messages = []  # messages to yield in receive_response

        async def query(self, prompt):
            """Send prompt to the agent subprocess."""
            pass

        async def receive_response(self):
            """Async generator of agent messages (matches real SDK)."""
            for msg in self._messages:
                if isinstance(msg, ErrorSentinel):
                    raise msg.error
                yield msg

    mod.ClaudeSDKClient = ClaudeSDKClient
    return mod


# Install the mock module before any instrumentation import touches it
_mock_mod = _build_mock_module()
sys.modules["claude_agent_sdk"] = _mock_mod

from opentelemetry.instrumentation.claude_agent_sdk import ClaudeAgentSDKInstrumentor  # noqa: E402


# ---------------------------------------------------------------------------
# Mock message types that the Claude Agent SDK yields
# ---------------------------------------------------------------------------

class StreamEvent:
    """Mimics a raw Anthropic streaming event forwarded by the agent."""

    def __init__(self, event_type, **kwargs):
        self.event = MagicMock()
        self.event.type = event_type

        if event_type == "message_start":
            message = MagicMock()
            message.usage = MagicMock(
                input_tokens=kwargs.get("input_tokens", 0),
            )
            message.model = kwargs.get("model", "claude-sonnet-4-20250514")
            self.event.message = message
        elif event_type == "message_delta":
            self.event.usage = MagicMock(
                output_tokens=kwargs.get("output_tokens", 0),
            )
            self.event.message = None
        else:
            self.event.message = None
            self.event.usage = None


class AssistantMessage:
    """Mimics an assistant response with content blocks."""

    def __init__(self, text_blocks=0, tool_use_blocks=0):
        self.content = []
        for _ in range(text_blocks):
            block = MagicMock()
            block.type = "text"
            self.content.append(block)
        for _ in range(tool_use_blocks):
            block = MagicMock()
            block.type = "tool_use"
            self.content.append(block)


class ResultMessage:
    """Mimics the final result message from an agent run."""

    def __init__(self, num_turns=1, session_id="sess-abc-123"):
        self.num_turns = num_turns
        self.session_id = session_id


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def instrumentor(tracer_provider):
    instr = ClaudeAgentSDKInstrumentor()
    instr.instrument(tracer_provider=tracer_provider, skip_dep_check=True)
    yield instr
    instr.uninstrument()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_client(messages):
    """Create a ClaudeSDKClient with preset messages to yield."""
    client = _mock_mod.ClaudeSDKClient()
    client._messages = messages
    return client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

async def test_single_turn_agent_run(instrumentor, span_exporter):
    """A simple single-turn agent run should produce one span with correct attributes."""

    client = _make_client([
        StreamEvent("message_start", input_tokens=150, model="claude-sonnet-4-20250514"),
        AssistantMessage(text_blocks=1),
        StreamEvent("message_delta", output_tokens=42),
        ResultMessage(num_turns=1, session_id="sess-001"),
    ])

    await client.query("What is 2+2?")
    async for _ in client.receive_response():
        pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == "claude_agent.chat"
    assert span.attributes.get("gen_ai.system") == "Anthropic"
    assert span.attributes.get("llm.request_type") == "chat"
    assert span.attributes.get("gen_ai.request.model") == "claude-sonnet-4-20250514"
    assert span.attributes.get("gen_ai.response.model") == "claude-sonnet-4-20250514"
    assert span.attributes.get("gen_ai.usage.input_tokens") == 150
    assert span.attributes.get("gen_ai.usage.output_tokens") == 42
    assert span.attributes.get("llm.usage.total_tokens") == 192
    assert span.attributes.get("claude_agent.num_turns") == 1
    assert span.attributes.get("claude_agent.session_id") == "sess-001"


async def test_multi_turn_agent_run(instrumentor, span_exporter):
    """Multi-turn agent runs should sum tokens across all turns."""

    client = _make_client([
        # Turn 1
        StreamEvent("message_start", input_tokens=200, model="claude-sonnet-4-20250514"),
        AssistantMessage(text_blocks=1, tool_use_blocks=1),
        StreamEvent("message_delta", output_tokens=80),
        # Turn 2
        StreamEvent("message_start", input_tokens=350),
        AssistantMessage(text_blocks=1, tool_use_blocks=2),
        StreamEvent("message_delta", output_tokens=120),
        # Final
        ResultMessage(num_turns=2, session_id="sess-multi"),
    ])

    await client.query("Search for X then summarize")
    async for _ in client.receive_response():
        pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.attributes.get("gen_ai.usage.input_tokens") == 550  # 200 + 350
    assert span.attributes.get("gen_ai.usage.output_tokens") == 200  # 80 + 120
    assert span.attributes.get("llm.usage.total_tokens") == 750
    assert span.attributes.get("claude_agent.num_turns") == 2
    assert span.attributes.get("claude_agent.tool_call_count") == 3  # 1 + 2


async def test_multi_turn_emits_indexed_completion_attributes(instrumentor, span_exporter):
    """Assistant messages with text should be emitted as indexed llm.completions.N.* attributes."""

    class RealAssistantMessage:
        def __init__(self, model=None, content=None):
            self.model = model
            self.content = content or []

    client = _make_client([
        RealAssistantMessage(
            model="claude-sonnet-4-20250514",
            content=[MagicMock(type="text", text="First answer")],
        ),
        RealAssistantMessage(content=[MagicMock(type="text", text="Second answer")]),
        ResultMessage(num_turns=2, session_id="sess-completions"),
    ])

    await client.query("Give two answers")
    async for _ in client.receive_response():
        pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.attributes.get("llm.completions.0.role") == "assistant"
    assert span.attributes.get("llm.completions.0.content") == "First answer"
    assert span.attributes.get("llm.completions.1.role") == "assistant"
    assert span.attributes.get("llm.completions.1.content") == "Second answer"


async def test_tool_call_counting(instrumentor, span_exporter):
    """Tool calls across multiple assistant messages should be counted."""

    client = _make_client([
        StreamEvent("message_start", input_tokens=100),
        AssistantMessage(tool_use_blocks=3),
        StreamEvent("message_delta", output_tokens=50),
        AssistantMessage(tool_use_blocks=2),
        ResultMessage(num_turns=1, session_id="sess-tools"),
    ])

    await client.query("Use multiple tools")
    async for _ in client.receive_response():
        pass

    spans = span_exporter.get_finished_spans()
    span = spans[0]
    assert span.attributes.get("claude_agent.tool_call_count") == 5


async def test_empty_stream(instrumentor, span_exporter):
    """An empty stream (no messages) should still produce a valid span."""

    client = _make_client([])

    await client.query("Hello")
    async for _ in client.receive_response():
        pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.attributes.get("gen_ai.system") == "Anthropic"
    assert span.attributes.get("gen_ai.usage.input_tokens") == 0
    assert span.attributes.get("gen_ai.usage.output_tokens") == 0
    assert span.attributes.get("llm.usage.total_tokens") == 0


async def test_stream_error_sets_error_status(instrumentor, span_exporter):
    """If the stream raises an exception, the span should record the error."""

    # Use ErrorSentinel in messages list — the mock yields it and then the
    # WrappedAgentStream propagates it as an exception
    error = ConnectionError("subprocess crashed")
    client = _make_client([
        StreamEvent("message_start", input_tokens=50),
        ErrorSentinel(error),
    ])

    await client.query("Hello")
    with pytest.raises(ConnectionError, match="subprocess crashed"):
        async for _ in client.receive_response():
            pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.status.status_code.name == "ERROR"
    assert span.attributes.get("error.type") == "ConnectionError"
    # Tokens seen before crash should still be recorded
    assert span.attributes.get("gen_ai.usage.input_tokens") == 50


async def test_model_from_instance_options(instrumentor, span_exporter):
    """Model name should be extracted from instance.options.model."""

    options = MagicMock()
    options.model = "claude-opus-4-20250514"
    client = _mock_mod.ClaudeSDKClient(options=options)
    client._messages = [
        StreamEvent("message_start", input_tokens=10, model="claude-opus-4-20250514"),
        StreamEvent("message_delta", output_tokens=5),
        ResultMessage(num_turns=1, session_id="sess-opus"),
    ]

    await client.query("Hello")
    async for _ in client.receive_response():
        pass

    spans = span_exporter.get_finished_spans()
    span = spans[0]
    assert span.attributes.get("gen_ai.request.model") == "claude-opus-4-20250514"


async def test_prompt_captured_from_query(instrumentor, span_exporter):
    """The prompt passed to query() should be stored and set on the span."""

    client = _make_client([
        StreamEvent("message_start", input_tokens=10),
        StreamEvent("message_delta", output_tokens=5),
        ResultMessage(num_turns=1),
    ])

    await client.query("Explain quantum computing")
    async for _ in client.receive_response():
        pass

    spans = span_exporter.get_finished_spans()
    span = spans[0]
    assert span.attributes.get("gen_ai.prompt") == "Explain quantum computing"
    assert span.attributes.get("llm.prompts.0.role") == "user"
    assert span.attributes.get("llm.prompts.0.content") == "Explain quantum computing"


async def test_dict_style_stream_events(instrumentor, span_exporter):
    """Events can be dicts (not objects) depending on SDK version. Tokens must still be captured."""

    class DictStreamEvent:
        """StreamEvent where .event is a plain dict (as seen in some SDK versions)."""
        def __init__(self, event_dict):
            self.event = event_dict

    client = _make_client([
        DictStreamEvent({
            "type": "message_start",
            "message": {
                "usage": {"input_tokens": 300},
                "model": "claude-sonnet-4-20250514",
            },
        }),
        AssistantMessage(text_blocks=1),
        DictStreamEvent({
            "type": "message_delta",
            "usage": {"output_tokens": 75},
        }),
        ResultMessage(num_turns=1, session_id="sess-dict"),
    ])

    await client.query("Dict events test")
    async for _ in client.receive_response():
        pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.attributes.get("gen_ai.usage.input_tokens") == 300
    assert span.attributes.get("gen_ai.usage.output_tokens") == 75
    assert span.attributes.get("llm.usage.total_tokens") == 375
    assert span.attributes.get("gen_ai.response.model") == "claude-sonnet-4-20250514"
    assert span.attributes.get("claude_agent.session_id") == "sess-dict"


async def test_stream_text_delta_emits_completion_without_assistant_text(instrumentor, span_exporter):
    """Stream text_deltas should emit llm.completions even when AssistantMessage has no text blocks."""

    class DictStreamEvent:
        def __init__(self, event_dict):
            self.event = event_dict

    client = _make_client([
        DictStreamEvent({
            "type": "message_start",
            "message": {
                "usage": {"input_tokens": 120},
                "model": "claude-sonnet-4-20250514",
            },
        }),
        DictStreamEvent({
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "stream-only completion"},
        }),
        DictStreamEvent({"type": "message_stop"}),
        DictStreamEvent({
            "type": "message_delta",
            "usage": {"output_tokens": 22},
        }),
        ResultMessage(num_turns=1, session_id="sess-stream-only"),
    ])

    await client.query("Stream-only completion test")
    async for _ in client.receive_response():
        pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.attributes.get("llm.completions.0.role") == "assistant"
    assert span.attributes.get("llm.completions.0.content") == "stream-only completion"
    assert span.attributes.get("gen_ai.usage.input_tokens") == 120
    assert span.attributes.get("gen_ai.usage.output_tokens") == 22


async def test_dedupes_adjacent_stream_and_assistant_completion_text(instrumentor, span_exporter):
    """The same completion seen in stream + assistant events should be emitted once."""

    class DictStreamEvent:
        def __init__(self, event_dict):
            self.event = event_dict

    class RealAssistantMessage:
        def __init__(self, model=None, content=None):
            self.model = model
            self.content = content or []

    client = _make_client([
        DictStreamEvent({
            "type": "message_start",
            "message": {
                "usage": {"input_tokens": 80},
                "model": "claude-sonnet-4-20250514",
            },
        }),
        DictStreamEvent({
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "duplicate completion"},
        }),
        DictStreamEvent({"type": "message_stop"}),
        RealAssistantMessage(
            model="claude-sonnet-4-20250514",
            content=[MagicMock(type="text", text="duplicate completion")],
        ),
        ResultMessage(num_turns=1, session_id="sess-dedupe"),
    ])

    await client.query("Dedupe test")
    async for _ in client.receive_response():
        pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.attributes.get("llm.completions.0.content") == "duplicate completion"
    assert span.attributes.get("llm.completions.1.content") is None


async def test_uninstrument_removes_wrapping(tracer_provider, span_exporter):
    """After uninstrument(), calls should not create spans."""

    instr = ClaudeAgentSDKInstrumentor()
    instr.instrument(tracer_provider=tracer_provider, skip_dep_check=True)

    client = _make_client([
        StreamEvent("message_start", input_tokens=10),
        ResultMessage(num_turns=1),
    ])

    # First call — instrumented
    await client.query("test")
    async for _ in client.receive_response():
        pass
    assert len(span_exporter.get_finished_spans()) == 1

    # Uninstrument
    instr.uninstrument()
    span_exporter.clear()

    # Second call — should NOT create spans
    client2 = _make_client([
        StreamEvent("message_start", input_tokens=10),
        ResultMessage(num_turns=1),
    ])

    await client2.query("test2")
    async for _ in client2.receive_response():
        pass

    assert len(span_exporter.get_finished_spans()) == 0


async def test_multiple_sequential_agent_runs(instrumentor, span_exporter):
    """Multiple sequential agent runs should each produce their own span."""

    for i in range(3):
        client = _make_client([
            StreamEvent("message_start", input_tokens=100 * (i + 1)),
            StreamEvent("message_delta", output_tokens=10 * (i + 1)),
            ResultMessage(num_turns=i + 1, session_id=f"sess-{i}"),
        ])

        await client.query(f"Query {i}")
        async for _ in client.receive_response():
            pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 3

    # Verify each span has distinct data
    for i, span in enumerate(spans):
        assert span.attributes.get("gen_ai.usage.input_tokens") == 100 * (i + 1)
        assert span.attributes.get("claude_agent.num_turns") == i + 1
        assert span.attributes.get("claude_agent.session_id") == f"sess-{i}"


async def test_early_break_finalizes_span(instrumentor, span_exporter):
    """Breaking out of the stream early should still finalize the span."""

    client = _make_client([
        StreamEvent("message_start", input_tokens=100),
        AssistantMessage(text_blocks=1),
        StreamEvent("message_delta", output_tokens=50),
        ResultMessage(num_turns=1, session_id="sess-break"),
    ])

    await client.query("Break early")
    gen = client.receive_response()
    async for msg in gen:
        break  # exit after first message
    # Explicitly close the async generator to ensure cleanup runs
    await gen.aclose()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1  # span must be finalized even with early break


# ---------------------------------------------------------------------------
# Real SDK behavior tests (no StreamEvent, tokens from ResultMessage.usage)
# ---------------------------------------------------------------------------

async def test_real_sdk_behavior_tokens_from_result_message(instrumentor, span_exporter):
    """Real SDK yields AssistantMessage + ResultMessage (no StreamEvent in default mode).

    Token data comes from ResultMessage.usage dict, model from AssistantMessage.model.
    This matches the actual SDK behavior discovered during live testing.
    """

    class RealAssistantMessage:
        """Mimics real AssistantMessage with model field."""
        def __init__(self, model=None, content=None):
            self.model = model
            self.content = content or []

    class RealResultMessage:
        """Mimics real ResultMessage with usage dict."""
        def __init__(self, num_turns=1, session_id="", usage=None):
            self.num_turns = num_turns
            self.session_id = session_id
            self.usage = usage  # dict like real SDK

    client = _make_client([
        RealAssistantMessage(
            model="claude-sonnet-4-20250514",
            content=[MagicMock(type="text", text="Hello from assistant")],
        ),
        RealResultMessage(
            num_turns=1,
            session_id="sess-real-001",
            usage={
                "input_tokens": 10,
                "output_tokens": 91,
                "cache_read_input_tokens": 1952,
                "cache_creation_input_tokens": 0,
            },
        ),
    ])

    await client.query("Hello from real SDK")
    async for _ in client.receive_response():
        pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    # Total input = input_tokens + cache_read + cache_create = 10 + 1952 + 0 = 1962
    assert span.attributes.get("gen_ai.usage.input_tokens") == 1962
    assert span.attributes.get("gen_ai.usage.output_tokens") == 91
    assert span.attributes.get("llm.usage.total_tokens") == 2053
    assert span.attributes.get("gen_ai.response.model") == "claude-sonnet-4-20250514"
    assert span.attributes.get("llm.completions.0.role") == "assistant"
    assert span.attributes.get("llm.completions.0.content") == "Hello from assistant"
    assert span.attributes.get("claude_agent.num_turns") == 1
    assert span.attributes.get("claude_agent.session_id") == "sess-real-001"


async def test_result_message_usage_overrides_stream_events(instrumentor, span_exporter):
    """If both StreamEvent tokens and ResultMessage.usage are present,
    ResultMessage.usage should take precedence (it's the authoritative source).
    """

    class RealResultMessage:
        def __init__(self, num_turns=1, session_id="", usage=None):
            self.num_turns = num_turns
            self.session_id = session_id
            self.usage = usage

    client = _make_client([
        # StreamEvent tokens (partial/accumulated)
        StreamEvent("message_start", input_tokens=100),
        StreamEvent("message_delta", output_tokens=50),
        # ResultMessage with authoritative totals
        RealResultMessage(
            num_turns=1,
            session_id="sess-override",
            usage={
                "input_tokens": 500,
                "output_tokens": 200,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
        ),
    ])

    await client.query("Override test")
    async for _ in client.receive_response():
        pass

    spans = span_exporter.get_finished_spans()
    span = spans[0]
    # ResultMessage.usage should override StreamEvent accumulation
    assert span.attributes.get("gen_ai.usage.input_tokens") == 500
    assert span.attributes.get("gen_ai.usage.output_tokens") == 200
    assert span.attributes.get("llm.usage.total_tokens") == 700


# ---------------------------------------------------------------------------
# PostHog / Sentry coexistence tests
# ---------------------------------------------------------------------------

async def test_coexists_with_multiple_span_processors(span_exporter):
    """Verify spans flow to multiple span processors (simulating Sentry + Moda)."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    # Simulate two consumers: "Moda" exporter and "Sentry" exporter
    moda_exporter = InMemorySpanExporter()
    sentry_exporter = InMemorySpanExporter()

    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(moda_exporter))
    provider.add_span_processor(SimpleSpanProcessor(sentry_exporter))

    instr = ClaudeAgentSDKInstrumentor()
    instr.instrument(tracer_provider=provider, skip_dep_check=True)

    try:
        client = _make_client([
            StreamEvent("message_start", input_tokens=100),
            StreamEvent("message_delta", output_tokens=50),
            ResultMessage(num_turns=1, session_id="sess-dual"),
        ])

        await client.query("Test coexistence")
        async for _ in client.receive_response():
            pass

        # Both exporters should receive the same span
        moda_spans = moda_exporter.get_finished_spans()
        sentry_spans = sentry_exporter.get_finished_spans()

        assert len(moda_spans) == 1
        assert len(sentry_spans) == 1

        # Same span data in both
        assert moda_spans[0].name == sentry_spans[0].name == "claude_agent.chat"
        assert moda_spans[0].attributes.get("gen_ai.usage.input_tokens") == 100
        assert sentry_spans[0].attributes.get("gen_ai.usage.input_tokens") == 100
    finally:
        instr.uninstrument()


async def test_no_interference_with_existing_tracer_provider():
    """Moda instrumentor should work with a pre-existing TracerProvider (Sentry scenario)."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    # Simulate Sentry having already set up a provider with its processor
    existing_provider = TracerProvider()
    sentry_exporter = InMemorySpanExporter()
    existing_provider.add_span_processor(SimpleSpanProcessor(sentry_exporter))

    # Moda adds its own processor to the SAME provider
    moda_exporter = InMemorySpanExporter()
    existing_provider.add_span_processor(SimpleSpanProcessor(moda_exporter))

    # Instrument with the shared provider
    instr = ClaudeAgentSDKInstrumentor()
    instr.instrument(tracer_provider=existing_provider, skip_dep_check=True)

    try:
        client = _make_client([
            StreamEvent("message_start", input_tokens=200),
            StreamEvent("message_delta", output_tokens=75),
            ResultMessage(num_turns=1),
        ])

        await client.query("Test shared provider")
        async for _ in client.receive_response():
            pass

        # Both should see the span
        assert len(sentry_exporter.get_finished_spans()) == 1
        assert len(moda_exporter.get_finished_spans()) == 1
        assert sentry_exporter.get_finished_spans()[0].attributes.get("gen_ai.system") == "Anthropic"
    finally:
        instr.uninstrument()
