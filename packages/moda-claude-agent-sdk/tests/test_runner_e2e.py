"""End-to-end test that replicates the client's actual agent runner script.

This test simulates the exact flow:
  1. moda.init() with a test exporter
  2. moda.conversation_id = chat_id
  3. moda.user_id = user_id
  4. ClaudeSDKClient(options=options) with async-with
  5. await client.query(prompt)
  6. async for msg in client.receive_response():  — with isinstance checks
  7. moda.flush()

The agent stream yields realistic messages: message_start, content_block_start/delta/stop,
message_delta, AssistantMessage with tool results, and ResultMessage.
"""

import sys
import types
import uuid
import json
import pytest

# ---------------------------------------------------------------------------
# Mock the full claude_agent_sdk module with all types the runner uses
# ---------------------------------------------------------------------------

_mod = types.ModuleType("claude_agent_sdk")


class ClaudeAgentOptions:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        if not hasattr(self, "model"):
            self.model = "claude-sonnet-4-20250514"


class ClaudeSDKClient:
    """Mock that supports async-with and yields preset messages."""

    def __init__(self, options=None):
        self.options = options or ClaudeAgentOptions()
        self._messages = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def query(self, prompt):
        pass

    async def receive_response(self):
        for msg in self._messages:
            yield msg


class StreamEvent:
    def __init__(self, event):
        self.event = event


class TextBlock:
    def __init__(self, text=""):
        self.type = "text"
        self.text = text


class ToolUseBlock:
    def __init__(self, id="", name="", input=None):
        self.type = "tool_use"
        self.id = id
        self.name = name
        self.input = input or {}


class ToolResultBlock:
    def __init__(self, tool_use_id="", content=""):
        self.type = "tool_result"
        self.tool_use_id = tool_use_id
        self.content = content


class AssistantMessage:
    def __init__(self, content=None, error=None):
        self.content = content or []
        self.error = error


class UserMessage:
    def __init__(self, content=None):
        self.content = content or []


class ResultMessage:
    def __init__(self, subtype="success", session_id="", num_turns=1):
        self.subtype = subtype
        self.session_id = session_id
        self.num_turns = num_turns


# Install all types into the mock module
_mod.ClaudeAgentOptions = ClaudeAgentOptions
_mod.ClaudeSDKClient = ClaudeSDKClient
_mod.StreamEvent = StreamEvent
_mod.TextBlock = TextBlock
_mod.ToolUseBlock = ToolUseBlock
_mod.ToolResultBlock = ToolResultBlock
_mod.AssistantMessage = AssistantMessage
_mod.UserMessage = UserMessage
_mod.ResultMessage = ResultMessage

from opentelemetry.instrumentation.claude_agent_sdk import ClaudeAgentSDKInstrumentor  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def span_exporter():
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    exporter = InMemorySpanExporter()
    yield exporter
    exporter.clear()


@pytest.fixture()
def tracer_provider(span_exporter):
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    return provider


@pytest.fixture()
def instrumentor(tracer_provider):
    # Replace the mock module so the instrumentor wraps OUR ClaudeSDKClient
    prev = sys.modules.get("claude_agent_sdk")
    sys.modules["claude_agent_sdk"] = _mod

    instr = ClaudeAgentSDKInstrumentor()
    instr.instrument(tracer_provider=tracer_provider, skip_dep_check=True)
    yield instr
    instr.uninstrument()

    # Restore previous mock if any
    if prev is not None:
        sys.modules["claude_agent_sdk"] = prev


# ---------------------------------------------------------------------------
# Build a realistic agent message sequence
# ---------------------------------------------------------------------------

def build_realistic_agent_messages():
    """Simulate what the Claude Agent SDK actually yields during a multi-turn run.

    Sequence:
      Turn 1: Agent thinks, then calls a tool (web_search)
      Turn 2: Agent receives tool result, writes a text response
      Final:  ResultMessage with session_id and num_turns
    """
    messages = []

    # --- Turn 1: message_start (tokens, model) ---
    messages.append(StreamEvent({
        "type": "message_start",
        "message": {
            "model": "claude-sonnet-4-20250514",
            "usage": {"input_tokens": 1200},
        },
    }))

    # content_block_start (text)
    messages.append(StreamEvent({
        "type": "content_block_start",
        "content_block": {"type": "text"},
    }))

    # content_block_delta (text chunk)
    messages.append(StreamEvent({
        "type": "content_block_delta",
        "delta": {"type": "text_delta", "text": "Let me search for that..."},
    }))

    # content_block_stop
    messages.append(StreamEvent({"type": "content_block_stop"}))

    # content_block_start (tool_use)
    messages.append(StreamEvent({
        "type": "content_block_start",
        "content_block": {"type": "tool_use", "id": "tool_01", "name": "web_search"},
    }))

    # content_block_delta (tool input JSON)
    messages.append(StreamEvent({
        "type": "content_block_delta",
        "delta": {"type": "input_json_delta", "partial_json": '{"query": "latest news"}'},
    }))

    # content_block_stop
    messages.append(StreamEvent({"type": "content_block_stop"}))

    # message_delta (output tokens for turn 1)
    messages.append(StreamEvent({
        "type": "message_delta",
        "usage": {"output_tokens": 350},
    }))

    # --- AssistantMessage with tool use block ---
    messages.append(AssistantMessage(content=[
        TextBlock("Let me search for that..."),
        ToolUseBlock(id="tool_01", name="web_search", input={"query": "latest news"}),
    ]))

    # --- UserMessage with tool result (fed back by SDK) ---
    messages.append(UserMessage(content=[
        ToolResultBlock(tool_use_id="tool_01", content=[
            {"type": "text", "text": "Here are the latest results..."},
        ]),
    ]))

    # --- Turn 2: message_start (more input tokens) ---
    messages.append(StreamEvent({
        "type": "message_start",
        "message": {
            "model": "claude-sonnet-4-20250514",
            "usage": {"input_tokens": 1800},
        },
    }))

    # content_block_start (text response)
    messages.append(StreamEvent({
        "type": "content_block_start",
        "content_block": {"type": "text"},
    }))

    # content_block_delta (text chunks)
    messages.append(StreamEvent({
        "type": "content_block_delta",
        "delta": {"type": "text_delta", "text": "Based on my search, here is a summary of the latest news..."},
    }))

    # content_block_stop
    messages.append(StreamEvent({"type": "content_block_stop"}))

    # message_delta (output tokens for turn 2)
    messages.append(StreamEvent({
        "type": "message_delta",
        "usage": {"output_tokens": 520},
    }))

    # --- AssistantMessage (final text) ---
    messages.append(AssistantMessage(content=[
        TextBlock("Based on my search, here is a summary of the latest news..."),
    ]))

    # --- ResultMessage ---
    messages.append(ResultMessage(
        subtype="success",
        session_id="sess-abcdef-123456",
        num_turns=2,
    ))

    return messages


# ---------------------------------------------------------------------------
# The actual test — mirrors the runner script's exact flow
# ---------------------------------------------------------------------------

async def test_runner_e2e(instrumentor, span_exporter):
    """Replicate the client's runner script flow end-to-end."""

    # --- Config (normally from JSON file) ---
    prompt = "What's the latest news about AI?"
    model = "claude-sonnet-4-20250514"

    # --- moda.init() would have been called, our fixture does the equivalent ---
    # moda.conversation_id = chat_id  (we can't set this without the real moda package,
    #   but the instrumentor reads it via get_conversation_id() which returns None here)

    # --- Build agent options (mirrors runner) ---
    options = ClaudeAgentOptions(
        system_prompt="You are a helpful assistant.",
        model=model,
        permission_mode="bypassPermissions",
        include_partial_messages=True,
    )

    # --- Tracking state (mirrors runner) ---
    text_buffer = ""
    text_id = None
    text_streamed = False
    tool_id = None
    tool_input_buffer = ""
    block_type = None
    final_session_id = ""
    events_pushed = []

    def push(event):
        events_pushed.append(event)

    def ensure_started():
        pass

    # --- Run agent (mirrors runner's async with + loop) ---
    async with ClaudeSDKClient(options=options) as client:
        client._messages = build_realistic_agent_messages()

        await client.query(prompt)

        async for msg in client.receive_response():

            # --- StreamEvent: real-time token streaming ---
            if isinstance(msg, StreamEvent):
                event = msg.event
                event_type = event.get("type") if isinstance(event, dict) else getattr(event, "type", None)

                if event_type == "content_block_start":
                    ensure_started()
                    cb = event.get("content_block", {})
                    cb_type = cb.get("type")

                    if cb_type == "text":
                        text_id = str(uuid.uuid4())
                        block_type = "text"

                    elif cb_type == "tool_use":
                        tool_id = cb.get("id")
                        _ = cb.get("name")  # tool_name tracked by runner
                        block_type = "tool_use"
                        tool_input_buffer = ""

                elif event_type == "content_block_delta":
                    delta = event.get("delta", {})
                    delta_type = delta.get("type")

                    if delta_type == "text_delta" and text_id:
                        chunk = delta.get("text", "")
                        if chunk:
                            text_streamed = True
                            text_buffer += chunk

                    elif delta_type == "input_json_delta" and tool_id:
                        tool_input_buffer += delta.get("partial_json", "")

                elif event_type == "content_block_stop":
                    if block_type == "text" and text_id:
                        text_id = None
                        text_buffer = ""
                        block_type = None

                    elif block_type == "tool_use" and tool_id:
                        try:
                            json.loads(tool_input_buffer) if tool_input_buffer else {}
                        except json.JSONDecodeError:
                            pass
                        tool_id = None
                        tool_input_buffer = ""
                        block_type = None

                elif event_type == "message_start":
                    ensure_started()

            # --- AssistantMessage ---
            elif isinstance(msg, AssistantMessage):
                if msg.error:
                    continue
                ensure_started()

            # --- UserMessage ---
            elif isinstance(msg, UserMessage):
                pass

            # --- ResultMessage ---
            elif isinstance(msg, ResultMessage):
                final_session_id = msg.session_id or ""

    # --- Verify Moda captured the span ---
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"

    span = spans[0]

    # Core attributes that Moda dashboard requires
    assert span.name == "claude_agent.chat"
    assert span.attributes["gen_ai.system"] == "Anthropic"
    assert span.attributes["llm.request_type"] == "chat"
    assert span.attributes["gen_ai.request.model"] == "claude-sonnet-4-20250514"

    # Token usage — summed across both turns
    assert span.attributes["gen_ai.usage.input_tokens"] == 3000  # 1200 + 1800
    assert span.attributes["gen_ai.usage.output_tokens"] == 870  # 350 + 520
    assert span.attributes["llm.usage.total_tokens"] == 3870

    # Agent metadata
    assert span.attributes["claude_agent.num_turns"] == 2
    assert span.attributes["claude_agent.session_id"] == "sess-abcdef-123456"
    assert span.attributes["claude_agent.tool_call_count"] == 1  # one web_search

    # Model from stream should match
    assert span.attributes["gen_ai.response.model"] == "claude-sonnet-4-20250514"

    # The runner's own logic should also have worked fine
    assert final_session_id == "sess-abcdef-123456"
    assert text_streamed is True

    print("\n=== RUNNER E2E TEST PASSED ===")
    print(f"  Span name:      {span.name}")
    print(f"  Model:          {span.attributes['gen_ai.request.model']}")
    print(f"  Input tokens:   {span.attributes['gen_ai.usage.input_tokens']}")
    print(f"  Output tokens:  {span.attributes['gen_ai.usage.output_tokens']}")
    print(f"  Total tokens:   {span.attributes['llm.usage.total_tokens']}")
    print(f"  Agent turns:    {span.attributes['claude_agent.num_turns']}")
    print(f"  Session ID:     {span.attributes['claude_agent.session_id']}")
    print(f"  Tool calls:     {span.attributes['claude_agent.tool_call_count']}")
