"""Span creation logic for Vapi call data.

Creates OpenTelemetry spans with proper parent-child relationships.
"""

from typing import Optional

from opentelemetry import trace
from opentelemetry.trace import Span, SpanKind, StatusCode

from moda.vapi.types import (
    VapiCall,
    VapiWebhookPayload,
    VapiTransfer,
    ProcessVapiOptions,
    ExtractedTurn,
    ExtractedToolCall,
)
from moda.vapi.parser import (
    extract_turns,
    extract_tool_calls,
    extract_transfers,
    extract_costs,
)

TRACER_NAME = "moda-sdk"
TRACER_VERSION = "0.1.0"


def is_end_of_call_report(payload: VapiWebhookPayload) -> bool:
    """Type guard to check if a webhook payload is an end-of-call report.

    Args:
        payload: The webhook payload to check.

    Returns:
        True if the payload is an end-of-call report with call data.
    """
    return payload.get("type") == "end-of-call-report" and payload.get("call") is not None


def _create_call_span(
    call: VapiCall,
    options: Optional[ProcessVapiOptions] = None,
) -> Span:
    """Create the parent call span with call-level attributes.

    Args:
        call: The Vapi call data.
        options: Optional configuration for conversation/user ID.

    Returns:
        The created parent span.
    """
    tracer = trace.get_tracer(TRACER_NAME, TRACER_VERSION)

    span = tracer.start_span(
        "vapi.call",
        kind=SpanKind.INTERNAL,
        attributes={
            "llm.vendor": "vapi",
        },
    )

    # Set conversation ID
    conversation_id = (
        options.get("conversation_id") if options else None
    ) or call.get("id")
    if conversation_id:
        span.set_attribute("moda.conversation_id", conversation_id)

    # Set user ID from customer number if available
    user_id = (options.get("user_id") if options else None) or (
        call.get("customer", {}).get("number")
    )
    if user_id:
        span.set_attribute("moda.user_id", user_id)

    # Set call-level attributes
    if call.get("duration") is not None:
        span.set_attribute("vapi.call.duration", call["duration"])

    if call.get("endedReason"):
        span.set_attribute("vapi.call.ended_reason", call["endedReason"])

    if call.get("cost") is not None:
        span.set_attribute("vapi.call.cost", call["cost"])

    # Add cost breakdown attributes
    costs = extract_costs(call.get("costs"))
    for cost_type, amount in costs.items():
        span.set_attribute(f"vapi.cost.{cost_type}", amount)

    return span


def _create_turn_span(
    turn: ExtractedTurn,
    parent_context: trace.Context,
) -> Span:
    """Create a turn span as a child of the parent span.

    Args:
        turn: The extracted turn data.
        parent_context: The parent span's context.

    Returns:
        The created turn span.
    """
    tracer = trace.get_tracer(TRACER_NAME, TRACER_VERSION)

    span = tracer.start_span(
        f"vapi.turn.{turn['index']}",
        context=parent_context,
        kind=SpanKind.INTERNAL,
        attributes={
            "llm.request.type": "chat",
        },
    )

    # Set completion attributes from assistant message
    span.set_attribute("llm.completions.0.role", "assistant")

    assistant_content = turn["assistantMessage"].get("content")
    if assistant_content:
        span.set_attribute("llm.completions.0.content", assistant_content)

    # Set request attributes from user message if present
    user_message = turn.get("userMessage")
    if user_message:
        user_content = user_message.get("content")
        if user_content:
            span.set_attribute("llm.prompts.0.role", "user")
            span.set_attribute("llm.prompts.0.content", user_content)

    # Apply timing if available
    timing = turn.get("timing")
    if timing:
        if timing.get("modelLatency") is not None:
            span.set_attribute("vapi.turn.model_latency_ms", timing["modelLatency"])
        if timing.get("voiceLatency") is not None:
            span.set_attribute("vapi.turn.voice_latency_ms", timing["voiceLatency"])
        if timing.get("transcriberLatency") is not None:
            span.set_attribute("vapi.turn.transcriber_latency_ms", timing["transcriberLatency"])
        if timing.get("totalLatency") is not None:
            span.set_attribute("vapi.turn.total_latency_ms", timing["totalLatency"])

    return span


def _create_tool_span(
    tool_call: ExtractedToolCall,
    parent_context: trace.Context,
) -> Span:
    """Create a tool span as a child of the parent span.

    Args:
        tool_call: The extracted tool call data.
        parent_context: The parent span's context.

    Returns:
        The created tool span.
    """
    tracer = trace.get_tracer(TRACER_NAME, TRACER_VERSION)

    span = tracer.start_span(
        f"vapi.tool.{tool_call['name']}",
        context=parent_context,
        kind=SpanKind.INTERNAL,
        attributes={
            "tool.name": tool_call["name"],
        },
    )

    # Add tool call ID if present
    if tool_call.get("id"):
        span.set_attribute("tool.call_id", tool_call["id"])

    # Add parameters if present
    if tool_call.get("parameters"):
        span.set_attribute("tool.parameters", tool_call["parameters"])

    # Add result if present
    if tool_call.get("result"):
        span.set_attribute("tool.result", tool_call["result"])

    return span


def _create_transfer_span(
    transfer: VapiTransfer,
    index: int,
    parent_context: trace.Context,
) -> Span:
    """Create a transfer span as a child of the parent span.

    Args:
        transfer: The transfer record.
        index: The index of the transfer.
        parent_context: The parent span's context.

    Returns:
        The created transfer span.
    """
    tracer = trace.get_tracer(TRACER_NAME, TRACER_VERSION)

    span = tracer.start_span(
        "vapi.transfer",
        context=parent_context,
        kind=SpanKind.INTERNAL,
    )

    # Add transfer attributes
    if transfer.get("fromAssistant"):
        span.set_attribute("vapi.transfer.from_assistant", transfer["fromAssistant"])

    if transfer.get("toAssistant"):
        span.set_attribute("vapi.transfer.to_assistant", transfer["toAssistant"])

    if transfer.get("status"):
        span.set_attribute("vapi.transfer.status", transfer["status"])

    span.set_attribute("vapi.transfer.index", index)

    return span


def _create_all_spans(
    call: VapiCall,
    options: Optional[ProcessVapiOptions] = None,
) -> tuple[Span, list[Span]]:
    """Create all spans for a Vapi call.

    Args:
        call: The Vapi call data.
        options: Optional configuration.

    Returns:
        Tuple of (parent_span, list of child_spans).
    """
    child_spans: list[Span] = []

    # Create parent call span
    parent_span = _create_call_span(call, options)

    artifact = call.get("artifact")

    # Create context with parent span for child spans
    parent_context = trace.set_span_in_context(parent_span)

    # Create turn spans
    turns = extract_turns(artifact)
    for turn in turns:
        turn_span = _create_turn_span(turn, parent_context)
        turn_span.set_status(StatusCode.OK)
        turn_span.end()
        child_spans.append(turn_span)

    # Create tool spans
    tool_calls = extract_tool_calls(artifact)
    for tool_call in tool_calls:
        tool_span = _create_tool_span(tool_call, parent_context)
        tool_span.set_status(StatusCode.OK)
        tool_span.end()
        child_spans.append(tool_span)

    # Create transfer spans
    transfers = extract_transfers(artifact)
    for index, transfer in enumerate(transfers):
        transfer_span = _create_transfer_span(transfer, index, parent_context)
        transfer_span.set_status(StatusCode.OK)
        transfer_span.end()
        child_spans.append(transfer_span)

    return parent_span, child_spans


def process_vapi_end_of_call_report(
    payload: VapiWebhookPayload,
    options: Optional[ProcessVapiOptions] = None,
) -> None:
    """Process a Vapi end-of-call report webhook and create OpenTelemetry spans.

    This function parses the webhook payload and creates:
    - A parent span for the entire call (`vapi.call`)
    - Child spans for each LLM turn (`vapi.turn.{index}`)
    - Child spans for each tool execution (`vapi.tool.{name}`)
    - Child spans for squad transfers (`vapi.transfer`)

    The function returns early without creating spans if:
    - The webhook type is not "end-of-call-report"
    - The call data is missing

    Args:
        payload: The Vapi webhook payload.
        options: Optional configuration with conversation_id and user_id overrides.

    Returns:
        None

    Example:
        ```python
        import moda
        from moda import process_vapi_end_of_call_report

        # Initialize Moda first
        moda.init("moda_xxx")

        # In your FastAPI/Flask webhook handler
        @app.post("/webhooks/vapi")
        def vapi_webhook(payload: dict):
            process_vapi_end_of_call_report(payload)
            return {"status": "ok"}

        # With custom conversation/user ID
        process_vapi_end_of_call_report(payload, {
            "conversation_id": "custom_session_123",
            "user_id": "user_456",
        })
        ```
    """
    # Validate webhook type
    if not is_end_of_call_report(payload):
        return

    call = payload["call"]

    # Create all spans (parent + children)
    parent_span, _ = _create_all_spans(call, options)

    # Mark parent span as successful and end it
    parent_span.set_status(StatusCode.OK)
    parent_span.end()
