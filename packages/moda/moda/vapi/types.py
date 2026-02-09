"""TypedDict definitions for Vapi end-of-call webhook payloads.

These types represent the structure of data sent by Vapi when a call ends.
"""

from typing import Any, Literal, TypedDict


class VapiCustomer(TypedDict, total=False):
    """Customer information from the call."""

    number: str
    """Customer phone number."""

    name: str
    """Customer name."""


class VapiToolCallFunction(TypedDict):
    """Function details in a tool call."""

    name: str
    """Name of the function."""

    arguments: str
    """JSON-stringified arguments."""


class VapiToolCall(TypedDict):
    """Tool call made by the assistant."""

    id: str
    """Unique ID for this tool call."""

    type: Literal["function"]
    """Type of tool (usually "function")."""

    function: VapiToolCallFunction
    """Function details."""


class VapiMessage(TypedDict, total=False):
    """Message in the conversation."""

    role: Literal["system", "user", "assistant", "tool", "function"]
    """Role of the message sender."""

    content: str
    """Message content."""

    name: str
    """Tool/function name (for tool/function roles)."""

    tool_call_id: str
    """Tool call ID (for tool result messages)."""

    tool_calls: list[VapiToolCall]
    """Tool calls made by the assistant."""

    timestamp: float
    """Timestamp of the message."""


class VapiNode(TypedDict, total=False):
    """Workflow node (for squad/workflow calls)."""

    id: str
    """Node identifier."""

    name: str
    """Node name/type."""

    messages: list[VapiMessage]
    """Messages within this node."""

    variables: dict[str, Any]
    """Variables set in this node."""


class VapiTransfer(TypedDict, total=False):
    """Squad transfer record."""

    fromAssistant: str
    """Source assistant identifier."""

    toAssistant: str
    """Destination assistant identifier."""

    status: str
    """Transfer status."""

    timestamp: float
    """Timestamp of the transfer."""


class VapiTurnLatency(TypedDict, total=False):
    """Latency data for a single turn."""

    turnIndex: int
    """Turn index."""

    modelLatency: float
    """Model latency in milliseconds."""

    voiceLatency: float
    """Voice synthesis latency in milliseconds."""

    transcriberLatency: float
    """Transcriber latency in milliseconds."""

    totalLatency: float
    """Total turn latency in milliseconds."""

    startTime: float
    """Start timestamp."""

    endTime: float
    """End timestamp."""


class VapiPerformanceMetrics(TypedDict, total=False):
    """Performance metrics from the call."""

    turnLatencies: list[VapiTurnLatency]
    """Per-turn latency measurements."""


class VapiCallArtifact(TypedDict, total=False):
    """Call artifact containing conversation data and metrics."""

    messages: list[VapiMessage]
    """Full conversation message history."""

    nodes: list[VapiNode]
    """Workflow nodes (for squad/workflow calls)."""

    transfers: list[VapiTransfer]
    """Squad transfer records."""

    performanceMetrics: VapiPerformanceMetrics
    """Performance metrics including turn latencies."""


class VapiCost(TypedDict):
    """Cost breakdown entry."""

    type: str
    """Type of cost (e.g., "model", "transcriber", "voice", "tts", "stt")."""

    cost: float
    """Cost amount."""


class VapiCostWithDetails(VapiCost, total=False):
    """Cost breakdown entry with optional details."""

    details: dict[str, Any]
    """Additional details."""


class VapiCall(TypedDict, total=False):
    """Vapi call object containing all call data."""

    id: str
    """Unique call identifier."""

    duration: float
    """Call duration in seconds."""

    endedReason: str
    """Reason the call ended."""

    cost: float
    """Total cost of the call."""

    customer: VapiCustomer
    """Customer information."""

    artifact: VapiCallArtifact
    """Call artifact containing messages, transfers, and metrics."""

    costs: list[VapiCost]
    """Cost breakdown by component."""


class VapiWebhookPayload(TypedDict, total=False):
    """Root webhook payload from Vapi."""

    type: str
    """Type of webhook event."""

    call: VapiCall
    """The call data (present in end-of-call-report)."""


class VapiEndOfCallReport(TypedDict):
    """Complete Vapi end-of-call report webhook payload."""

    type: Literal["end-of-call-report"]
    """Always "end-of-call-report" for this webhook type."""

    call: VapiCall
    """The call data."""


class ExtractedTurn(TypedDict, total=False):
    """Extracted turn data for span creation."""

    index: int
    """Turn index."""

    userMessage: VapiMessage
    """User message that prompted this turn."""

    assistantMessage: VapiMessage
    """Assistant response."""

    timing: VapiTurnLatency
    """Timing data if available."""


class ExtractedToolCall(TypedDict, total=False):
    """Extracted tool call data for span creation."""

    name: str
    """Tool/function name."""

    id: str
    """Tool call ID."""

    parameters: str
    """Parameters passed to the tool."""

    result: str
    """Result from the tool."""

    index: int
    """Index in the conversation."""


class ProcessVapiOptions(TypedDict, total=False):
    """Options for process_vapi_end_of_call_report."""

    conversation_id: str
    """Override the conversation ID (defaults to call.id)."""

    user_id: str
    """Override the user ID (defaults to customer.number)."""
