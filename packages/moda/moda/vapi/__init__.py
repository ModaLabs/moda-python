"""Vapi integration for Moda LLM observability.

Processes Vapi end-of-call report webhooks and creates OpenTelemetry spans
for the full conversation, including LLM turns, tool calls, and squad transfers.

Example:
    from moda import process_vapi_end_of_call_report

    # In your webhook handler
    @app.post("/vapi/webhook")
    def vapi_webhook(payload: dict):
        process_vapi_end_of_call_report(payload)
        return {"status": "ok"}
"""

from moda.vapi.types import (
    VapiAnalysis,
    VapiEndOfCallReport,
    VapiMessagePayload,
    VapiRealWebhookPayload,
    VapiTranscriptEntry,
    VapiWebhookPayload,
    VapiCall,
    VapiCallArtifact,
    VapiMessage,
    VapiToolCall,
    VapiNode,
    VapiTransfer,
    VapiPerformanceMetrics,
    VapiTurnLatency,
    VapiCost,
    VapiCustomer,
    ProcessVapiOptions,
    ExtractedTurn,
    ExtractedToolCall,
)

from moda.vapi.parser import (
    normalize_payload,
    extract_analysis,
    extract_turns,
    extract_tool_calls,
    extract_transfers,
    extract_costs,
    get_turn_timing,
)

from moda.vapi.spans import (
    process_vapi_end_of_call_report,
    is_end_of_call_report,
)

__all__ = [
    # Main functions
    "process_vapi_end_of_call_report",
    "is_end_of_call_report",
    "normalize_payload",
    "extract_analysis",
    # Types
    "VapiAnalysis",
    "VapiEndOfCallReport",
    "VapiMessagePayload",
    "VapiRealWebhookPayload",
    "VapiTranscriptEntry",
    "VapiWebhookPayload",
    "VapiCall",
    "VapiCallArtifact",
    "VapiMessage",
    "VapiToolCall",
    "VapiNode",
    "VapiTransfer",
    "VapiPerformanceMetrics",
    "VapiTurnLatency",
    "VapiCost",
    "VapiCustomer",
    "ProcessVapiOptions",
    "ExtractedTurn",
    "ExtractedToolCall",
    # Parsing utilities
    "extract_turns",
    "extract_tool_calls",
    "extract_transfers",
    "extract_costs",
    "get_turn_timing",
]
