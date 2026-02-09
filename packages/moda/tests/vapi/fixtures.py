"""Test fixtures for Vapi integration tests."""

from moda.vapi.types import VapiEndOfCallReport, VapiWebhookPayload, VapiCallArtifact

# Sample Vapi end-of-call payload for testing
SAMPLE_PAYLOAD: VapiEndOfCallReport = {
    "type": "end-of-call-report",
    "call": {
        "id": "call_abc123",
        "duration": 120,
        "endedReason": "customer-ended-call",
        "cost": 0.25,
        "customer": {
            "number": "+1234567890",
            "name": "Test User",
        },
        "artifact": {
            "messages": [
                {"role": "user", "content": "Hello, I need help with my order"},
                {
                    "role": "assistant",
                    "content": "Hi! I'd be happy to help you with your order. Can you provide your order number?",
                },
                {"role": "user", "content": "It's ORDER-123"},
                {
                    "role": "assistant",
                    "content": "Let me look that up for you.",
                    "tool_calls": [
                        {
                            "id": "tool_call_1",
                            "type": "function",
                            "function": {
                                "name": "lookupOrder",
                                "arguments": '{"orderId": "ORDER-123"}',
                            },
                        },
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "tool_call_1",
                    "content": '{"status": "shipped", "tracking": "TRK-456"}',
                },
                {
                    "role": "assistant",
                    "content": "Your order ORDER-123 has been shipped! The tracking number is TRK-456.",
                },
                {"role": "user", "content": "Thank you!"},
                {
                    "role": "assistant",
                    "content": "You're welcome! Is there anything else I can help you with?",
                },
            ],
            "transfers": [
                {
                    "fromAssistant": "receptionist",
                    "toAssistant": "order-support",
                    "status": "completed",
                },
            ],
            "performanceMetrics": {
                "turnLatencies": [
                    {
                        "turnIndex": 0,
                        "modelLatency": 150,
                        "voiceLatency": 200,
                        "totalLatency": 350,
                    },
                    {
                        "turnIndex": 1,
                        "modelLatency": 180,
                        "voiceLatency": 210,
                        "totalLatency": 390,
                    },
                    {
                        "turnIndex": 2,
                        "modelLatency": 200,
                        "voiceLatency": 180,
                        "totalLatency": 380,
                    },
                    {
                        "turnIndex": 3,
                        "modelLatency": 140,
                        "voiceLatency": 190,
                        "totalLatency": 330,
                    },
                ],
            },
        },
        "costs": [
            {"type": "model", "cost": 0.15},
            {"type": "transcriber", "cost": 0.05},
            {"type": "voice", "cost": 0.05},
        ],
    },
}


# Minimal payload with just required fields
MINIMAL_PAYLOAD: VapiEndOfCallReport = {
    "type": "end-of-call-report",
    "call": {
        "id": "call_minimal",
    },
}


# Payload with no artifact
PAYLOAD_NO_ARTIFACT: VapiEndOfCallReport = {
    "type": "end-of-call-report",
    "call": {
        "id": "call_no_artifact",
        "duration": 60,
    },
}


# Payload with empty messages
PAYLOAD_EMPTY_MESSAGES: VapiEndOfCallReport = {
    "type": "end-of-call-report",
    "call": {
        "id": "call_empty_messages",
        "artifact": {
            "messages": [],
        },
    },
}


# Non end-of-call-report payload
STATUS_UPDATE_PAYLOAD: VapiWebhookPayload = {
    "type": "status-update",
}


# Artifact with no tool calls
ARTIFACT_NO_TOOLS: VapiCallArtifact = {
    "messages": [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ],
}


# Artifact with empty transfers
ARTIFACT_NO_TRANSFERS: VapiCallArtifact = {
    "messages": [],
}
