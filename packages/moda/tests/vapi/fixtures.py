"""Test fixtures for Vapi integration tests."""

from typing import Any

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


# ---- Real VAPI webhook format fixtures ----

# Full real webhook payload with message wrapper
REAL_WEBHOOK_PAYLOAD: dict[str, Any] = {
    "message": {
        "type": "end-of-call-report",
        "call": {
            "id": "call_real_123",
            "assistantId": "asst_456",
            "status": "ended",
            "endedReason": "assistant-ended-call",
            "duration": 155.5,
            "cost": 0.045,
            "customer": {
                "number": "+19876543210",
                "name": "Real User",
            },
            "analysis": {
                "summary": "Customer asked about billing. Issue resolved.",
                "structuredData": {"intent": "billing_inquiry", "resolved": True},
                "successEvaluation": "true",
            },
            "costs": [
                {"type": "model", "cost": 0.03},
                {"type": "transcriber", "cost": 0.01},
                {"type": "voice", "cost": 0.005},
            ],
        },
        "transcript": [
            {"role": "user", "message": "Hi, I have a question about my bill"},
            {
                "role": "assistant",
                "message": "Of course! I'd be happy to help with your billing question. What would you like to know?",
            },
            {"role": "user", "message": "Why was I charged twice?"},
            {
                "role": "assistant",
                "message": "Let me look into that for you. I can see the duplicate charge and will process a refund.",
            },
        ],
        "summary": "Customer asked about billing. Issue resolved.",
        "structuredData": {"intent": "billing_inquiry", "resolved": True},
        "startedAt": "2024-01-15T10:30:00Z",
        "endedAt": "2024-01-15T10:32:35Z",
        "cost": 0.15,
        "recordingUrl": "https://recordings.vapi.ai/call_real_123.wav",
        "stereoRecordingUrl": "https://recordings.vapi.ai/call_real_123_stereo.wav",
    },
}


# Real webhook with transcript as array of {role, message}
REAL_WEBHOOK_TRANSCRIPT_ARRAY: dict[str, Any] = {
    "message": {
        "type": "end-of-call-report",
        "call": {
            "id": "call_transcript_arr",
            "status": "ended",
            "duration": 60,
        },
        "transcript": [
            {"role": "user", "message": "Hello"},
            {"role": "assistant", "message": "Hi there! How can I help?"},
            {"role": "user", "message": "What time do you close?"},
            {"role": "assistant", "message": "We close at 9 PM."},
        ],
        "startedAt": "2024-01-15T14:00:00Z",
        "endedAt": "2024-01-15T14:01:00Z",
    },
}


# Real webhook with transcript as flat string
REAL_WEBHOOK_TRANSCRIPT_STRING: dict[str, Any] = {
    "message": {
        "type": "end-of-call-report",
        "call": {
            "id": "call_transcript_str",
            "status": "ended",
            "duration": 30,
        },
        "transcript": "User: Hello\nAssistant: Hi there!",
        "startedAt": "2024-01-15T15:00:00Z",
        "endedAt": "2024-01-15T15:00:30Z",
    },
}


# Real webhook that has both call.artifact.messages and message.transcript
REAL_WEBHOOK_WITH_ARTIFACT: dict[str, Any] = {
    "message": {
        "type": "end-of-call-report",
        "call": {
            "id": "call_with_artifact",
            "assistantId": "asst_789",
            "status": "ended",
            "duration": 90,
            "cost": 0.08,
            "artifact": {
                "messages": [
                    {"role": "user", "content": "I need to book an appointment"},
                    {
                        "role": "assistant",
                        "content": "I'd be happy to help you book an appointment. What day works best?",
                    },
                    {"role": "user", "content": "Next Tuesday"},
                    {
                        "role": "assistant",
                        "content": "Let me check availability for next Tuesday.",
                        "tool_calls": [
                            {
                                "id": "tool_call_2",
                                "type": "function",
                                "function": {
                                    "name": "checkAvailability",
                                    "arguments": '{"date": "2024-01-23"}',
                                },
                            },
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "tool_call_2",
                        "content": '{"available": true, "slots": ["10:00", "14:00"]}',
                    },
                    {
                        "role": "assistant",
                        "content": "Great news! We have slots at 10 AM and 2 PM next Tuesday.",
                    },
                ],
            },
            "analysis": {
                "summary": "Customer booked an appointment for next Tuesday.",
                "successEvaluation": "true",
            },
            "costs": [
                {"type": "model", "cost": 0.05},
                {"type": "voice", "cost": 0.03},
            ],
        },
        "transcript": [
            {"role": "user", "message": "I need to book an appointment"},
            {"role": "assistant", "message": "I'd be happy to help you book an appointment."},
        ],
        "summary": "Customer booked an appointment for next Tuesday.",
        "startedAt": "2024-01-15T11:00:00Z",
        "endedAt": "2024-01-15T11:01:30Z",
    },
}
