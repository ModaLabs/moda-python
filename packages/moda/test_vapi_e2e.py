"""End-to-end test of VAPI integration with real webhook payload."""

import moda
from moda.vapi.spans import is_end_of_call_report, process_vapi_end_of_call_report
from opentelemetry.sdk.trace.export import ConsoleSpanExporter

# Initialize Moda with console exporter so we can see what spans get created
moda.init(
    api_key="REDACTED_API_KEY",
    app_name="vapi-e2e-test",
    exporter=ConsoleSpanExporter(),
)

# Real VAPI webhook payload
real_payload = {
    "message": {
        "type": "end-of-call-report",
        "call": {
            "id": "123e4567-e89b-12d3-a456-426614174000",
            "assistantId": "987f6543-e21b-12d3-a456-426614174000",
            "status": "ended",
            "endedReason": "assistant-ended-call",
            "duration": 155.5,
            "cost": 0.045,
        },
        "summary": "Customer called regarding an issue with their iPhone 13 Pro Max restarting randomly. They have already tried force restarting but the issue persists. Advised them to update to iOS 17.4. Scheduled a Genius Bar appointment for tomorrow at 2 PM.",
        "structuredData": {
            "issue_type": "hardware_failure",
            "device_model": "iPhone 13 Pro Max",
            "troubleshooting_done": ["force_restart"],
            "resolution": "appointment_scheduled",
            "appointment_time": "2024-05-25T14:00:00Z",
            "customer_sentiment": "frustrated",
        },
        "transcript": [
            {"role": "user", "message": "Hi, my phone keeps turning off."},
            {
                "role": "assistant",
                "message": "I'm sorry to hear that. I can help. Which iPhone model are you using?",
            },
            {"role": "user", "message": "It's the 13 Pro Max."},
        ],
    }
}

print("=" * 60)
print("TEST 1: Pass real payload directly (as-is)")
print("=" * 60)
result = is_end_of_call_report(real_payload)
print(f"is_end_of_call_report(real_payload) = {result}")
print()

print("Calling process_vapi_end_of_call_report(real_payload)...")
process_vapi_end_of_call_report(real_payload)
print("(no spans created - returned early)")
print()

print("=" * 60)
print("TEST 2: Unwrap message and pass inner payload")
print("=" * 60)
inner_payload = real_payload["message"]
result2 = is_end_of_call_report(inner_payload)
print(f"is_end_of_call_report(inner_payload) = {result2}")
print()

print("Calling process_vapi_end_of_call_report(inner_payload)...")
process_vapi_end_of_call_report(inner_payload)
print()

print("=" * 60)
print("STRUCTURAL ANALYSIS")
print("=" * 60)
print()
print("What the integration expects:")
print("  payload.type = 'end-of-call-report'")
print("  payload.call.id")
print("  payload.call.artifact.messages[].role / .content")
print("  payload.call.artifact.transfers[]")
print("  payload.call.artifact.performanceMetrics.turnLatencies[]")
print("  payload.call.costs[].type / .cost")
print("  payload.call.customer.number")
print()
print("What the real payload has:")
print(f"  payload.message.type = '{inner_payload.get('type')}'")
print(f"  payload.message.call.id = '{inner_payload.get('call', {}).get('id')}'")
print(f"  payload.message.call has 'artifact'? {inner_payload.get('call', {}).get('artifact') is not None}")
print(f"  payload.message has 'transcript'? {inner_payload.get('transcript') is not None}")
print(f"  payload.message has 'summary'? {inner_payload.get('summary') is not None}")
print(f"  payload.message has 'structuredData'? {inner_payload.get('structuredData') is not None}")
print()
print("ISSUES FOUND:")
print("  1. Real payload wraps everything in 'message' - integration expects top-level 'type'/'call'")
print("  2. Transcript is at payload.message.transcript (with 'message' field), not payload.call.artifact.messages (with 'content' field)")
print("  3. Real payload has 'summary' and 'structuredData' - integration ignores these")
print("  4. Real payload.call has no 'artifact', 'customer', or 'costs' breakdown")
print("  5. Real payload has 'assistantId' and 'status' - integration doesn't capture these")
