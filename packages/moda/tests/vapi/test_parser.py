"""Unit tests for Vapi parsing functions."""

import pytest

from moda.vapi.parser import (
    normalize_payload,
    extract_analysis,
    extract_turns,
    extract_tool_calls,
    extract_transfers,
    extract_costs,
    get_turn_timing,
)
from moda.vapi.types import VapiCost

from tests.vapi.fixtures import (
    SAMPLE_PAYLOAD,
    ARTIFACT_NO_TOOLS,
    ARTIFACT_NO_TRANSFERS,
    REAL_WEBHOOK_PAYLOAD,
    REAL_WEBHOOK_TRANSCRIPT_ARRAY,
    REAL_WEBHOOK_TRANSCRIPT_STRING,
    REAL_WEBHOOK_WITH_ARTIFACT,
)


class TestExtractTurns:
    """Tests for extract_turns function."""

    def test_extracts_assistant_turns_from_messages(self):
        """Should extract assistant turns from messages."""
        turns = extract_turns(SAMPLE_PAYLOAD["call"]["artifact"])

        assert len(turns) == 4  # 4 assistant messages
        assert turns[0]["index"] == 0
        assert "I'd be happy to help" in turns[0]["assistantMessage"]["content"]
        assert "Hello" in turns[0]["userMessage"]["content"]

    def test_returns_empty_list_for_none_artifact(self):
        """Should return empty list for None artifact."""
        turns = extract_turns(None)
        assert turns == []

    def test_returns_empty_list_for_empty_messages(self):
        """Should return empty list for empty messages."""
        turns = extract_turns({"messages": []})
        assert turns == []

    def test_associates_timing_data_with_turns(self):
        """Should associate timing data with turns."""
        turns = extract_turns(SAMPLE_PAYLOAD["call"]["artifact"])

        assert turns[0]["timing"]["modelLatency"] == 150
        assert turns[0]["timing"]["totalLatency"] == 350


class TestExtractToolCalls:
    """Tests for extract_tool_calls function."""

    def test_extracts_tool_calls_from_assistant_messages(self):
        """Should extract tool calls from assistant messages."""
        tool_calls = extract_tool_calls(SAMPLE_PAYLOAD["call"]["artifact"])

        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "lookupOrder"
        assert tool_calls[0]["id"] == "tool_call_1"
        assert tool_calls[0]["parameters"] == '{"orderId": "ORDER-123"}'
        assert tool_calls[0]["result"] == '{"status": "shipped", "tracking": "TRK-456"}'

    def test_returns_empty_list_for_none_artifact(self):
        """Should return empty list for None artifact."""
        tool_calls = extract_tool_calls(None)
        assert tool_calls == []

    def test_returns_empty_list_when_no_tool_calls(self):
        """Should return empty list when no tool calls."""
        tool_calls = extract_tool_calls(ARTIFACT_NO_TOOLS)
        assert tool_calls == []


class TestExtractTransfers:
    """Tests for extract_transfers function."""

    def test_extracts_transfer_records(self):
        """Should extract transfer records."""
        transfers = extract_transfers(SAMPLE_PAYLOAD["call"]["artifact"])

        assert len(transfers) == 1
        assert transfers[0]["fromAssistant"] == "receptionist"
        assert transfers[0]["toAssistant"] == "order-support"
        assert transfers[0]["status"] == "completed"

    def test_returns_empty_list_for_none_artifact(self):
        """Should return empty list for None artifact."""
        transfers = extract_transfers(None)
        assert transfers == []

    def test_returns_empty_list_when_no_transfers(self):
        """Should return empty list when no transfers."""
        transfers = extract_transfers(ARTIFACT_NO_TRANSFERS)
        assert transfers == []


class TestExtractCosts:
    """Tests for extract_costs function."""

    def test_extracts_and_normalizes_costs(self):
        """Should extract and normalize costs."""
        costs = extract_costs(SAMPLE_PAYLOAD["call"]["costs"])

        assert costs["model"] == 0.15
        assert costs["transcriber"] == 0.05
        assert costs["voice"] == 0.05

    def test_returns_empty_dict_for_none_costs(self):
        """Should return empty dict for None costs."""
        costs = extract_costs(None)
        assert len(costs) == 0

    def test_normalizes_cost_type_names(self):
        """Should normalize cost type names."""
        costs: list[VapiCost] = [
            {"type": "llm", "cost": 0.10},
            {"type": "stt", "cost": 0.05},
            {"type": "tts", "cost": 0.05},
        ]
        extracted = extract_costs(costs)

        assert extracted["model"] == 0.10
        assert extracted["transcriber"] == 0.05
        assert extracted["voice"] == 0.05

    def test_accumulates_costs_of_same_type(self):
        """Should accumulate costs of same type."""
        costs: list[VapiCost] = [
            {"type": "model", "cost": 0.10},
            {"type": "model", "cost": 0.05},
        ]
        extracted = extract_costs(costs)

        assert extracted["model"] == pytest.approx(0.15)


class TestGetTurnTiming:
    """Tests for get_turn_timing function."""

    def test_gets_timing_by_turn_index(self):
        """Should get timing by turn index."""
        timing = get_turn_timing(SAMPLE_PAYLOAD["call"]["artifact"], 0)

        assert timing is not None
        assert timing["modelLatency"] == 150

    def test_returns_none_for_none_artifact(self):
        """Should return None for None artifact."""
        timing = get_turn_timing(None, 0)
        assert timing is None

    def test_returns_none_for_missing_metrics(self):
        """Should return None for missing metrics."""
        timing = get_turn_timing({"messages": []}, 0)
        assert timing is None

    def test_returns_none_for_out_of_bounds_index(self):
        """Should return None for out of bounds index."""
        timing = get_turn_timing(SAMPLE_PAYLOAD["call"]["artifact"], 100)
        assert timing is None


class TestNormalizePayload:
    """Tests for normalize_payload function."""

    def test_unwraps_message_wrapped_payload(self):
        """Should unwrap a message-wrapped payload."""
        normalized = normalize_payload(REAL_WEBHOOK_PAYLOAD)

        assert normalized["type"] == "end-of-call-report"
        assert normalized["call"]["id"] == "call_real_123"

    def test_passes_through_legacy_payload(self):
        """Should pass through an already-unwrapped legacy payload."""
        normalized = normalize_payload(SAMPLE_PAYLOAD)

        assert normalized["type"] == "end-of-call-report"
        assert normalized["call"]["id"] == "call_abc123"
        # Original artifact messages should be preserved
        assert len(normalized["call"]["artifact"]["messages"]) == 8

    def test_converts_transcript_role_message_to_role_content(self):
        """Should convert {role, message} transcript entries to {role, content}."""
        normalized = normalize_payload(REAL_WEBHOOK_TRANSCRIPT_ARRAY)

        messages = normalized["call"]["artifact"]["messages"]
        assert len(messages) == 4
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Hi there! How can I help?"

    def test_does_not_override_existing_artifact_messages(self):
        """Should not override existing artifact messages with transcript."""
        normalized = normalize_payload(REAL_WEBHOOK_WITH_ARTIFACT)

        messages = normalized["call"]["artifact"]["messages"]
        # Should keep the original 6 artifact messages, not the 2 transcript entries
        assert len(messages) == 6
        assert messages[0]["content"] == "I need to book an appointment"

    def test_merges_analysis_from_message_level(self):
        """Should merge analysis from message-level summary/structuredData."""
        payload = {
            "message": {
                "type": "end-of-call-report",
                "call": {"id": "call_no_analysis"},
                "summary": "Test summary",
                "structuredData": {"key": "value"},
            },
        }
        normalized = normalize_payload(payload)

        assert normalized["call"]["analysis"]["summary"] == "Test summary"
        assert normalized["call"]["analysis"]["structuredData"] == {"key": "value"}

    def test_preserves_call_level_analysis_over_message_level(self):
        """Should preserve call-level analysis when it exists."""
        normalized = normalize_payload(REAL_WEBHOOK_PAYLOAD)

        analysis = normalized["call"]["analysis"]
        assert analysis["summary"] == "Customer asked about billing. Issue resolved."
        assert analysis["successEvaluation"] == "true"

    def test_merges_timing_fields_from_message_level(self):
        """Should merge startedAt/endedAt from message level to call level."""
        normalized = normalize_payload(REAL_WEBHOOK_TRANSCRIPT_ARRAY)

        assert normalized["call"]["startedAt"] == "2024-01-15T14:00:00Z"
        assert normalized["call"]["endedAt"] == "2024-01-15T14:01:00Z"

    def test_handles_string_transcript_gracefully(self):
        """Should handle string transcript without converting to messages."""
        normalized = normalize_payload(REAL_WEBHOOK_TRANSCRIPT_STRING)

        # String transcript should not produce artifact messages
        artifact = normalized["call"].get("artifact", {})
        assert not artifact.get("messages")

    def test_handles_empty_payload(self):
        """Should handle empty/minimal payload gracefully."""
        normalized = normalize_payload({})

        assert normalized["call"] == {}


class TestExtractAnalysis:
    """Tests for extract_analysis function."""

    def test_extracts_analysis_from_call(self):
        """Should extract analysis from call object."""
        call = {
            "id": "test",
            "analysis": {
                "summary": "Test summary",
                "structuredData": {"key": "value"},
                "successEvaluation": "true",
            },
        }
        analysis = extract_analysis(call)

        assert analysis is not None
        assert analysis["summary"] == "Test summary"
        assert analysis["structuredData"] == {"key": "value"}
        assert analysis["successEvaluation"] == "true"

    def test_returns_none_when_no_analysis(self):
        """Should return None when no analysis data."""
        analysis = extract_analysis({"id": "test"})
        assert analysis is None

    def test_returns_none_for_empty_analysis(self):
        """Should return None for empty analysis dict."""
        analysis = extract_analysis({"id": "test", "analysis": {}})
        assert analysis is None
