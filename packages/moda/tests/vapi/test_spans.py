"""Unit tests for Vapi span creation and main entry point."""

import pytest
from unittest.mock import MagicMock, patch


from moda.vapi.spans import (
    process_vapi_end_of_call_report,
    is_end_of_call_report,
)

from tests.vapi.fixtures import (
    SAMPLE_PAYLOAD,
    MINIMAL_PAYLOAD,
    PAYLOAD_NO_ARTIFACT,
    STATUS_UPDATE_PAYLOAD,
    REAL_WEBHOOK_PAYLOAD,
    REAL_WEBHOOK_TRANSCRIPT_ARRAY,
    REAL_WEBHOOK_WITH_ARTIFACT,
)


class TestIsEndOfCallReport:
    """Tests for is_end_of_call_report type guard."""

    def test_returns_true_for_valid_end_of_call_report(self):
        """Should return True for valid end-of-call-report."""
        assert is_end_of_call_report(SAMPLE_PAYLOAD) is True

    def test_returns_false_for_other_webhook_types(self):
        """Should return False for other webhook types."""
        assert is_end_of_call_report(STATUS_UPDATE_PAYLOAD) is False

    def test_returns_false_when_call_is_missing(self):
        """Should return False when call is missing."""
        payload = {"type": "end-of-call-report"}
        assert is_end_of_call_report(payload) is False


class TestProcessVapiEndOfCallReport:
    """Tests for process_vapi_end_of_call_report function."""

    @pytest.fixture
    def mock_tracer(self):
        """Create a mock tracer and span."""
        mock_span = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.start_span.return_value = mock_span

        with patch("moda.vapi.spans.trace") as mock_trace:
            mock_trace.get_tracer.return_value = mock_tracer
            mock_trace.set_span_in_context.return_value = MagicMock()
            yield {
                "trace": mock_trace,
                "tracer": mock_tracer,
                "span": mock_span,
            }

    def test_creates_spans_for_valid_end_of_call_report(self, mock_tracer):
        """Should create spans for valid end-of-call report."""
        process_vapi_end_of_call_report(SAMPLE_PAYLOAD)

        # Should create parent span
        mock_tracer["tracer"].start_span.assert_any_call(
            "vapi.call",
            kind=pytest.importorskip("opentelemetry.trace").SpanKind.INTERNAL,
            attributes={"llm.vendor": "vapi"},
        )

        # Should set conversation ID
        mock_tracer["span"].set_attribute.assert_any_call(
            "moda.conversation_id", "call_abc123"
        )

        # Should set user ID from customer number
        mock_tracer["span"].set_attribute.assert_any_call(
            "moda.user_id", "+1234567890"
        )

        # Should set call attributes
        mock_tracer["span"].set_attribute.assert_any_call("vapi.call.duration", 120)
        mock_tracer["span"].set_attribute.assert_any_call(
            "vapi.call.ended_reason", "customer-ended-call"
        )
        mock_tracer["span"].set_attribute.assert_any_call("vapi.call.cost", 0.25)

        # Should end the span
        mock_tracer["span"].set_status.assert_called()
        mock_tracer["span"].end.assert_called()

    def test_does_not_create_spans_for_non_end_of_call_webhooks(self, mock_tracer):
        """Should not create spans for non-end-of-call webhooks."""
        process_vapi_end_of_call_report(STATUS_UPDATE_PAYLOAD)

        mock_tracer["tracer"].start_span.assert_not_called()

    def test_uses_custom_conversation_id_when_provided(self, mock_tracer):
        """Should use custom conversation_id when provided."""
        process_vapi_end_of_call_report(
            SAMPLE_PAYLOAD,
            {"conversation_id": "custom_conv_123"},
        )

        mock_tracer["span"].set_attribute.assert_any_call(
            "moda.conversation_id", "custom_conv_123"
        )

    def test_uses_custom_user_id_when_provided(self, mock_tracer):
        """Should use custom user_id when provided."""
        process_vapi_end_of_call_report(
            SAMPLE_PAYLOAD,
            {"user_id": "user_456"},
        )

        mock_tracer["span"].set_attribute.assert_any_call("moda.user_id", "user_456")

    def test_handles_missing_artifact_gracefully(self, mock_tracer):
        """Should handle missing artifact gracefully."""
        # Should not throw
        process_vapi_end_of_call_report(PAYLOAD_NO_ARTIFACT)

        # Should still create parent span
        mock_tracer["tracer"].start_span.assert_any_call(
            "vapi.call",
            kind=pytest.importorskip("opentelemetry.trace").SpanKind.INTERNAL,
            attributes={"llm.vendor": "vapi"},
        )

    def test_handles_missing_customer_gracefully(self, mock_tracer):
        """Should handle missing customer gracefully."""
        process_vapi_end_of_call_report(MINIMAL_PAYLOAD)

        # Should not set user_id if customer not present
        user_id_calls = [
            call
            for call in mock_tracer["span"].set_attribute.call_args_list
            if call[0][0] == "moda.user_id"
        ]
        assert len(user_id_calls) == 0

    def test_handles_missing_performance_metrics_gracefully(self, mock_tracer):
        """Should handle missing performance metrics gracefully."""
        payload = {
            "type": "end-of-call-report",
            "call": {
                "id": "call_no_metrics",
                "artifact": {
                    "messages": [
                        {"role": "user", "content": "Hello"},
                        {"role": "assistant", "content": "Hi!"},
                    ],
                },
            },
        }

        # Should not throw
        process_vapi_end_of_call_report(payload)

    def test_sets_cost_breakdown_attributes(self, mock_tracer):
        """Should set cost breakdown attributes."""
        process_vapi_end_of_call_report(SAMPLE_PAYLOAD)

        mock_tracer["span"].set_attribute.assert_any_call("vapi.cost.model", 0.15)
        mock_tracer["span"].set_attribute.assert_any_call("vapi.cost.transcriber", 0.05)
        mock_tracer["span"].set_attribute.assert_any_call("vapi.cost.voice", 0.05)


class TestVapiSpanCreation:
    """Tests for span creation hierarchy."""

    @pytest.fixture
    def mock_tracer_with_span_tracking(self):
        """Create a mock tracer that tracks created spans."""
        created_spans = []

        def create_mock_span(name, **kwargs):
            span = MagicMock()
            span.name = name
            span.set_attribute = MagicMock()
            span.set_status = MagicMock()
            span.end = MagicMock()
            created_spans.append(span)
            return span

        mock_tracer = MagicMock()
        mock_tracer.start_span.side_effect = create_mock_span

        with patch("moda.vapi.spans.trace") as mock_trace:
            mock_trace.get_tracer.return_value = mock_tracer
            mock_trace.set_span_in_context.return_value = MagicMock()
            yield {
                "trace": mock_trace,
                "tracer": mock_tracer,
                "spans": created_spans,
            }

    def test_creates_turn_spans_with_correct_names(self, mock_tracer_with_span_tracking):
        """Should create turn spans with correct names."""
        process_vapi_end_of_call_report(SAMPLE_PAYLOAD)

        span_names = [s.name for s in mock_tracer_with_span_tracking["spans"]]
        turn_span_names = [n for n in span_names if n.startswith("vapi.turn.")]

        assert "vapi.turn.0" in turn_span_names
        assert "vapi.turn.1" in turn_span_names
        assert "vapi.turn.2" in turn_span_names
        assert "vapi.turn.3" in turn_span_names

    def test_creates_tool_spans_with_function_names(self, mock_tracer_with_span_tracking):
        """Should create tool spans with function names."""
        process_vapi_end_of_call_report(SAMPLE_PAYLOAD)

        span_names = [s.name for s in mock_tracer_with_span_tracking["spans"]]
        tool_span_names = [n for n in span_names if n.startswith("vapi.tool.")]

        assert "vapi.tool.lookupOrder" in tool_span_names

    def test_creates_transfer_spans(self, mock_tracer_with_span_tracking):
        """Should create transfer spans."""
        process_vapi_end_of_call_report(SAMPLE_PAYLOAD)

        span_names = [s.name for s in mock_tracer_with_span_tracking["spans"]]
        transfer_span_names = [n for n in span_names if n == "vapi.transfer"]

        assert len(transfer_span_names) == 1

    def test_ends_all_created_spans(self, mock_tracer_with_span_tracking):
        """Should end all created spans."""
        process_vapi_end_of_call_report(SAMPLE_PAYLOAD)

        for span in mock_tracer_with_span_tracking["spans"]:
            span.end.assert_called()


class TestIsEndOfCallReportRealFormat:
    """Tests for is_end_of_call_report with real VAPI format."""

    def test_returns_true_for_message_wrapped_payload(self):
        """Should return True for message-wrapped end-of-call-report."""
        assert is_end_of_call_report(REAL_WEBHOOK_PAYLOAD) is True

    def test_returns_false_for_message_wrapped_non_end_of_call(self):
        """Should return False for message-wrapped non-end-of-call type."""
        payload = {
            "message": {
                "type": "status-update",
                "call": {"id": "test"},
            },
        }
        assert is_end_of_call_report(payload) is False

    def test_returns_false_for_message_without_call(self):
        """Should return False when message has no call data."""
        payload = {
            "message": {
                "type": "end-of-call-report",
            },
        }
        assert is_end_of_call_report(payload) is False


class TestProcessRealWebhookPayload:
    """Tests for processing real VAPI webhook payloads."""

    @pytest.fixture
    def mock_tracer_with_span_tracking(self):
        """Create a mock tracer that tracks created spans."""
        created_spans = []

        def create_mock_span(name, **kwargs):
            span = MagicMock()
            span.name = name
            span.set_attribute = MagicMock()
            span.set_status = MagicMock()
            span.end = MagicMock()
            created_spans.append(span)
            return span

        mock_tracer = MagicMock()
        mock_tracer.start_span.side_effect = create_mock_span

        with patch("moda.vapi.spans.trace") as mock_trace:
            mock_trace.get_tracer.return_value = mock_tracer
            mock_trace.set_span_in_context.return_value = MagicMock()
            yield {
                "trace": mock_trace,
                "tracer": mock_tracer,
                "spans": created_spans,
            }

    def test_creates_spans_from_real_webhook_payload(self, mock_tracer_with_span_tracking):
        """Should create spans from a real message-wrapped payload."""
        process_vapi_end_of_call_report(REAL_WEBHOOK_PAYLOAD)

        span_names = [s.name for s in mock_tracer_with_span_tracking["spans"]]
        assert "vapi.call" in span_names

    def test_creates_turn_spans_from_transcript_array(self, mock_tracer_with_span_tracking):
        """Should create turn spans from transcript array with {role, message} entries."""
        process_vapi_end_of_call_report(REAL_WEBHOOK_TRANSCRIPT_ARRAY)

        span_names = [s.name for s in mock_tracer_with_span_tracking["spans"]]
        turn_spans = [n for n in span_names if n.startswith("vapi.turn.")]

        # 2 assistant messages in the transcript -> 2 turn spans
        assert len(turn_spans) == 2
        assert "vapi.turn.0" in turn_spans
        assert "vapi.turn.1" in turn_spans

    def test_sets_analysis_attributes_on_parent_span(self, mock_tracer_with_span_tracking):
        """Should set analysis attributes on the parent call span."""
        process_vapi_end_of_call_report(REAL_WEBHOOK_PAYLOAD)

        parent_span = mock_tracer_with_span_tracking["spans"][0]

        parent_span.set_attribute.assert_any_call(
            "vapi.analysis.summary",
            "Customer asked about billing. Issue resolved.",
        )
        parent_span.set_attribute.assert_any_call(
            "vapi.analysis.success_evaluation", "true"
        )

    def test_sets_structured_data_as_json_string(self, mock_tracer_with_span_tracking):
        """Should set structured data as a JSON string attribute."""
        process_vapi_end_of_call_report(REAL_WEBHOOK_PAYLOAD)

        parent_span = mock_tracer_with_span_tracking["spans"][0]

        # Find the structured_data call
        structured_data_calls = [
            call
            for call in parent_span.set_attribute.call_args_list
            if call[0][0] == "vapi.analysis.structured_data"
        ]
        assert len(structured_data_calls) == 1
        import json

        data = json.loads(structured_data_calls[0][0][1])
        assert data["intent"] == "billing_inquiry"
        assert data["resolved"] is True

    def test_sets_assistant_id_attribute(self, mock_tracer_with_span_tracking):
        """Should set assistant_id attribute on parent span."""
        process_vapi_end_of_call_report(REAL_WEBHOOK_PAYLOAD)

        parent_span = mock_tracer_with_span_tracking["spans"][0]
        parent_span.set_attribute.assert_any_call(
            "vapi.call.assistant_id", "asst_456"
        )

    def test_sets_call_status_attribute(self, mock_tracer_with_span_tracking):
        """Should set call status attribute on parent span."""
        process_vapi_end_of_call_report(REAL_WEBHOOK_PAYLOAD)

        parent_span = mock_tracer_with_span_tracking["spans"][0]
        parent_span.set_attribute.assert_any_call("vapi.call.status", "ended")

    def test_sets_timing_attributes(self, mock_tracer_with_span_tracking):
        """Should set started_at and ended_at attributes."""
        process_vapi_end_of_call_report(REAL_WEBHOOK_PAYLOAD)

        parent_span = mock_tracer_with_span_tracking["spans"][0]
        parent_span.set_attribute.assert_any_call(
            "vapi.call.started_at", "2024-01-15T10:30:00Z"
        )
        parent_span.set_attribute.assert_any_call(
            "vapi.call.ended_at", "2024-01-15T10:32:35Z"
        )

    def test_prefers_artifact_messages_over_transcript(self, mock_tracer_with_span_tracking):
        """Should use artifact messages when both artifact and transcript exist."""
        process_vapi_end_of_call_report(REAL_WEBHOOK_WITH_ARTIFACT)

        span_names = [s.name for s in mock_tracer_with_span_tracking["spans"]]
        turn_spans = [n for n in span_names if n.startswith("vapi.turn.")]
        tool_spans = [n for n in span_names if n.startswith("vapi.tool.")]

        # Should have 3 assistant turns from artifact (not 1 from transcript)
        assert len(turn_spans) == 3
        # Should have tool call from artifact
        assert "vapi.tool.checkAvailability" in tool_spans

    def test_backward_compatible_with_legacy_format(self, mock_tracer_with_span_tracking):
        """Should still work with the legacy payload format."""
        process_vapi_end_of_call_report(SAMPLE_PAYLOAD)

        span_names = [s.name for s in mock_tracer_with_span_tracking["spans"]]
        assert "vapi.call" in span_names
        turn_spans = [n for n in span_names if n.startswith("vapi.turn.")]
        assert len(turn_spans) == 4
