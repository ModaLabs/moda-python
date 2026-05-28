"""Regression tests for OTel GenAI semconv array-shape attributes (OpenAI).

The OTel GenAI semconv declares ``gen_ai.request.stop_sequences`` as
``string[]``. OTel Python's ``Span.set_attribute`` accepts sequences of
primitives directly — encoding the value as a JSON string would land in
OTLP as ``string_value`` instead of ``array_value`` and break spec-aware
consumers.

These tests pin the array shape against a future regression in the OpenAI
shared `_set_request_attributes` path.
"""

from unittest.mock import MagicMock


def _captured_attrs(mock_span: MagicMock) -> dict:
    """Collapse all positional set_attribute calls into a dict."""
    out: dict = {}
    for call in mock_span.set_attribute.call_args_list:
        if len(call.args) >= 2:
            out[call.args[0]] = call.args[1]
    return out


def _make_span() -> MagicMock:
    span = MagicMock()
    span.is_recording.return_value = True
    return span


def test_request_attributes_stop_sequences_list_input_stays_a_list():
    from opentelemetry.instrumentation.openai.shared import _set_request_attributes

    span = _make_span()
    _set_request_attributes(span, {"model": "gpt-4o", "stop": ["END", "###"]})

    attrs = _captured_attrs(span)
    value = attrs.get("gen_ai.request.stop_sequences")
    assert value == ["END", "###"]
    assert isinstance(value, list)
    assert not isinstance(value, str)


def test_request_attributes_stop_sequences_single_string_is_wrapped():
    from opentelemetry.instrumentation.openai.shared import _set_request_attributes

    span = _make_span()
    _set_request_attributes(span, {"model": "gpt-4o", "stop": "END"})

    attrs = _captured_attrs(span)
    value = attrs.get("gen_ai.request.stop_sequences")
    assert value == ["END"]
    assert isinstance(value, list)


def test_request_attributes_stop_sequences_filters_non_strings():
    from opentelemetry.instrumentation.openai.shared import _set_request_attributes

    span = _make_span()
    _set_request_attributes(
        span, {"model": "gpt-4o", "stop": ["ok", 5, None, "also-ok"]}
    )

    attrs = _captured_attrs(span)
    assert attrs.get("gen_ai.request.stop_sequences") == ["ok", "also-ok"]


def test_request_attributes_skips_stop_sequences_when_all_invalid():
    from opentelemetry.instrumentation.openai.shared import _set_request_attributes

    span = _make_span()
    _set_request_attributes(span, {"model": "gpt-4o", "stop": [1, None]})

    attrs = _captured_attrs(span)
    assert "gen_ai.request.stop_sequences" not in attrs


def test_request_attributes_skips_stop_sequences_when_missing():
    from opentelemetry.instrumentation.openai.shared import _set_request_attributes

    span = _make_span()
    _set_request_attributes(span, {"model": "gpt-4o"})

    attrs = _captured_attrs(span)
    assert "gen_ai.request.stop_sequences" not in attrs
