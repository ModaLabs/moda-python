"""Regression tests for OTel GenAI semconv array-shape attributes.

The OTel GenAI semconv declares ``gen_ai.request.stop_sequences`` and
``gen_ai.response.finish_reasons`` as ``string[]``. OTel Python's
``Span.set_attribute`` accepts sequences of primitives directly — encoding
the values as JSON strings would land in OTLP as ``string_value`` instead
of ``array_value`` and break spec-aware consumers.

These tests pin the array shape against a future regression and exercise
both the sync and async response paths in span_utils.
"""

import asyncio
from unittest.mock import MagicMock

import pytest


def _captured_attrs(mock_span: MagicMock) -> dict:
    """Collapse all positional set_attribute calls into a dict."""
    out: dict = {}
    for call in mock_span.set_attribute.call_args_list:
        if len(call.args) >= 2:
            out[call.args[0]] = call.args[1]
    return out


@pytest.mark.asyncio
async def test_input_attributes_stop_sequences_is_list_not_json_string():
    from opentelemetry.instrumentation.anthropic.span_utils import aset_input_attributes

    span = MagicMock()
    await aset_input_attributes(span, {"stop_sequences": ["END", "###"]})

    attrs = _captured_attrs(span)
    value = attrs.get("gen_ai.request.stop_sequences")
    assert value == ["END", "###"], "stop_sequences must be a list, not a JSON string"
    assert isinstance(value, list)
    assert not isinstance(value, str)


@pytest.mark.asyncio
async def test_input_attributes_skips_stop_sequences_when_missing():
    from opentelemetry.instrumentation.anthropic.span_utils import aset_input_attributes

    span = MagicMock()
    await aset_input_attributes(span, {})  # no stop_sequences

    attrs = _captured_attrs(span)
    assert "gen_ai.request.stop_sequences" not in attrs


@pytest.mark.asyncio
async def test_input_attributes_filters_non_string_stop_sequences():
    from opentelemetry.instrumentation.anthropic.span_utils import aset_input_attributes

    span = MagicMock()
    await aset_input_attributes(
        span, {"stop_sequences": ["ok", 5, None, "also-ok"]}
    )

    attrs = _captured_attrs(span)
    assert attrs.get("gen_ai.request.stop_sequences") == ["ok", "also-ok"]


@pytest.mark.asyncio
async def test_input_attributes_skips_when_all_stop_sequences_invalid():
    from opentelemetry.instrumentation.anthropic.span_utils import aset_input_attributes

    span = MagicMock()
    await aset_input_attributes(span, {"stop_sequences": [1, None]})

    attrs = _captured_attrs(span)
    assert "gen_ai.request.stop_sequences" not in attrs


@pytest.mark.asyncio
async def test_async_response_attributes_finish_reasons_is_one_element_list():
    from opentelemetry.instrumentation.anthropic import span_utils

    # Patch _aextract_response_data so we don't have to construct a real
    # Anthropic Message object — we only care about the finish_reasons path.
    async def fake_extract(response):
        return response

    original = span_utils._aextract_response_data if hasattr(
        span_utils, "_aextract_response_data"
    ) else None
    # Function is imported inside aset_response_attributes via
    # `from .utils import _aextract_response_data`, so patch the source.
    from opentelemetry.instrumentation.anthropic import utils as anthropic_utils
    saved = anthropic_utils._aextract_response_data
    anthropic_utils._aextract_response_data = fake_extract
    try:
        span = MagicMock()
        await span_utils.aset_response_attributes(
            span,
            {"model": "claude-x", "id": "msg_1", "stop_reason": "end_turn"},
        )
    finally:
        anthropic_utils._aextract_response_data = saved

    attrs = _captured_attrs(span)
    value = attrs.get("gen_ai.response.finish_reasons")
    assert value == ["end_turn"], "finish_reasons must be a list, not a JSON string"
    assert isinstance(value, list)
    assert not isinstance(value, str)


def test_sync_response_attributes_finish_reasons_is_one_element_list():
    from opentelemetry.instrumentation.anthropic import span_utils

    # Patch _extract_response_data symmetric to the async test above.
    from opentelemetry.instrumentation.anthropic import utils as anthropic_utils
    saved = anthropic_utils._extract_response_data
    anthropic_utils._extract_response_data = lambda r: r
    try:
        span = MagicMock()
        span_utils.set_response_attributes(
            span,
            {
                "model": "claude-x",
                "id": "msg_1",
                "stop_reason": "max_tokens",
                "content": [],
            },
        )
    finally:
        anthropic_utils._extract_response_data = saved

    attrs = _captured_attrs(span)
    value = attrs.get("gen_ai.response.finish_reasons")
    assert value == ["max_tokens"]
    assert isinstance(value, list)
    assert not isinstance(value, str)


@pytest.mark.asyncio
async def test_async_response_attributes_skips_finish_reasons_when_missing():
    from opentelemetry.instrumentation.anthropic import span_utils
    from opentelemetry.instrumentation.anthropic import utils as anthropic_utils

    saved = anthropic_utils._aextract_response_data

    async def fake_extract(response):
        return response

    anthropic_utils._aextract_response_data = fake_extract
    try:
        span = MagicMock()
        await span_utils.aset_response_attributes(
            span, {"model": "claude-x", "id": "msg_1"}
        )
    finally:
        anthropic_utils._aextract_response_data = saved

    attrs = _captured_attrs(span)
    assert "gen_ai.response.finish_reasons" not in attrs
