"""Unit tests configuration module."""

import os
import re
import pytest
from opentelemetry import trace
from traceloop.sdk import Traceloop
from traceloop.sdk.instruments import Instruments
from traceloop.sdk.tracing.tracing import TracerWrapper
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, BatchSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.context import attach, Context
from opentelemetry.sdk.trace import ReadableSpan
from traceloop.sdk.client.http import HTTPClient
from traceloop.sdk.datasets.datasets import Datasets

pytest_plugins = []


@pytest.fixture(scope="session")
def exporter():
    exporter = InMemorySpanExporter()
    Traceloop.init(
        app_name="test",
        resource_attributes={"something": "yes"},
        disable_batch=True,
        exporter=exporter,
    )
    return exporter


def _set_global_tracer_provider(provider) -> None:
    """Force-restore the OpenTelemetry global TracerProvider.

    ``trace.set_tracer_provider`` refuses to overwrite an already-set global
    (it logs "Overriding of current TracerProvider is not allowed"), so for a
    TEST-ONLY restore we write the module global directly. This is the seam the
    isolation fixture uses to put the session exporter's provider back after a
    test swapped it.
    """
    trace._TRACER_PROVIDER_SET_ONCE._done = False
    trace._TRACER_PROVIDER = provider


@pytest.fixture(autouse=True)
def _isolate_global_tracer_state(exporter):
    """Snapshot and restore BOTH the OTel global TracerProvider and the Moda
    ``TracerWrapper`` singleton around every test (TEST-ONLY, order-independence).

    Pre-existing isolation defect inherited from the WS-SDK-PY integration: the
    session-scoped ``exporter`` fixture initializes the global provider + the
    ``TracerWrapper`` singleton once, but several tests re-init the SDK on a
    fresh in-memory exporter, delete the singleton to exercise the
    "uninitialized" path, or call ``trace.set_tracer_provider`` — swapping the
    global provider the session exporter was attached to. Left un-restored, that
    leaks: downstream exporter-based tests capture 0 spans and the "fresh init"
    loud-fail tests see an already-initialized SDK.

    We depend on ``exporter`` so the session provider+singleton exist as the
    baseline, snapshot them, then on teardown put them back verbatim. This runs
    before the autouse ``clear_exporter`` (declared after it), so the session
    exporter's provider is guaranteed re-attached before the next test clears
    and uses it. Well-behaved tests that already snapshot/restore themselves are
    unaffected — the restore is idempotent.
    """
    saved_provider = trace.get_tracer_provider()
    saved_instance = getattr(TracerWrapper, "instance", None)
    try:
        yield
    finally:
        if getattr(TracerWrapper, "instance", None) is not saved_instance:
            if saved_instance is not None:
                TracerWrapper.instance = saved_instance
            elif hasattr(TracerWrapper, "instance"):
                del TracerWrapper.instance
        if trace.get_tracer_provider() is not saved_provider:
            _set_global_tracer_provider(saved_provider)


@pytest.fixture(autouse=True)
def clear_exporter(exporter, _isolate_global_tracer_state):
    exporter.clear()
    # Reset the tracing context to ensure tests don't affect each other
    # Create a new empty context and attach it
    # This effectively removes all previous context values
    attach(Context())


@pytest.fixture(autouse=True)
def environment():
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = "test_api_key"


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "filter_headers": ["authorization"],
        "ignore_hosts": ["openaipublic.blob.core.windows.net"],
    }


@pytest.fixture
def exporter_with_custom_span_processor():
    # Clear singleton if existed
    if hasattr(TracerWrapper, "instance"):
        _trace_wrapper_instance = TracerWrapper.instance
        del TracerWrapper.instance

    class CustomSpanProcessor(SimpleSpanProcessor):
        def on_start(self, span, parent_context=None):
            span.set_attribute("custom_span", "yes")

    exporter = InMemorySpanExporter()
    Traceloop.init(
        exporter=exporter,
        processor=CustomSpanProcessor(exporter),
    )

    yield exporter

    # Restore singleton if any
    if _trace_wrapper_instance:
        TracerWrapper.instance = _trace_wrapper_instance


@pytest.fixture(scope="function")
def exporter_with_custom_span_postprocess_callback(exporter):
    if hasattr(TracerWrapper, "instance"):
        _trace_wrapper_instance = TracerWrapper.instance
        del TracerWrapper.instance

    def span_postprocess_callback(span: ReadableSpan) -> None:
        prompt_pattern = re.compile(r"gen_ai\.prompt\.\d+\.content$")
        completion_pattern = re.compile(r"gen_ai\.completion\.\d+\.content$")
        if hasattr(span, "_attributes"):
            attributes = span._attributes if span._attributes else {}
            # Find and encode all matching attributes
            for key, value in attributes.items():
                if (
                    prompt_pattern.match(key) or completion_pattern.match(key)
                ) and isinstance(value, str):
                    attributes[key] = "REDACTED"  # Modify the attributes directly

    Traceloop.init(
        exporter=exporter,
        span_postprocess_callback=span_postprocess_callback,
    )

    yield exporter

    if hasattr(TracerWrapper, "instance"):
        # Get the span processor
        if hasattr(TracerWrapper.instance, "_TracerWrapper__spans_processor"):
            span_processor = TracerWrapper.instance._TracerWrapper__spans_processor
            # Reset the on_end method to its original class implementation.
            # This is needed to make this test run in isolation as SpanProcessor is a singleton.
            if isinstance(span_processor, SimpleSpanProcessor):
                span_processor.on_end = SimpleSpanProcessor.on_end.__get__(
                    span_processor, SimpleSpanProcessor
                )
            elif isinstance(span_processor, BatchSpanProcessor):
                span_processor.on_end = BatchSpanProcessor.on_end.__get__(
                    span_processor, BatchSpanProcessor
                )
    if _trace_wrapper_instance:
        TracerWrapper.instance = _trace_wrapper_instance


@pytest.fixture
def exporter_with_custom_instrumentations():
    # Clear singleton if existed
    if hasattr(TracerWrapper, "instance"):
        _trace_wrapper_instance = TracerWrapper.instance
        del TracerWrapper.instance

    exporter = InMemorySpanExporter()
    Traceloop.init(
        exporter=exporter,
        disable_batch=True,
        instruments={i for i in Instruments},
        block_instruments={Instruments.ANTHROPIC},
    )

    yield exporter

    # Restore singleton if any
    if _trace_wrapper_instance:
        TracerWrapper.instance = _trace_wrapper_instance


@pytest.fixture
def exporter_with_no_metrics():
    # Clear singleton if existed
    if hasattr(TracerWrapper, "instance"):
        _trace_wrapper_instance = TracerWrapper.instance
        del TracerWrapper.instance

    os.environ["TRACELOOP_METRICS_ENABLED"] = "false"

    exporter = InMemorySpanExporter()

    Traceloop.init(
        exporter=exporter,
        disable_batch=True,
    )

    yield exporter

    # Restore singleton if any
    if _trace_wrapper_instance:
        TracerWrapper.instance = _trace_wrapper_instance
        os.environ["TRACELOOP_METRICS_ENABLED"] = "true"


@pytest.fixture
def exporters_with_multiple_span_processors():
    # Clear singleton if existed
    if hasattr(TracerWrapper, "instance"):
        _trace_wrapper_instance = TracerWrapper.instance
        del TracerWrapper.instance

    class CustomSpanProcessor(SimpleSpanProcessor):
        def on_start(self, span, parent_context=None):
            span.set_attribute("custom_processor", "enabled")
            span.set_attribute("processor_type", "custom")

    class MetricsSpanProcessor(SimpleSpanProcessor):
        def __init__(self, exporter):
            super().__init__(exporter)
            self.span_count = 0

        def on_start(self, span, parent_context=None):
            self.span_count += 1
            span.set_attribute("metrics_processor", "enabled")
            span.set_attribute("span_count", self.span_count)

    # Create exporters for different processors
    default_exporter = InMemorySpanExporter()
    custom_exporter = InMemorySpanExporter()
    metrics_exporter = InMemorySpanExporter()

    # Get the default Traceloop processor
    default_processor = Traceloop.get_default_span_processor(
        disable_batch=True, exporter=default_exporter
    )

    # Create custom processors
    custom_processor = CustomSpanProcessor(custom_exporter)
    metrics_processor = MetricsSpanProcessor(metrics_exporter)

    # Initialize with multiple processors
    processors = [default_processor, custom_processor, metrics_processor]

    Traceloop.init(
        app_name="test_multiple_processors",
        api_endpoint="http://localhost:4318",  # Use local endpoint to avoid API key requirement
        processor=processors,
        disable_batch=True,
    )

    # Return all exporters so we can verify each processor worked
    yield {
        "default": default_exporter,
        "custom": custom_exporter,
        "metrics": metrics_exporter,
        "processor": processors,
    }

    # Restore singleton if any
    if _trace_wrapper_instance:
        TracerWrapper.instance = _trace_wrapper_instance


@pytest.fixture
def datasets():
    """Create a Datasets instance with HTTP client for VCR recording/playback"""
    api_key = os.environ.get("TRACELOOP_API_KEY", "fake-key-for-vcr-playback")
    base_url = os.environ.get("TRACELOOP_BASE_URL", "https://api-staging.traceloop.com")

    http = HTTPClient(base_url=base_url, api_key=api_key, version="1.0.0")
    return Datasets(http)


@pytest.fixture(scope="session")
def anyio_backend():
    """Force anyio to use only asyncio backend."""
    return "asyncio"
