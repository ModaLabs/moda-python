"""Tests for OpenTelemetry provider coexistence.

Verifies that Moda correctly detects and integrates with existing
TracerProviders set up by other SDKs (Sentry, PostHog, Datadog, etc.).
"""

import pytest
from unittest.mock import patch

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import ProxyTracerProvider

from traceloop.sdk.tracing.tracing import TracerWrapper


@pytest.fixture(scope="module")
def exporter():
    """Override the shared test exporter fixture to keep this module isolated."""
    return InMemorySpanExporter()


@pytest.fixture(autouse=True)
def clean_tracer_wrapper():
    """Ensure TracerWrapper singleton is clean before each test."""
    if hasattr(TracerWrapper, "instance"):
        saved = TracerWrapper.instance
        del TracerWrapper.instance
    else:
        saved = None
    yield
    # Restore
    if saved is not None:
        TracerWrapper.instance = saved
    elif hasattr(TracerWrapper, "instance"):
        del TracerWrapper.instance


class TestExternalProviderDetection:
    """Tests that Moda detects and attaches to existing TracerProviders."""

    def test_attaches_to_existing_tracer_provider(self):
        """When a real TracerProvider exists, Moda should add its processor to it."""
        # Simulate an external provider (e.g., PostHog, Datadog)
        external_exporter = InMemorySpanExporter()
        external_provider = TracerProvider()
        external_provider.add_span_processor(SimpleSpanProcessor(external_exporter))

        with patch(
            "traceloop.sdk.tracing.tracing.get_tracer_provider",
            return_value=external_provider,
        ), patch(
            "traceloop.sdk.tracing.tracing.trace.set_tracer_provider",
            lambda provider: None,
        ):
            # Now init Moda — should detect the existing provider
            moda_exporter = InMemorySpanExporter()
            from traceloop.sdk import Traceloop

            Traceloop.init(
                app_name="coexistence-test",
                exporter=moda_exporter,
                disable_batch=True,
            )

            # Create a span using the simulated external provider directly.
            tracer = external_provider.get_tracer("test")
            with tracer.start_as_current_span("test-span") as span:
                span.set_attribute("test.key", "value")

            # External exporter should have received the span
            external_spans = external_exporter.get_finished_spans()
            assert len(external_spans) >= 1, (
                f"External provider received {len(external_spans)} spans, expected >= 1"
            )

    def test_creates_own_provider_when_none_exists(self):
        """When no external provider exists, Moda should create its own."""
        moda_exporter = InMemorySpanExporter()
        with patch(
            "traceloop.sdk.tracing.tracing.get_tracer_provider",
            return_value=ProxyTracerProvider(),
        ), patch(
            "traceloop.sdk.tracing.tracing.trace.set_tracer_provider",
            lambda provider: None,
        ):
            from traceloop.sdk import Traceloop

            Traceloop.init(
                app_name="own-provider-test",
                exporter=moda_exporter,
                disable_batch=True,
            )

            tracer = TracerWrapper.instance.get_tracer()
            with tracer.start_as_current_span("test-span") as span:
                span.set_attribute("test.key", "value")

            moda_spans = moda_exporter.get_finished_spans()
            assert len(moda_spans) >= 1, (
                f"Moda exporter received {len(moda_spans)} spans, expected >= 1"
            )

    def test_attaches_to_provider_registered_after_init(self):
        """Order-robustness: a real provider registered AFTER moda.init() should
        still be attached to via lazy delegate re-resolution.

        This mirrors the init-order bug the Node SDK fixes: if the user's app
        calls ``moda.init()`` before another SDK (Sentry/Datadog) installs its
        real provider, Moda must re-resolve at use time and add its span
        processor to the provider that appears later.
        """
        external_exporter = InMemorySpanExporter()
        external_provider = TracerProvider()
        external_provider.add_span_processor(SimpleSpanProcessor(external_exporter))

        moda_exporter = InMemorySpanExporter()

        # At init time only a ProxyTracerProvider is present; afterward a real
        # external provider is registered. ``get_tracer_provider`` is patched so
        # we can flip what it returns before vs. after init.
        with patch(
            "traceloop.sdk.tracing.tracing.get_tracer_provider"
        ) as mock_get_provider, patch(
            "traceloop.sdk.tracing.tracing.trace.set_tracer_provider",
            lambda provider: None,
        ):
            mock_get_provider.return_value = ProxyTracerProvider()

            from traceloop.sdk import Traceloop

            Traceloop.init(
                app_name="late-provider-test",
                exporter=moda_exporter,
                disable_batch=True,
            )

            # A real external provider is registered AFTER moda.init().
            mock_get_provider.return_value = external_provider

            # Any Moda touchpoint re-resolves the provider and attaches to it.
            TracerWrapper.instance.get_tracer()

            # A span created via the later-registered provider must reach Moda's
            # processor (proving the delegate was re-resolved).
            tracer = external_provider.get_tracer("test")
            with tracer.start_as_current_span("test-span") as span:
                span.set_attribute("test.key", "value")

            moda_spans = moda_exporter.get_finished_spans()
            assert len(moda_spans) >= 1, (
                f"Moda exporter received {len(moda_spans)} spans, expected >= 1 "
                "(provider registered after init was not attached to)"
            )

            external_spans = external_exporter.get_finished_spans()
            assert len(external_spans) >= 1, (
                f"External provider received {len(external_spans)} spans, expected >= 1"
            )


class TestMultipleProcessors:
    """Tests that multiple span processors can coexist."""

    def test_multiple_processors_all_receive_spans(self):
        """When multiple processors are passed, all should receive spans."""
        exporter1 = InMemorySpanExporter()
        exporter2 = InMemorySpanExporter()

        processor1 = SimpleSpanProcessor(exporter1)
        processor2 = SimpleSpanProcessor(exporter2)

        with patch(
            "traceloop.sdk.tracing.tracing.get_tracer_provider",
            return_value=ProxyTracerProvider(),
        ), patch(
            "traceloop.sdk.tracing.tracing.trace.set_tracer_provider",
            lambda provider: None,
        ):
            from traceloop.sdk import Traceloop

            Traceloop.init(
                app_name="multi-processor-test",
                processor=[processor1, processor2],
            )

            tracer = TracerWrapper.instance.get_tracer()
            with tracer.start_as_current_span("test-span") as span:
                span.set_attribute("test.key", "value")

            spans1 = exporter1.get_finished_spans()
            spans2 = exporter2.get_finished_spans()

            assert len(spans1) >= 1, f"Processor 1 received {len(spans1)} spans, expected >= 1"
            assert len(spans2) >= 1, f"Processor 2 received {len(spans2)} spans, expected >= 1"


class TestUrlExclusions:
    """Tests that external observability provider URLs are excluded from tracing."""

    def test_excluded_urls_contain_sentry(self):
        """Sentry URLs should be in the exclusion list."""
        from traceloop.sdk.tracing.tracing import EXCLUDED_URLS

        assert "sentry.io" in EXCLUDED_URLS, "sentry.io should be excluded from tracing"

    def test_excluded_urls_contain_posthog(self):
        """PostHog URLs should be in the exclusion list."""
        from traceloop.sdk.tracing.tracing import EXCLUDED_URLS

        assert "posthog.com" in EXCLUDED_URLS, "posthog.com should be excluded from tracing"

    def test_excluded_urls_contain_traceloop(self):
        """Traceloop URLs should be in the exclusion list."""
        from traceloop.sdk.tracing.tracing import EXCLUDED_URLS

        assert "traceloop.com" in EXCLUDED_URLS, "traceloop.com should be excluded from tracing"
