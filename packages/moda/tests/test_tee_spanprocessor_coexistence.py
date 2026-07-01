"""Coexistence tests for the tee-a-SpanProcessor / never-a-second-global rule.

MODA-544 (Gate 2 prerequisite). Parallel to the moda-node tee-in fix. The
LOCKED rule: Moda tees a SpanProcessor into the host TracerProvider and never
registers a second global provider when a real host provider already exists.

These tests cover:
  * Both init orders (Moda-first AND host-first) landing on a single global.
  * All four silent modes:
      (1) init-order race (Moda before host)         -> re-check, not snapshot
      (2) env-only OTLP (bare proxy at init)          -> Moda owns single global
      (3) non-global / out-of-band host provider      -> tee, don't clobber
      (4) snapshot-once                                -> attachment re-evaluated
  * moda.environment stamped at SPAN level, always, on the tee-in path.
  * Loud-fail contract: an unsafe tee-in raises to the caller.

We exercise the tracing internals directly (init_tracer_provider,
resolve_host_provider, the on_start stamp, TracerWrapper re-attach) rather than
the full Traceloop.init() path, because OpenTelemetry's global _TRACER_PROVIDER
is SET_ONCE per process and cannot be re-registered across tests. Each test
patches get_tracer_provider / set_tracer_provider to model the global.
"""

import os
from unittest.mock import patch

import pytest

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import ProxyTracerProvider

import traceloop.sdk.tracing.tracing as tracing_mod
from traceloop.sdk.tracing.tracing import (
    DEPLOYMENT_ENVIRONMENT_ATTRIBUTE,
    MODA_ENVIRONMENT_ATTRIBUTE,
    TracerProviderCoexistenceError,
    TracerWrapper,
    default_span_processor_on_start,
    init_tracer_provider,
    resolve_host_provider,
    resolve_moda_environment,
)


@pytest.fixture(autouse=True)
def _clean_tracer_wrapper():
    """Isolate the TracerWrapper singleton and static params for each test."""
    saved_instance = getattr(TracerWrapper, "instance", None)
    if saved_instance is not None:
        del TracerWrapper.instance
    saved_attrs = TracerWrapper.resource_attributes
    TracerWrapper.resource_attributes = {}
    saved_env = {
        k: os.environ.get(k)
        for k in ("MODA_ENVIRONMENT", "TRACELOOP_ENVIRONMENT")
    }
    for k in saved_env:
        os.environ.pop(k, None)

    yield

    TracerWrapper.resource_attributes = saved_attrs
    for k, v in saved_env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    if getattr(TracerWrapper, "instance", None) is not None:
        del TracerWrapper.instance
    if saved_instance is not None:
        TracerWrapper.instance = saved_instance


def _resource() -> Resource:
    return Resource.create({"service.name": "moda-544-test"})


# ---------------------------------------------------------------------------
# init_tracer_provider: single-global rule + both orders
# ---------------------------------------------------------------------------


class TestSingleGlobalProvider:
    def test_host_first_reuses_host_provider_no_second_global(self):
        """Host registered first: Moda must reuse it, never set a 2nd global."""
        host = TracerProvider()
        set_calls = []

        with patch.object(tracing_mod, "get_tracer_provider", return_value=host), patch.object(
            tracing_mod.trace,
            "set_tracer_provider",
            side_effect=lambda p: set_calls.append(p),
        ):
            provider = init_tracer_provider(resource=_resource())

        assert provider is host, "Moda should reuse the existing host provider"
        assert set_calls == [], "Moda must NOT register a second global provider"

    def test_moda_first_creates_single_global(self):
        """No host yet (bare proxy): Moda creates and registers exactly one."""
        set_calls = []

        with patch.object(
            tracing_mod, "get_tracer_provider", return_value=ProxyTracerProvider()
        ), patch.object(
            tracing_mod.trace,
            "set_tracer_provider",
            side_effect=lambda p: set_calls.append(p),
        ):
            provider = init_tracer_provider(resource=_resource())

        assert isinstance(provider, TracerProvider)
        assert not isinstance(provider, ProxyTracerProvider)
        assert set_calls == [provider], "Moda registers exactly one global provider"


# ---------------------------------------------------------------------------
# Silent mode (2): env-only OTLP -> bare proxy at init
# ---------------------------------------------------------------------------


class TestEnvOnlyOtlpMode:
    def test_bare_proxy_treated_as_no_host(self):
        with patch.object(
            tracing_mod, "get_tracer_provider", return_value=ProxyTracerProvider()
        ):
            assert resolve_host_provider() is None

    def test_moda_owns_global_when_only_env_configured(self):
        set_calls = []
        with patch.object(
            tracing_mod, "get_tracer_provider", return_value=ProxyTracerProvider()
        ), patch.object(
            tracing_mod.trace,
            "set_tracer_provider",
            side_effect=lambda p: set_calls.append(p),
        ):
            provider = init_tracer_provider(resource=_resource())
        assert len(set_calls) == 1 and set_calls[0] is provider


# ---------------------------------------------------------------------------
# Silent mode (3): non-global / out-of-band real host provider
# ---------------------------------------------------------------------------


class TestNonGlobalProvider:
    def test_real_host_detected_and_not_clobbered(self):
        host = TracerProvider()
        with patch.object(tracing_mod, "get_tracer_provider", return_value=host):
            assert resolve_host_provider() is host

    def test_proxy_wrapping_real_delegate_is_unwrapped(self):
        """Sentry-style proxy that exposes a real _delegate should unwrap."""
        real = TracerProvider()

        class FakeProxy(ProxyTracerProvider):
            _delegate = real

        with patch.object(
            tracing_mod, "get_tracer_provider", return_value=FakeProxy()
        ):
            assert resolve_host_provider() is real


# ---------------------------------------------------------------------------
# Silent mode (1) + (4): init-order race, re-check (not snapshot-once)
# ---------------------------------------------------------------------------


class TestInitOrderRaceRecheck:
    def test_moda_processor_reattaches_when_host_appears_later(self):
        """Moda inits before the host. When the host provider registers later,
        the first span must trigger a re-check that tees Moda's processors onto
        the host provider (single global, no second set_tracer_provider)."""
        moda_exporter = InMemorySpanExporter()
        moda_processor = SimpleSpanProcessor(moda_exporter)

        # Model: Moda created its own provider at init.
        moda_provider = TracerProvider()

        # Build a minimal TracerWrapper-like object exercising the re-attach.
        obj = object.__new__(TracerWrapper)
        obj._TracerWrapper__moda_processors = [moda_processor]
        obj._TracerWrapper__attached_provider = moda_provider
        obj._TracerWrapper__reattach_checked = False

        host = TracerProvider()
        host_exporter = InMemorySpanExporter()
        host.add_span_processor(SimpleSpanProcessor(host_exporter))

        # Host provider registers AFTER Moda init.
        with patch.object(tracing_mod, "get_tracer_provider", return_value=host):
            obj._reattach_to_host_provider_if_changed()

        assert obj._TracerWrapper__attached_provider is host, (
            "Moda must re-attach to the late-registered host provider"
        )

        # A host-created span should now reach Moda's exporter (tee'd in).
        tracer = host.get_tracer("test")
        with tracer.start_as_current_span("host-span"):
            pass

        assert len(moda_exporter.get_finished_spans()) >= 1, (
            "Moda processor should receive host spans after re-attach"
        )

    def test_recheck_runs_at_most_once(self):
        obj = object.__new__(TracerWrapper)
        obj._TracerWrapper__moda_processors = []
        moda_provider = TracerProvider()
        obj._TracerWrapper__attached_provider = moda_provider
        obj._TracerWrapper__reattach_checked = False

        calls = []

        def _spy():
            calls.append(1)
            return None

        with patch.object(tracing_mod, "resolve_host_provider", side_effect=_spy):
            obj._reattach_to_host_provider_if_changed()
            obj._reattach_to_host_provider_if_changed()

        assert len(calls) == 1, "Re-check must run at most once"

    def test_no_reattach_when_no_real_host_appears(self):
        obj = object.__new__(TracerWrapper)
        obj._TracerWrapper__moda_processors = []
        moda_provider = TracerProvider()
        obj._TracerWrapper__attached_provider = moda_provider
        obj._TracerWrapper__reattach_checked = False

        with patch.object(
            tracing_mod, "get_tracer_provider", return_value=ProxyTracerProvider()
        ):
            obj._reattach_to_host_provider_if_changed()

        assert obj._TracerWrapper__attached_provider is moda_provider


# ---------------------------------------------------------------------------
# Loud-fail contract
# ---------------------------------------------------------------------------


class TestLoudFailContract:
    def test_real_provider_without_add_span_processor_raises(self):
        """A real, non-proxy global that lacks add_span_processor must raise a
        loud error to the caller, not logging.error + silent return."""

        class ReadOnlyProvider:  # no add_span_processor, not a proxy
            def get_tracer(self, *a, **k):
                raise NotImplementedError

        with patch.object(
            tracing_mod, "get_tracer_provider", return_value=ReadOnlyProvider()
        ):
            with pytest.raises(TracerProviderCoexistenceError):
                init_tracer_provider(resource=_resource())

    def test_loud_fail_does_not_leave_half_built_singleton(self):
        """When init_tracer_provider raises, TracerWrapper must not leave a
        broken singleton behind."""

        class ReadOnlyProvider:
            def get_tracer(self, *a, **k):
                raise NotImplementedError

        TracerWrapper.endpoint = "https://example.invalid/v1/traces"
        try:
            with patch.object(
                tracing_mod, "get_tracer_provider", return_value=ReadOnlyProvider()
            ):
                with pytest.raises(TracerProviderCoexistenceError):
                    TracerWrapper()
            assert not hasattr(TracerWrapper, "instance"), (
                "Failed init must not leave a half-built singleton"
            )
        finally:
            TracerWrapper.endpoint = None


# ---------------------------------------------------------------------------
# Span-level moda.environment stamp (mirror of the node fix)
# ---------------------------------------------------------------------------


class _FakeSpan:
    def __init__(self):
        self.attributes = {}

    def set_attribute(self, key, value):
        self.attributes[key] = value


class TestSpanLevelEnvironmentStamp:
    def test_stamps_from_resource_deployment_environment(self):
        TracerWrapper.resource_attributes = {
            DEPLOYMENT_ENVIRONMENT_ATTRIBUTE: "staging"
        }
        assert resolve_moda_environment() == "staging"

        span = _FakeSpan()
        default_span_processor_on_start(span, None)
        assert span.attributes.get(MODA_ENVIRONMENT_ATTRIBUTE) == "staging"

    def test_stamps_from_env_var(self):
        os.environ["MODA_ENVIRONMENT"] = "development"
        assert resolve_moda_environment() == "development"

        span = _FakeSpan()
        default_span_processor_on_start(span, None)
        assert span.attributes.get(MODA_ENVIRONMENT_ATTRIBUTE) == "development"

    def test_resource_attribute_takes_precedence_over_env(self):
        TracerWrapper.resource_attributes = {
            DEPLOYMENT_ENVIRONMENT_ATTRIBUTE: "staging"
        }
        os.environ["MODA_ENVIRONMENT"] = "development"
        assert resolve_moda_environment() == "staging"

    def test_no_stamp_and_no_fabricated_production_when_unset(self):
        """Critical: we must NOT fabricate "production". When nothing is
        configured, no span-level stamp is applied and ingest applies its own
        documented default instead of us lying about the environment."""
        assert resolve_moda_environment() is None

        span = _FakeSpan()
        default_span_processor_on_start(span, None)
        assert MODA_ENVIRONMENT_ATTRIBUTE not in span.attributes

    def test_tee_in_path_stamps_span_environment(self):
        """End-to-end on the tee-in path: a host provider whose resource carries
        NO moda environment still yields spans stamped with moda.environment,
        so ingest does not default the tee-in trace to production."""
        TracerWrapper.resource_attributes = {
            DEPLOYMENT_ENVIRONMENT_ATTRIBUTE: "staging"
        }

        host_exporter = InMemorySpanExporter()
        host = TracerProvider()  # host resource has no moda.environment
        host_processor = SimpleSpanProcessor(host_exporter)
        # Moda tees its on_start onto the processor (as TracerWrapper does).
        host_processor.on_start = default_span_processor_on_start
        host.add_span_processor(host_processor)

        tracer = host.get_tracer("test")
        with tracer.start_as_current_span("tee-in-span"):
            pass

        spans = host_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get(MODA_ENVIRONMENT_ATTRIBUTE) == "staging", (
            "tee-in spans must carry span-level moda.environment"
        )
