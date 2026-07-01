"""Tests for the named `environment` init param and span-level override.

Mirrors the Node SDK contract:
  * Resource-level: `moda.init(environment=...)` stamps both `moda.environment`
    and `deployment.environment` on the resource, resolving
    explicit arg > MODA_ENVIRONMENT env var > 'production' (Node's default).
  * Span-level: `with moda.set_environment(...)` stamps `moda.environment`
    directly on spans created inside the block, winning over the resource
    default.

Note on isolation: the OpenTelemetry global TracerProvider (and thus the
resource) is set once per process, so re-initializing within the test session
cannot swap the *global* provider's resource. We therefore assert the
resolved resource attributes at their injection point
(`TracerWrapper.resource_attributes`, handed to `set_static_params`) and prove
the resource -> span wiring with a locally-constructed provider built from
those attributes. Span-level stamping is exercised through the real SDK tracer
so the on-start hook runs.
"""

import pytest
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

import moda
from traceloop.sdk import Traceloop
from traceloop.sdk.tracing.tracing import TracerWrapper


def _emit_span(name: str = "test-span"):
    """Create a span through the SDK tracer so the on-start hook runs."""
    tracer = TracerWrapper.instance.get_tracer()
    with tracer.start_as_current_span(name):
        pass


def _resource_span():
    """Build a span from a local provider using the resolved resource attrs.

    This proves the resolved `TracerWrapper.resource_attributes` flow onto a
    span's resource without depending on the process-global TracerProvider
    (which is set once and cannot be swapped mid-session).
    """
    exp = InMemorySpanExporter()
    provider = TracerProvider(
        resource=Resource.create(TracerWrapper.resource_attributes)
    )
    provider.add_span_processor(SimpleSpanProcessor(exp))
    with provider.get_tracer("test").start_as_current_span("r"):
        pass
    return exp.get_finished_spans()[0]


@pytest.fixture
def fresh_init():
    """Provide an init helper that isolates the TracerWrapper singleton.

    Each call clears the singleton and re-inits with a fresh in-memory
    exporter so environment resolution (which happens at init time) can be
    exercised independently. The original singleton is restored on teardown.
    """
    saved = getattr(TracerWrapper, "instance", None)
    if saved is not None:
        del TracerWrapper.instance

    def _init(**kwargs):
        if hasattr(TracerWrapper, "instance"):
            del TracerWrapper.instance
        exp = InMemorySpanExporter()
        Traceloop.init(
            app_name="test-env",
            disable_batch=True,
            exporter=exp,
            **kwargs,
        )
        return exp

    yield _init

    if hasattr(TracerWrapper, "instance"):
        del TracerWrapper.instance
    if saved is not None:
        TracerWrapper.instance = saved


# --------------------------------------------------------------------------- #
# Resource-level (init param)
# --------------------------------------------------------------------------- #


def test_explicit_environment_on_resource(fresh_init):
    fresh_init(environment="staging")

    assert TracerWrapper.resource_attributes["moda.environment"] == "staging"
    assert TracerWrapper.resource_attributes["deployment.environment"] == "staging"

    span = _resource_span()
    assert span.resource.attributes["moda.environment"] == "staging"
    assert span.resource.attributes["deployment.environment"] == "staging"


def test_environment_defaults_to_production(fresh_init, monkeypatch):
    monkeypatch.delenv("MODA_ENVIRONMENT", raising=False)
    fresh_init()

    assert TracerWrapper.resource_attributes["moda.environment"] == "production"
    assert TracerWrapper.resource_attributes["deployment.environment"] == "production"

    span = _resource_span()
    assert span.resource.attributes["moda.environment"] == "production"
    assert span.resource.attributes["deployment.environment"] == "production"


def test_environment_env_var_is_honored(fresh_init, monkeypatch):
    monkeypatch.setenv("MODA_ENVIRONMENT", "from-env")
    fresh_init()

    assert TracerWrapper.resource_attributes["moda.environment"] == "from-env"
    assert TracerWrapper.resource_attributes["deployment.environment"] == "from-env"


def test_explicit_arg_wins_over_env_var(fresh_init, monkeypatch):
    monkeypatch.setenv("MODA_ENVIRONMENT", "from-env")
    fresh_init(environment="explicit")

    assert TracerWrapper.resource_attributes["moda.environment"] == "explicit"
    assert TracerWrapper.resource_attributes["deployment.environment"] == "explicit"


# --------------------------------------------------------------------------- #
# Span-level override
# --------------------------------------------------------------------------- #


def test_span_level_override_wins_over_resource_default(exporter):
    # The shared `exporter` fixture inits with the default ('production')
    # resource. A span-level override must win for spans inside the block.
    with moda.set_environment("canary"):
        _emit_span("inside")
    _emit_span("outside")

    spans = {s.name: s for s in exporter.get_finished_spans()}
    inside = spans["inside"]
    outside = spans["outside"]

    # Span created inside the block carries the override as a span attribute,
    # overriding the resource-level default.
    assert inside.attributes.get("moda.environment") == "canary"
    # Span outside the block gets no span-level override; it falls back to the
    # resource default (which carries moda.environment).
    assert outside.attributes.get("moda.environment") is None
    assert "moda.environment" in outside.resource.attributes


def test_set_environment_restores_previous_on_exit():
    assert moda.get_environment() is None
    with moda.set_environment("canary"):
        assert moda.get_environment() == "canary"
    assert moda.get_environment() is None


def test_set_environment_value_persists():
    try:
        moda.set_environment_value("persisted")
        assert moda.get_environment() == "persisted"
    finally:
        moda.set_environment_value(None)
    assert moda.get_environment() is None
