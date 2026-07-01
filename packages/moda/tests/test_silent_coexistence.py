"""Silent-coexistence tests for ``moda.init()`` across the realistic host modes.

COEXISTENCE is a locked first principle of the Moda SDK: ``moda.init()`` must
NEVER regress an evaluator's existing observability. ``tests/test_otel_coexistence.py``
proves the *functional* half (Moda detects/attaches to an existing provider or
creates its own). This module proves the *silence* half — that Moda is quiet and
non-disruptive — across the four realistic OpenTelemetry hosting arrangements
plus the init-before-provider ordering case:

  1. Sentry-style    — an external SDK registered a real ``TracerProvider``
                       before ``moda.init()``; Moda attaches its processor via
                       ``add_span_processor`` and both exporters keep receiving
                       spans.
  2. Datadog/OTLP    — a provider is present and an ``OTEL_EXPORTER_OTLP_*``
                       endpoint is configured via env; Moda must not clobber the
                       env and must not crash.
  3. PostHog-style   — Moda's ``EXCLUDED_URLS`` keeps the host provider's own
                       export endpoints out of Moda's HTTP instrumentation.
  4. None/greenfield — no external provider; Moda creates and registers its own
                       (the ``ProxyTracerProvider`` path).
  5. init-before-provider — Moda initialises before the host registers its
                       provider; the host provider that appears later is
                       unaffected.

In every mode we assert the silent-mode invariant on the valid-key happy path
with the default (``warn``) error behaviour:
  (a) no ``WARNING+`` logging records and nothing written to stderr,
  (b) the external provider's finished-span count is identical with vs without
      ``moda.init()`` (zero regression),
  (c) Moda's own spans land in the expected exporter,
  (d) ``moda.init()`` does not raise.

Isolation notes
---------------
* ``TracerWrapper`` is a process-wide singleton; ``clean_tracer_wrapper`` resets
  it around every test so ordering / repeated runs are stable. The module also
  overrides the shared session-scoped ``exporter`` fixture (as
  ``tests/test_otel_coexistence.py`` does) so conftest's global ``Traceloop.init``
  never runs for this module.
* ``instruments=set()`` is passed to ``Traceloop.init`` so these tests exercise
  Moda's *tracing-pipeline* coexistence in isolation. LLM-library instrumentation
  is orthogonal to the coexistence invariant, and because those instrumentors are
  global (and some optional backends such as ``sagemaker`` are not installed),
  re-running them under a reset singleton emits noise that a real one-shot
  ``moda.init()`` in a fresh process never produces.
* ``ThreadingInstrumentor`` is likewise global and is instrumented
  unconditionally inside ``TracerWrapper``; the fixture uninstruments it before
  each test so a reset-singleton re-init does not log "already instrumented" —
  again a pure test artifact, not a coexistence regression.
"""

import logging
import os
from contextlib import contextmanager
from unittest.mock import patch

import pytest
from opentelemetry.instrumentation.threading import ThreadingInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import ProxyTracerProvider

from traceloop.sdk.tracing.tracing import TracerWrapper

# A small, fixed host workload emitted through the *external* provider in each
# mode. Kept deterministic so span counts compare exactly with vs without Moda.
HOST_SPAN_NAMES = ("host-op-a", "host-op-b", "host-op-c")

# A recognisable, valid-looking key so we exercise the "valid-key happy path"
# rather than the missing-key branch.
VALID_KEY = "moda-test-key-silent-coexistence"


# --------------------------------------------------------------------------- #
# Fixtures (mirroring tests/test_otel_coexistence.py isolation patterns)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def exporter():
    """Override the shared session ``exporter`` fixture to keep this module
    isolated. Returning a bare in-memory exporter (without calling
    ``Traceloop.init``) prevents conftest's global initialisation — and its
    instrumentation side effects — from running for this module.
    """
    return InMemorySpanExporter()


@pytest.fixture(autouse=True)
def clean_tracer_wrapper():
    """Reset the process-wide ``TracerWrapper`` singleton and the global
    ``ThreadingInstrumentor`` around every test.

    This makes each test behave like a fresh ``moda.init()`` in a new process,
    which is what keeps the silence invariant meaningful and order-independent.
    """
    # ThreadingInstrumentor is global; uninstrument so Moda's unconditional
    # re-instrument inside TracerWrapper does not warn "already instrumented".
    ThreadingInstrumentor().uninstrument()

    if hasattr(TracerWrapper, "instance"):
        saved = TracerWrapper.instance
        del TracerWrapper.instance
    else:
        saved = None

    yield

    if saved is not None:
        TracerWrapper.instance = saved
    elif hasattr(TracerWrapper, "instance"):
        del TracerWrapper.instance


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _external_provider():
    """Build a real external ``TracerProvider`` with its own in-memory exporter,
    simulating a host SDK (Sentry/Datadog/PostHog) that set tracing up already.
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


def _emit(provider, names=HOST_SPAN_NAMES):
    """Emit the fixed host workload through ``provider``."""
    tracer = provider.get_tracer("external-host-app")
    for name in names:
        with tracer.start_as_current_span(name) as span:
            span.set_attribute("host.op", name)


@contextmanager
def _simulate_provider(provider):
    """Make ``moda.init()`` observe ``provider`` as the ambient global provider.

    ``trace.set_tracer_provider`` is neutralised so the greenfield path can
    create its own provider without mutating (or warning about overriding) the
    real process-global OTel provider.
    """
    with patch(
        "traceloop.sdk.tracing.tracing.get_tracer_provider",
        return_value=provider,
    ), patch(
        "traceloop.sdk.tracing.tracing.trace.set_tracer_provider",
        lambda provider: None,
    ):
        yield


def _init_moda(moda_exporter, **kwargs):
    """Initialise Moda on the valid-key happy path, capturing spans in
    ``moda_exporter`` and disabling LLM-library instrumentation (see module
    docstring for why).
    """
    from traceloop.sdk import Traceloop

    return Traceloop.init(
        api_key=VALID_KEY,
        app_name="silent-coexistence-test",
        exporter=moda_exporter,
        disable_batch=True,
        instruments=set(),
        **kwargs,
    )


def _assert_silent(caplog, capsys, action):
    """Run ``action`` (a ``moda.init()`` happy-path step) and assert the silence
    invariant: no ``WARNING+`` log records and nothing on stderr. Returns
    whatever ``action`` returns. A raise inside ``action`` fails the test,
    covering the "does not raise" part of the invariant.
    """
    capsys.readouterr()  # discard any output produced before the guarded action
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        result = action()
    captured = capsys.readouterr()

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert not warnings, "moda.init() emitted WARNING+ log records: " + "; ".join(
        f"{r.levelname} {r.name}: {r.getMessage()}" for r in warnings
    )
    assert captured.err == "", f"moda.init() wrote to stderr: {captured.err!r}"
    return result


# --------------------------------------------------------------------------- #
# The five coexistence modes
# --------------------------------------------------------------------------- #
class TestSilentCoexistence:
    def test_sentry_style_external_provider(self, caplog, capsys):
        """Mode 1 — a real external ``TracerProvider`` exists before init.

        Moda attaches its processor to it; the host exporter still sees exactly
        its own workload (zero regression) and Moda's exporter, sharing the same
        provider, sees the same spans.
        """
        # Baseline: identical host workload with NO Moda in the picture.
        baseline_provider, baseline_exp = _external_provider()
        _emit(baseline_provider)
        baseline_count = len(baseline_exp.get_finished_spans())
        assert baseline_count == len(HOST_SPAN_NAMES)

        # With Moda attached to a fresh, identical external provider.
        provider, ext_exp = _external_provider()
        moda_exp = InMemorySpanExporter()

        def _action():
            with _simulate_provider(provider):
                _init_moda(moda_exp)
                # Moda reused the existing provider rather than creating its own.
                assert TracerWrapper.instance.get_tracer() is not None
                _emit(provider)

        _assert_silent(caplog, capsys, _action)

        # (b) zero regression: host exporter count identical with vs without Moda.
        assert len(ext_exp.get_finished_spans()) == baseline_count
        # (c) Moda's processor is attached to the same provider, so its exporter
        #     also received the host spans — coexistence, not replacement.
        assert len(moda_exp.get_finished_spans()) == baseline_count

    def test_datadog_otlp_env_only(self, caplog, capsys, monkeypatch):
        """Mode 2 — provider present plus an ``OTEL_EXPORTER_OTLP_*`` env endpoint.

        Moda must not clobber the host's OTLP env configuration and must not
        crash; external spans still flow unchanged.
        """
        otlp_endpoint = "http://collector.datadog.internal:4318"
        otlp_traces = f"{otlp_endpoint}/v1/traces"
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", otlp_endpoint)
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", otlp_traces)

        baseline_provider, baseline_exp = _external_provider()
        _emit(baseline_provider)
        baseline_count = len(baseline_exp.get_finished_spans())

        provider, ext_exp = _external_provider()
        moda_exp = InMemorySpanExporter()

        def _action():
            with _simulate_provider(provider):
                _init_moda(moda_exp)
                _emit(provider)

        _assert_silent(caplog, capsys, _action)

        # Moda left the host's OTLP env configuration untouched.
        assert os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") == otlp_endpoint
        assert os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") == otlp_traces
        # Zero regression; external spans still flow.
        assert len(ext_exp.get_finished_spans()) == baseline_count == len(HOST_SPAN_NAMES)
        assert len(moda_exp.get_finished_spans()) == len(HOST_SPAN_NAMES)

    def test_posthog_style_excluded_urls(self, caplog, capsys):
        """Mode 3 — PostHog-style: host observability export endpoints are kept
        out of Moda's own HTTP tracing via ``EXCLUDED_URLS``.

        Also asserts the silent, zero-regression invariant with a PostHog-style
        provider present.
        """
        from traceloop.sdk.tracing.tracing import EXCLUDED_URLS

        # The host observability backends (and Moda/LLM endpoints) must be
        # excluded so Moda never traces the host's telemetry export traffic.
        for host in (
            "posthog.com",
            "sentry.io",
            "traceloop.com",
            "api.openai.com",
            "api.anthropic.com",
        ):
            assert host in EXCLUDED_URLS, f"{host} should be in EXCLUDED_URLS"

        # Silence + zero regression alongside a live host provider.
        baseline_provider, baseline_exp = _external_provider()
        _emit(baseline_provider)
        baseline_count = len(baseline_exp.get_finished_spans())

        provider, ext_exp = _external_provider()
        moda_exp = InMemorySpanExporter()

        def _action():
            with _simulate_provider(provider):
                _init_moda(moda_exp)
                _emit(provider)

        _assert_silent(caplog, capsys, _action)

        assert len(ext_exp.get_finished_spans()) == baseline_count == len(HOST_SPAN_NAMES)
        assert len(moda_exp.get_finished_spans()) == len(HOST_SPAN_NAMES)

    def test_none_greenfield_creates_own_provider(self, caplog, capsys):
        """Mode 4 — greenfield: no external provider (``ProxyTracerProvider``).

        Moda creates and registers its own provider; its spans land in its own
        exporter. There is no pre-existing observability to regress, so the
        invariant reduces to silence + Moda's own spans landing.
        """
        moda_exp = InMemorySpanExporter()

        def _action():
            with _simulate_provider(ProxyTracerProvider()):
                _init_moda(moda_exp)
                tracer = TracerWrapper.instance.get_tracer()
                for name in HOST_SPAN_NAMES:
                    with tracer.start_as_current_span(name):
                        pass

        _assert_silent(caplog, capsys, _action)

        # Moda created its own provider and its spans land in its exporter.
        assert len(moda_exp.get_finished_spans()) == len(HOST_SPAN_NAMES)

    def test_init_before_provider_ordering(self, caplog, capsys):
        """Mode 5 — init-before-provider ordering.

        Moda initialises FIRST in a greenfield process; a host ``TracerProvider``
        registered afterwards must be completely unaffected — its spans still
        flow to its own exporter and Moda stays silent.

        NOTE: when PY-4 (lazy delegate re-resolution) lands, this can be
        strengthened to also assert Moda re-attaches to the later provider.
        """
        # Baseline: host provider alone, no Moda.
        baseline_provider, baseline_exp = _external_provider()
        _emit(baseline_provider)
        baseline_count = len(baseline_exp.get_finished_spans())

        moda_exp = InMemorySpanExporter()

        def _init_first():
            with _simulate_provider(ProxyTracerProvider()):
                _init_moda(moda_exp)

        _assert_silent(caplog, capsys, _init_first)

        # Host provider is registered AFTER Moda and emits its workload.
        late_provider, late_exp = _external_provider()
        _emit(late_provider)

        # Zero regression: the later host provider is untouched by Moda.
        assert len(late_exp.get_finished_spans()) == baseline_count == len(HOST_SPAN_NAMES)
