"""Tests for the loud-fail contract wired into the runtime paths (WS-SDK-PY / PY-3).

Covers the shared ``OnError`` contract (imported from ``traceloop.sdk.errors``,
the PY-1 serial root) applied to every remaining silent-failure path:

* the missing / invalid API-key guard in ``Moda.init`` / ``moda.init``,
* the intentional opt-outs (``enabled=False`` and tracing-disabled) which must
  never escalate to ``throw``,
* the ``init_tracer_provider`` failure branch (un-attachable / ``None`` provider),
* ``TracerWrapper.flush`` swallowing hard export/init failures.

For every path we assert the three settled modes:
  - ``throw``  -> raises ``ModaConfigError`` naming the config to fix,
  - ``warn``   -> logs/prints a warning and does NOT raise (coexistence),
  - ``silent`` -> no output, no raise.
"""

import logging

import pytest
from opentelemetry.sdk.resources import Resource

from traceloop.sdk.errors import (
    OnError,
    DEFAULT_ON_ERROR,
    ModaConfigError,
    ModaMissingApiKeyError,
    ModaExporterError,
    coerce_on_error,
    resolve_on_error,
    handle_config_issue,
)
from traceloop.sdk.tracing import tracing as tracing_mod
from traceloop.sdk.tracing.tracing import TracerWrapper, init_tracer_provider


# --- Env / global-state isolation ------------------------------------------

# Every key/endpoint override that could satisfy the missing-key guard, plus
# the loud-fail env override, so each test starts from a clean slate.
_ISOLATED_ENV = [
    "MODA_API_KEY",
    "TRACELOOP_API_KEY",
    "MODA_BASE_URL",
    "TRACELOOP_BASE_URL",
    "MODA_HEADERS",
    "TRACELOOP_HEADERS",
    "MODA_ON_ERROR",
    "TRACELOOP_TRACING_ENABLED",
]


@pytest.fixture
def clean_env(monkeypatch):
    for var in _ISOLATED_ENV:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


@pytest.fixture(autouse=True)
def _restore_global_tracer_state():
    """Restore process-wide TracerWrapper flags mutated by these tests.

    ``Moda.init`` sets the class-level ``TracerWrapper.on_error`` on every call
    and ``enabled=False`` flips ``TracerWrapper.__disabled`` to True — both
    persist for the whole session, so without this reset later tests in the
    suite would receive zero spans or an unexpected loud-fail mode. Snapshot and
    restore them around each test.
    """
    saved_on_error = TracerWrapper.on_error
    yield
    TracerWrapper.on_error = saved_on_error
    TracerWrapper.set_disabled(False)


# --- Contract shape (mirrors Node: silent | warn | throw) ------------------

def test_on_error_values_match_node_contract():
    assert [member.value for member in OnError] == ["silent", "warn", "throw"]
    assert OnError.SILENT == "silent"
    assert OnError.WARN == "warn"
    assert OnError.THROW == "throw"


def test_default_is_warn():
    assert DEFAULT_ON_ERROR is OnError.WARN


def test_exporter_error_is_config_error():
    assert issubclass(ModaMissingApiKeyError, ModaConfigError)
    assert issubclass(ModaExporterError, ModaConfigError)


# --- resolve / coerce precedence -------------------------------------------

@pytest.mark.parametrize(
    "value,expected",
    [
        (None, OnError.WARN),
        ("throw", OnError.THROW),
        ("THROW", OnError.THROW),
        ("  warn  ", OnError.WARN),
        (OnError.SILENT, OnError.SILENT),
        ("nonsense", OnError.WARN),  # unknown falls back to default, never raises
    ],
)
def test_coerce_on_error(value, expected):
    assert coerce_on_error(value) is expected


def test_resolve_default(clean_env):
    assert resolve_on_error() is OnError.WARN


def test_resolve_env_honored(clean_env):
    clean_env.setenv("MODA_ON_ERROR", "throw")
    assert resolve_on_error() is OnError.THROW


def test_resolve_explicit_beats_env(clean_env):
    clean_env.setenv("MODA_ON_ERROR", "throw")
    assert resolve_on_error("silent") is OnError.SILENT


# --- handle_config_issue dispatch ------------------------------------------

def test_handle_throw_raises():
    with pytest.raises(ModaConfigError, match="boom"):
        handle_config_issue("boom", on_error="throw")


def test_handle_throw_custom_error_cls():
    with pytest.raises(ModaMissingApiKeyError):
        handle_config_issue("no key", on_error="throw", error_cls=ModaMissingApiKeyError)


def test_handle_default_warns(capsys):
    handle_config_issue("heads up")
    assert "heads up" in capsys.readouterr().out


def test_handle_silent_is_silent(capsys):
    handle_config_issue("quiet", on_error="silent")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


# ===========================================================================
# Missing-API-key path (moda.init end-to-end, all three modes)
# ===========================================================================

def test_init_throw_missing_key_raises(clean_env):
    import moda

    with pytest.raises(ModaConfigError) as exc_info:
        moda.init(app_name="loud-fail-test", on_error="throw")
    # The message must name the config to fix, not be a bare OTEL stack trace.
    assert "MODA_API_KEY" in str(exc_info.value)


def test_init_throw_missing_key_is_missing_key_subtype(clean_env):
    import moda

    with pytest.raises(ModaMissingApiKeyError):
        moda.init(app_name="loud-fail-test", on_error="throw")


def test_init_warn_missing_key_no_raise(clean_env, capsys):
    import moda

    moda.init(app_name="loud-fail-test", on_error="warn")
    assert "Missing Moda API key" in capsys.readouterr().out


def test_init_default_missing_key_no_raise(clean_env, capsys):
    import moda

    # Default (no on_error arg) must reproduce today's behavior exactly.
    moda.init(app_name="loud-fail-test")
    assert "Missing Moda API key" in capsys.readouterr().out


def test_init_silent_missing_key_no_output(clean_env, capsys):
    import moda

    moda.init(app_name="loud-fail-test", on_error="silent")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_init_env_var_throw_honored(clean_env):
    import moda

    clean_env.setenv("MODA_ON_ERROR", "throw")
    with pytest.raises(ModaConfigError):
        moda.init(app_name="loud-fail-test")


def test_init_explicit_beats_env(clean_env, capsys):
    import moda

    # Env says throw, explicit arg says warn -> explicit wins (no raise).
    clean_env.setenv("MODA_ON_ERROR", "throw")
    moda.init(app_name="loud-fail-test", on_error="warn")
    assert "Missing Moda API key" in capsys.readouterr().out


def test_get_on_error_reflects_resolution(clean_env):
    import moda
    from traceloop.sdk import Moda

    moda.init(app_name="loud-fail-test", on_error="silent")
    assert Moda.get_on_error() is OnError.SILENT


# ===========================================================================
# Intentional opt-outs are never fatal, even under throw
# (enabled=False and tracing-disabled are configuration, not misconfiguration.)
# ===========================================================================

def test_init_disabled_flag_never_throws_under_throw(clean_env, capsys):
    import moda

    # Self-contradictory combo must not blow up: honor the disable, warn only.
    moda.init(app_name="loud-fail-test", enabled=False, on_error="throw")
    assert "disabled via init flag" in capsys.readouterr().out


def test_init_tracing_disabled_never_throws_under_throw(clean_env, capsys):
    import moda

    clean_env.setenv("TRACELOOP_TRACING_ENABLED", "false")
    moda.init(app_name="loud-fail-test", on_error="throw")
    assert "Tracing is disabled" in capsys.readouterr().out


def test_init_disabled_flag_silent_is_silent(clean_env, capsys):
    import moda

    moda.init(app_name="loud-fail-test", enabled=False, on_error="silent")
    assert capsys.readouterr().out == ""


# ===========================================================================
# Provider-init path: un-attachable / None tracer provider
# ===========================================================================

class _NoProcessorProvider:
    """Stand-in for a foreign OTel provider that can't take a span processor.

    It is not a ``ProxyTracerProvider`` and deliberately lacks
    ``add_span_processor``, so it drives ``init_tracer_provider`` into its
    failure branch.
    """


@pytest.fixture
def foreign_provider(monkeypatch):
    monkeypatch.setattr(
        tracing_mod, "get_tracer_provider", lambda: _NoProcessorProvider()
    )
    return Resource.create({"service.name": "loud-fail-test"})


def test_provider_init_throw_raises(foreign_provider):
    with pytest.raises(ModaConfigError) as exc_info:
        init_tracer_provider(resource=foreign_provider, on_error=OnError.THROW)
    # Names the offending config / how to fix, not a bare OTEL stack trace.
    assert "add_span_processor" in str(exc_info.value)


def test_provider_init_throw_is_exporter_error_subtype(foreign_provider):
    with pytest.raises(ModaExporterError):
        init_tracer_provider(resource=foreign_provider, on_error=OnError.THROW)


def test_provider_init_warn_returns_none_no_raise(foreign_provider, capsys):
    provider = init_tracer_provider(resource=foreign_provider, on_error=OnError.WARN)
    assert provider is None
    assert "add_span_processor" in capsys.readouterr().out


def test_provider_init_default_returns_none_no_raise(foreign_provider, capsys):
    # Default mode (no on_error arg) preserves the historical non-raising behavior.
    provider = init_tracer_provider(resource=foreign_provider)
    assert provider is None
    assert "add_span_processor" in capsys.readouterr().out


def test_provider_init_silent_returns_none_no_output(foreign_provider, capsys):
    provider = init_tracer_provider(resource=foreign_provider, on_error=OnError.SILENT)
    assert provider is None
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_provider_init_failure_leaves_sdk_uninitialized(monkeypatch):
    """Regression: an un-attachable provider must not leave a half-initialized
    singleton behind under warn/silent.

    ``verify_initialized()`` only checks ``hasattr(cls, "instance")``; if the
    bail path left the (broken, None-provider) instance registered, decorators
    and manual tracing would treat the SDK as initialized and later crash inside
    ``get_tracer()`` on the None provider. The bail must instead drop the
    registration so every tracing entry point gracefully no-ops.
    """
    # Preserve and clear the process-wide singleton / static endpoint so we can
    # drive TracerWrapper.__new__ down the failure branch in isolation.
    saved_instance = getattr(TracerWrapper, "instance", None)
    if hasattr(TracerWrapper, "instance"):
        del TracerWrapper.instance
    saved_endpoint = TracerWrapper.endpoint
    monkeypatch.setattr(TracerWrapper, "on_error", OnError.WARN)
    monkeypatch.setenv("TRACELOOP_SUPPRESS_WARNINGS", "true")  # quiet the no-op path
    try:
        # Non-empty endpoint so __new__ proceeds past the early no-endpoint bail.
        TracerWrapper.set_static_params({}, True, "https://example.test/v1/traces", {})
        monkeypatch.setattr(
            tracing_mod, "get_tracer_provider", lambda: _NoProcessorProvider()
        )

        # Constructing the wrapper must not raise under warn ...
        wrapper = TracerWrapper()

        # ... and the broken provider must NOT be reported as initialized, so
        # decorators/manual tracing skip span creation instead of crashing.
        assert TracerWrapper.verify_initialized() is False
        assert not hasattr(TracerWrapper, "instance")

        # Belt-and-suspenders: even a caller that still holds the bailed wrapper
        # and calls get_tracer() directly must degrade to a no-op tracer rather
        # than crash on the None provider (coexistence under warn/silent).
        tracer = wrapper.get_tracer()
        assert tracer is not None
        # A real tracer that can start (and no-op export) spans, not a crash.
        tracer.start_span("loud-fail-noop").end()
    finally:
        if hasattr(TracerWrapper, "instance"):
            del TracerWrapper.instance
        if saved_instance is not None:
            TracerWrapper.instance = saved_instance
        TracerWrapper.endpoint = saved_endpoint


def test_provider_init_throw_leaves_sdk_uninitialized(monkeypatch):
    """Under throw, init_tracer_provider raises inside TracerWrapper.__new__
    *after* cls.instance was already assigned.

    A caught ``Moda.init(on_error='throw')`` failure must not leave the partial
    singleton behind: ``verify_initialized()`` would otherwise report the SDK
    initialized even though no Moda provider/processor was ever attached, and
    decorated/manual tracing would run against a stale wrapper.
    """
    saved_instance = getattr(TracerWrapper, "instance", None)
    if hasattr(TracerWrapper, "instance"):
        del TracerWrapper.instance
    saved_endpoint = TracerWrapper.endpoint
    monkeypatch.setattr(TracerWrapper, "on_error", OnError.THROW)
    monkeypatch.setenv("TRACELOOP_SUPPRESS_WARNINGS", "true")  # quiet the no-op path
    try:
        TracerWrapper.set_static_params({}, True, "https://example.test/v1/traces", {})
        monkeypatch.setattr(
            tracing_mod, "get_tracer_provider", lambda: _NoProcessorProvider()
        )

        # The un-attachable provider must fail loudly ...
        with pytest.raises(ModaExporterError):
            TracerWrapper()

        # ... and must NOT leave a registered-but-broken singleton behind.
        assert not hasattr(TracerWrapper, "instance")
        assert TracerWrapper.verify_initialized() is False
    finally:
        if hasattr(TracerWrapper, "instance"):
            del TracerWrapper.instance
        if saved_instance is not None:
            TracerWrapper.instance = saved_instance
        TracerWrapper.endpoint = saved_endpoint


def test_provider_init_healthy_still_returns_provider(monkeypatch):
    # Sanity: a normal (Proxy) global provider still yields a real provider and
    # never trips the loud-fail path regardless of mode.
    from opentelemetry.trace import ProxyTracerProvider

    monkeypatch.setattr(
        tracing_mod, "get_tracer_provider", lambda: ProxyTracerProvider()
    )
    provider = init_tracer_provider(
        resource=Resource.create({}), on_error=OnError.THROW
    )
    assert provider is not None


# ===========================================================================
# flush() path: hard export / never-initialized failures
# ===========================================================================

class _BareWrapper(TracerWrapper):
    """A TracerWrapper whose singleton/OTEL setup is bypassed.

    Overriding ``__new__`` lets us exercise ``flush`` / ``_force_flush_processor``
    in isolation without touching the shared session singleton.
    """

    def __new__(cls):
        return object.__new__(cls)


class _FakeProcessor:
    def __init__(self, result=True, exc=None):
        self.result = result
        self.exc = exc
        self.calls = 0

    def force_flush(self, *args, **kwargs):
        self.calls += 1
        if self.exc is not None:
            raise self.exc
        return self.result


def _wrapper_with_processor(processor):
    wrapper = _BareWrapper()
    # Name-mangled attribute the real single-processor path uses.
    setattr(wrapper, "_TracerWrapper__spans_processor", processor)
    return wrapper


def test_flush_success_returns_true_no_output(monkeypatch, capsys):
    monkeypatch.setattr(TracerWrapper, "on_error", OnError.THROW)
    wrapper = _wrapper_with_processor(_FakeProcessor(result=True))
    assert wrapper.flush() is True
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_flush_throw_raises_on_false_result(monkeypatch):
    monkeypatch.setattr(TracerWrapper, "on_error", OnError.THROW)
    wrapper = _wrapper_with_processor(_FakeProcessor(result=False))
    with pytest.raises(ModaExporterError):
        wrapper.flush()


def test_flush_throw_raises_on_exception(monkeypatch):
    monkeypatch.setattr(TracerWrapper, "on_error", OnError.THROW)
    wrapper = _wrapper_with_processor(
        _FakeProcessor(exc=RuntimeError("connection refused"))
    )
    with pytest.raises(ModaExporterError):
        wrapper.flush()


def test_flush_warn_swallows_failure(monkeypatch, capsys):
    monkeypatch.setattr(TracerWrapper, "on_error", OnError.WARN)
    wrapper = _wrapper_with_processor(_FakeProcessor(result=False))
    assert wrapper.flush() is False  # no raise
    assert "flush" in capsys.readouterr().out.lower()


def test_flush_silent_swallows_failure(monkeypatch, capsys):
    monkeypatch.setattr(TracerWrapper, "on_error", OnError.SILENT)
    wrapper = _wrapper_with_processor(_FakeProcessor(result=False))
    assert wrapper.flush() is False  # no raise
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_flush_uninitialized_throw_raises(monkeypatch):
    # No span processor was ever attached -> a silently-broken pipeline.
    monkeypatch.setattr(TracerWrapper, "on_error", OnError.THROW)
    wrapper = _BareWrapper()
    with pytest.raises(ModaExporterError):
        wrapper.flush()


def test_flush_uninitialized_warn_no_raise(monkeypatch, capsys):
    monkeypatch.setattr(TracerWrapper, "on_error", OnError.WARN)
    wrapper = _BareWrapper()
    assert wrapper.flush() is False
    assert "never initialized" in capsys.readouterr().out


def test_flush_silent_debug_logs(monkeypatch, caplog):
    monkeypatch.setattr(TracerWrapper, "on_error", OnError.SILENT)
    wrapper = _BareWrapper()
    with caplog.at_level(logging.DEBUG, logger="traceloop.sdk.tracing.tracing"):
        assert wrapper.flush() is False
    assert "never initialized" in caplog.text
