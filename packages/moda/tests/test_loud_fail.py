"""Tests for the shared loud-fail contract (WS-SDK-PY serial root).

Covers the OnError enum, the handle_config_issue dispatch helper, the
explicit-arg > MODA_ON_ERROR env > default('warn') precedence, and the
end-to-end behavior of moda.init(on_error=...) on a missing API key.
"""

import logging

import pytest

from traceloop.sdk.errors import (
    OnError,
    DEFAULT_ON_ERROR,
    ModaConfigError,
    ModaMissingApiKeyError,
    coerce_on_error,
    resolve_on_error,
    handle_config_issue,
)


# --- Env isolation ---------------------------------------------------------

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
]


@pytest.fixture
def clean_env(monkeypatch):
    for var in _ISOLATED_ENV:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


@pytest.fixture(autouse=True)
def _restore_tracer_disabled():
    """Reset the process-wide TracerWrapper disable flag after each test.

    The `enabled=False` tests below call Moda.init, which flips the class-level
    TracerWrapper.__disabled to True. That flag persists across the whole
    session, so without this reset every later test in the suite would receive
    zero spans. Restore it to the enabled default after each test here.
    """
    from traceloop.sdk.tracing.tracing import TracerWrapper

    yield
    TracerWrapper.set_disabled(False)


# --- Contract shape (mirrors Node: silent | warn | throw) ------------------

def test_on_error_values_match_node_contract():
    assert [member.value for member in OnError] == ["silent", "warn", "throw"]
    # str-enum: equality with raw strings works for ergonomic call sites.
    assert OnError.SILENT == "silent"
    assert OnError.WARN == "warn"
    assert OnError.THROW == "throw"


def test_default_is_warn():
    assert DEFAULT_ON_ERROR is OnError.WARN


def test_missing_api_key_error_is_config_error():
    assert issubclass(ModaMissingApiKeyError, ModaConfigError)


def test_public_import_surface():
    # Acceptance: this import must succeed for sibling issues.
    from traceloop.sdk.errors import (  # noqa: F401
        OnError as _OnError,
        ModaConfigError as _ModaConfigError,
        handle_config_issue as _handle,
    )


# --- coerce_on_error -------------------------------------------------------

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


# --- resolve_on_error precedence -------------------------------------------

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


def test_handle_warn_prints_without_raising(capsys):
    handle_config_issue("heads up", on_error="warn")
    assert "heads up" in capsys.readouterr().out


def test_handle_default_warns(capsys):
    handle_config_issue("heads up")
    assert "heads up" in capsys.readouterr().out


def test_handle_silent_is_silent(capsys):
    handle_config_issue("quiet", on_error="silent")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_handle_silent_debug_logs(caplog):
    with caplog.at_level(logging.DEBUG):
        handle_config_issue("quiet", on_error="silent", logger=logging.getLogger("moda.test"))
    assert "quiet" in caplog.text


# --- End-to-end via moda.init on a missing API key -------------------------

def test_init_throw_missing_key_raises(clean_env):
    import moda

    with pytest.raises(ModaConfigError):
        moda.init(app_name="loud-fail-test", on_error="throw")


def test_init_warn_missing_key_no_raise(clean_env, capsys):
    import moda

    moda.init(app_name="loud-fail-test", on_error="warn")
    assert "Missing Moda API key" in capsys.readouterr().out


def test_init_default_missing_key_no_raise(clean_env, capsys):
    import moda

    moda.init(app_name="loud-fail-test")
    assert "Missing Moda API key" in capsys.readouterr().out


def test_init_silent_missing_key_no_output(clean_env, capsys):
    import moda

    moda.init(app_name="loud-fail-test", on_error="silent")
    captured = capsys.readouterr()
    assert "Missing Moda API key" not in captured.out


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


# --- Intentional opt-outs are never fatal, even under throw ----------------
# (enabled=False and tracing-disabled are configuration, not misconfiguration.)

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
