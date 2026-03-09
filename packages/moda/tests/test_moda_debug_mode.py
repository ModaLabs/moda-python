from __future__ import annotations

import moda


class _DummyModa:
    last_instance: "_DummyModa | None" = None

    def __init__(self):
        _DummyModa.last_instance = self
        self.init_kwargs = {}

    def init(self, **kwargs):
        self.init_kwargs = kwargs

    def flush(self):
        return None


def _call_init(monkeypatch, env_value: str | None, explicit_debug=Ellipsis):
    monkeypatch.setitem(moda.init.__globals__, "Moda", _DummyModa)
    monkeypatch.setitem(moda.init.__globals__, "_moda_instance", None)

    if env_value is None:
        monkeypatch.delenv("MODA_DEBUG", raising=False)
    else:
        monkeypatch.setenv("MODA_DEBUG", env_value)

    if explicit_debug is Ellipsis:
        moda.init(api_key="moda_test_key")
    else:
        moda.init(api_key="moda_test_key", debug=explicit_debug)

    assert _DummyModa.last_instance is not None
    return _DummyModa.last_instance.init_kwargs


def test_debug_enabled_from_env_when_argument_omitted(monkeypatch):
    kwargs = _call_init(monkeypatch, env_value="true")
    assert kwargs["disable_batch"] is True


def test_debug_disabled_from_env_when_argument_omitted(monkeypatch):
    kwargs = _call_init(monkeypatch, env_value="false")
    assert "disable_batch" not in kwargs


def test_explicit_debug_false_overrides_env_true(monkeypatch):
    kwargs = _call_init(monkeypatch, env_value="true", explicit_debug=False)
    assert "disable_batch" not in kwargs


def test_explicit_debug_true_overrides_env_false(monkeypatch):
    kwargs = _call_init(monkeypatch, env_value="false", explicit_debug=True)
    assert kwargs["disable_batch"] is True


def test_unknown_env_value_falls_back_to_default_false(monkeypatch):
    kwargs = _call_init(monkeypatch, env_value="definitely-not-a-bool")
    assert "disable_batch" not in kwargs
