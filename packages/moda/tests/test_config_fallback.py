from __future__ import annotations

import json
import os

import pytest

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


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    """Isolate env + config home and stub the underlying Moda instance."""
    monkeypatch.setitem(moda.init.__globals__, "Moda", _DummyModa)
    monkeypatch.setitem(moda.init.__globals__, "_moda_instance", None)
    monkeypatch.delenv("MODA_API_KEY", raising=False)
    monkeypatch.setenv("MODA_CONFIG_HOME", str(tmp_path))
    _DummyModa.last_instance = None
    yield


def _write_config(tmp_path, obj) -> None:
    (tmp_path / "config.json").write_text(json.dumps(obj), encoding="utf-8")


def test_explicit_key_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("MODA_API_KEY", "moda_env_key")
    _write_config(tmp_path, {"api_key": "moda_config_key"})
    moda.init(api_key="moda_explicit")
    assert _DummyModa.last_instance.init_kwargs["api_key"] == "moda_explicit"


def test_env_key_used_when_no_explicit(monkeypatch, tmp_path):
    monkeypatch.setenv("MODA_API_KEY", "moda_env_key")
    _write_config(tmp_path, {"api_key": "moda_config_key"})
    moda.init()
    assert _DummyModa.last_instance.init_kwargs["api_key"] == "moda_env_key"


def test_config_file_fallback_when_env_absent(tmp_path):
    _write_config(tmp_path, {"api_key": "moda_config_key"})
    moda.init()
    assert _DummyModa.last_instance.init_kwargs["api_key"] == "moda_config_key"


def test_loud_fail_when_no_source(tmp_path):
    with pytest.raises(ValueError) as excinfo:
        moda.init()
    message = str(excinfo.value)
    assert "MODA_API_KEY" in message
    assert "config.json" in message
    assert "moda init" in message


def test_no_loud_fail_when_exporter_supplied(tmp_path):
    # A custom exporter (used in tests) does not require an api key.
    moda.init(exporter=object())
    assert _DummyModa.last_instance is not None


def test_malformed_config_is_tolerated(tmp_path):
    (tmp_path / "config.json").write_text("{ not json", encoding="utf-8")
    with pytest.raises(ValueError):
        moda.init()


def test_config_path_honors_env(monkeypatch, tmp_path):
    monkeypatch.setenv("MODA_CONFIG_HOME", str(tmp_path))
    assert moda._moda_config_path() == os.path.join(str(tmp_path), "config.json")
