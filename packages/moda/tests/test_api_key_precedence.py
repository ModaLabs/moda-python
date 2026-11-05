import pytest

import traceloop.sdk as sdk
from traceloop.sdk import Moda


@pytest.fixture
def captured(monkeypatch):
    seen = {}

    class FakeWrapper:
        def __init__(self, **kwargs):
            seen["uploader_key"] = kwargs["image_uploader"].api_key

        @staticmethod
        def set_static_params(resource_attributes, enable_content, endpoint, headers):
            seen["headers"] = headers

    monkeypatch.setattr(sdk, "TracerWrapper", FakeWrapper)
    monkeypatch.setenv("MODA_METRICS_ENABLED", "false")
    monkeypatch.setenv("TRACELOOP_METRICS_ENABLED", "false")
    monkeypatch.delenv("MODA_HEADERS", raising=False)
    monkeypatch.delenv("TRACELOOP_HEADERS", raising=False)
    monkeypatch.delenv("TRACELOOP_API_KEY", raising=False)
    return seen


def test_a_key_passed_to_init_wins_over_the_environment(monkeypatch, captured):
    monkeypatch.setenv("MODA_API_KEY", "env-key")
    Moda.init(api_key="passed-key", app_name="precedence")
    assert captured["headers"] == {"Authorization": "Bearer passed-key"}
    assert captured["uploader_key"] == "passed-key"


def test_the_environment_key_is_used_when_none_is_passed(monkeypatch, captured):
    monkeypatch.setenv("MODA_API_KEY", "env-key")
    Moda.init(app_name="precedence")
    assert captured["headers"] == {"Authorization": "Bearer env-key"}
    assert captured["uploader_key"] == "env-key"
