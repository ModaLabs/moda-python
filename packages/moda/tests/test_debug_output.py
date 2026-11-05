"""moda.init(debug=True) must not print any part of the API key."""

from unittest.mock import patch

import pytest

import moda
from traceloop.sdk import Moda

API_KEY = "moda_0123456789abcdef"


def _key_fragments(key, size=4):
    return {key[i:i + size] for i in range(len(key) - size + 1)}


@pytest.mark.parametrize(
    ("api_key", "env_key", "expected"),
    [
        (API_KEY, None, "API key set: True"),
        (None, API_KEY, "API key set: True"),
        (None, None, "API key set: False"),
    ],
)
def test_debug_never_prints_key_characters(monkeypatch, capsys, api_key, env_key, expected):
    monkeypatch.delenv("TRACELOOP_API_KEY", raising=False)
    monkeypatch.delenv("MODA_API_KEY", raising=False)
    if env_key:
        monkeypatch.setenv("MODA_API_KEY", env_key)

    with patch.object(Moda, "init"), patch("logging.basicConfig"):
        moda.init(api_key, debug=True)

    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert expected in output
    assert not [fragment for fragment in _key_fragments(API_KEY) if fragment in output]
