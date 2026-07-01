"""Tests for the ``.moda/last-conversation`` first-flush marker.

After the app emits its first trace, the CLI VERIFY watcher needs to know which
conversation to poll for. The SDK writes the active ``conversation_id`` to
``<cwd>/.moda/last-conversation`` on the first flush that has one. Path + format
match the Node SDK writer so a single CLI watcher reads both.
"""

import pytest
from unittest.mock import patch

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import ProxyTracerProvider

from traceloop.sdk import Traceloop
from traceloop.sdk.context import set_conversation_id_value
from traceloop.sdk.tracing.tracing import TracerWrapper


@pytest.fixture(autouse=True)
def isolated_tracer_wrapper():
    """Give each test a clean TracerWrapper singleton and restore the shared one.

    The tests below construct their own wrapper (with an isolated provider) so
    they can observe first-flush behaviour without disturbing the session-scoped
    wrapper used by the rest of the suite.
    """
    saved = getattr(TracerWrapper, "instance", None)
    if saved is not None:
        del TracerWrapper.instance
    yield
    set_conversation_id_value(None)
    if saved is not None:
        TracerWrapper.instance = saved
    elif hasattr(TracerWrapper, "instance"):
        del TracerWrapper.instance


def _init_isolated_wrapper():
    """Init a TracerWrapper bound to a throwaway provider (no global pollution)."""
    Traceloop.init(
        app_name="last-conversation-test",
        exporter=InMemorySpanExporter(),
        disable_batch=True,
    )


def test_marker_uses_cwd_at_init_not_flush(tmp_path, monkeypatch):
    """The marker is written relative to the CWD at init, even if the app
    changes directory before flushing (the watcher watches the launch dir)."""
    init_dir = tmp_path / "init_here"
    later_dir = tmp_path / "moved_here"
    init_dir.mkdir()
    later_dir.mkdir()

    monkeypatch.delenv("MODA_PROJECT_DIR", raising=False)
    monkeypatch.chdir(init_dir)
    with patch(
        "traceloop.sdk.tracing.tracing.get_tracer_provider",
        return_value=ProxyTracerProvider(),
    ), patch(
        "traceloop.sdk.tracing.tracing.trace.set_tracer_provider",
        lambda provider: None,
    ):
        _init_isolated_wrapper()
        set_conversation_id_value("conv_cwd")
        monkeypatch.chdir(later_dir)  # app changes cwd after init
        Traceloop.flush()

    assert (init_dir / ".moda" / "last-conversation").read_bytes() == b"conv_cwd"
    assert not (later_dir / ".moda" / "last-conversation").exists()


def test_marker_uses_project_dir_env_over_cwd(tmp_path, monkeypatch):
    """MODA_PROJECT_DIR pins the marker location regardless of the process CWD,
    even when startup code chdir's *before* moda.init() (the launcher sets this
    to the exact directory the VERIFY watcher polls)."""
    project_dir = tmp_path / "project"
    other_dir = tmp_path / "elsewhere"
    project_dir.mkdir()
    other_dir.mkdir()

    monkeypatch.setenv("MODA_PROJECT_DIR", str(project_dir))
    # App has already chdir'd away before Moda is even initialized.
    monkeypatch.chdir(other_dir)

    with patch(
        "traceloop.sdk.tracing.tracing.get_tracer_provider",
        return_value=ProxyTracerProvider(),
    ), patch(
        "traceloop.sdk.tracing.tracing.trace.set_tracer_provider",
        lambda provider: None,
    ):
        _init_isolated_wrapper()
        set_conversation_id_value("conv_proj")
        Traceloop.flush()

    assert (project_dir / ".moda" / "last-conversation").read_bytes() == b"conv_proj"
    assert not (other_dir / ".moda" / "last-conversation").exists()


def test_marker_written_on_first_flush(tmp_path, monkeypatch):
    """First flush after setting a conversation id writes the marker file."""
    monkeypatch.delenv("MODA_PROJECT_DIR", raising=False)
    monkeypatch.chdir(tmp_path)

    with patch(
        "traceloop.sdk.tracing.tracing.get_tracer_provider",
        return_value=ProxyTracerProvider(),
    ), patch(
        "traceloop.sdk.tracing.tracing.trace.set_tracer_provider",
        lambda provider: None,
    ):
        _init_isolated_wrapper()
        set_conversation_id_value("conv_x")
        Traceloop.flush()

    marker = tmp_path / ".moda" / "last-conversation"
    assert marker.exists(), "expected .moda/last-conversation to be written"
    # Byte-match the Node writer: raw conversation id, no trailing newline.
    assert marker.read_bytes() == b"conv_x"


def test_marker_not_rewritten_on_second_flush(tmp_path, monkeypatch):
    """A second flush must not rewrite (or overwrite) the marker."""
    monkeypatch.chdir(tmp_path)

    with patch(
        "traceloop.sdk.tracing.tracing.get_tracer_provider",
        return_value=ProxyTracerProvider(),
    ), patch(
        "traceloop.sdk.tracing.tracing.trace.set_tracer_provider",
        lambda provider: None,
    ):
        _init_isolated_wrapper()
        set_conversation_id_value("conv_x")
        Traceloop.flush()

        # Changing the conversation id and flushing again must not rewrite.
        set_conversation_id_value("conv_y")
        Traceloop.flush()

    marker = tmp_path / ".moda" / "last-conversation"
    assert marker.read_bytes() == b"conv_x", "second flush must not rewrite the marker"


def test_no_marker_when_conversation_id_unset(tmp_path, monkeypatch):
    """No conversation id at first flush -> no file written and no crash."""
    monkeypatch.chdir(tmp_path)
    set_conversation_id_value(None)

    with patch(
        "traceloop.sdk.tracing.tracing.get_tracer_provider",
        return_value=ProxyTracerProvider(),
    ), patch(
        "traceloop.sdk.tracing.tracing.trace.set_tracer_provider",
        lambda provider: None,
    ):
        _init_isolated_wrapper()
        # No conversation id set -> nothing to hand off.
        Traceloop.flush()

    assert not (tmp_path / ".moda" / "last-conversation").exists()


def test_marker_written_on_later_flush_when_id_set_afterwards(tmp_path, monkeypatch):
    """A flush with no id skips (without latching); a later flush still writes."""
    monkeypatch.chdir(tmp_path)
    set_conversation_id_value(None)

    with patch(
        "traceloop.sdk.tracing.tracing.get_tracer_provider",
        return_value=ProxyTracerProvider(),
    ), patch(
        "traceloop.sdk.tracing.tracing.trace.set_tracer_provider",
        lambda provider: None,
    ):
        _init_isolated_wrapper()
        Traceloop.flush()  # no id yet -> skipped, not latched
        assert not (tmp_path / ".moda" / "last-conversation").exists()

        set_conversation_id_value("conv_late")
        Traceloop.flush()

    marker = tmp_path / ".moda" / "last-conversation"
    assert marker.read_bytes() == b"conv_late"


def test_marker_write_failure_is_non_fatal(tmp_path, monkeypatch):
    """A failed marker write must not crash flush (routed through loud-fail)."""
    monkeypatch.chdir(tmp_path)

    with patch(
        "traceloop.sdk.tracing.tracing.get_tracer_provider",
        return_value=ProxyTracerProvider(),
    ), patch(
        "traceloop.sdk.tracing.tracing.trace.set_tracer_provider",
        lambda provider: None,
    ):
        _init_isolated_wrapper()
        set_conversation_id_value("conv_x")
        # Force the atomic write to blow up; flush must still return cleanly.
        with patch(
            "traceloop.sdk.tracing.tracing.tempfile.mkstemp",
            side_effect=OSError("disk full"),
        ):
            Traceloop.flush()

    assert not (tmp_path / ".moda" / "last-conversation").exists()
