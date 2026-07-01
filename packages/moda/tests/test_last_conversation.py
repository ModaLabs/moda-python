"""Tests for the .moda/last-conversation first-trace attribution writer (MODA-585)."""

import os
import stat

from traceloop.sdk.tracing.last_conversation import (
    LastConversationWriterSpanProcessor,
    write_last_conversation,
)


class _FakeSpan:
    """Minimal ReadableSpan-shaped object carrying (optionally) the id attr."""

    def __init__(self, conversation_id=None):
        self.attributes = {}
        if conversation_id is not None:
            self.attributes["moda.conversation_id"] = conversation_id


def _last_conversation_path(cwd):
    return os.path.join(cwd, ".moda", "last-conversation")


def _read(cwd):
    with open(_last_conversation_path(cwd), encoding="utf-8") as handle:
        return handle.read()


# --- write_last_conversation -------------------------------------------------


def test_write_creates_moda_dir_and_single_line(tmp_path):
    cwd = str(tmp_path)
    write_last_conversation("conv_abc123", cwd)

    path = _last_conversation_path(cwd)
    assert os.path.exists(path)
    assert _read(cwd) == "conv_abc123\n"


def test_write_overwrites_existing_file(tmp_path):
    cwd = str(tmp_path)
    write_last_conversation("conv_first", cwd)
    write_last_conversation("conv_second", cwd)

    assert _read(cwd).strip() == "conv_second"


def test_write_swallows_errors_on_read_only_cwd(tmp_path):
    cwd = str(tmp_path)
    os.chmod(cwd, stat.S_IREAD | stat.S_IEXEC)  # r-x, no write

    # Detect whether the perm change is actually enforced (root ignores it).
    enforced = True
    try:
        os.makedirs(os.path.join(cwd, ".moda-probe"))
        enforced = False
    except OSError:
        pass

    try:
        # Must not raise regardless of enforcement.
        write_last_conversation("conv_ro", cwd)
        if enforced:
            assert not os.path.exists(_last_conversation_path(cwd))
    finally:
        os.chmod(cwd, stat.S_IRWXU)  # restore for cleanup


# --- LastConversationWriterSpanProcessor ------------------------------------


def test_processor_writes_id_from_first_span(tmp_path):
    cwd = str(tmp_path)
    processor = LastConversationWriterSpanProcessor(cwd=cwd)

    processor.on_end(_FakeSpan("conv_first_trace"))

    assert _read(cwd).strip() == "conv_first_trace"


def test_processor_ignores_spans_without_conversation_id(tmp_path):
    cwd = str(tmp_path)
    processor = LastConversationWriterSpanProcessor(cwd=cwd)

    processor.on_end(_FakeSpan(None))
    processor.on_end(_FakeSpan(""))
    assert not os.path.exists(_last_conversation_path(cwd))

    processor.on_end(_FakeSpan("conv_real"))
    assert _read(cwd).strip() == "conv_real"


def test_processor_writes_once_per_run_not_per_span(tmp_path):
    cwd = str(tmp_path)
    processor = LastConversationWriterSpanProcessor(cwd=cwd)

    processor.on_end(_FakeSpan("conv_run_1"))
    # Later spans within the same run (even different ids) must be ignored.
    processor.on_end(_FakeSpan("conv_run_1"))
    processor.on_end(_FakeSpan("conv_different"))

    assert _read(cwd).strip() == "conv_run_1"


def test_fresh_processor_overwrites_on_new_run(tmp_path):
    cwd = str(tmp_path)
    # A new run == new process == new processor instance.
    LastConversationWriterSpanProcessor(cwd=cwd).on_end(_FakeSpan("conv_run_1"))
    LastConversationWriterSpanProcessor(cwd=cwd).on_end(_FakeSpan("conv_run_2"))

    assert _read(cwd).strip() == "conv_run_2"


def test_processor_swallows_write_failure_on_read_only_cwd(tmp_path):
    cwd = str(tmp_path)
    os.chmod(cwd, stat.S_IREAD | stat.S_IEXEC)
    try:
        processor = LastConversationWriterSpanProcessor(cwd=cwd)
        # Must not raise.
        processor.on_end(_FakeSpan("conv_ro"))
    finally:
        os.chmod(cwd, stat.S_IRWXU)


def test_processor_force_flush_and_shutdown(tmp_path):
    processor = LastConversationWriterSpanProcessor(cwd=str(tmp_path))
    assert processor.force_flush() is True
    processor.shutdown()  # must not raise


def test_processor_write_id_matches_compute_conversation_id_format():
    """The written id is the same value produced by compute_conversation_id,
    i.e. the exact key the Data API /conversations/{id}/context resolves on."""
    from traceloop.sdk.conversation import compute_conversation_id

    messages = [{"role": "user", "content": "hello world"}]
    conv_id = compute_conversation_id(messages)
    assert conv_id.startswith("conv_")

    span = _FakeSpan(conv_id)
    # The processor reads exactly the moda.conversation_id attribute.
    from traceloop.sdk.tracing.last_conversation import _read_conversation_id

    assert _read_conversation_id(span) == conv_id
