"""First-trace attribution channel (MODA-585).

On the FIRST exported span that carries a ``moda.conversation_id`` attribute,
the SDK writes that id to ``<cwd>/.moda/last-conversation`` (single line,
overwritten each run). This is the PRIMARY first-trace attribution mechanism
the verify watcher reads (``readLastConversationId`` in the CLI's
``verify-read.ts``): with auto-instrumentation the developer never knows the
``conversation_id`` ahead of time, so this file is how "watch for THIS run's
trace" becomes deterministic.

Contract:
    * The id written is EXACTLY the ``moda.conversation_id`` span attribute --
      the same key the Data API ``/conversations/{id}/context`` resolves on
      (set by the OpenAI/Anthropic instrumentors via ``compute_conversation_id``).
    * Written ONCE per process (guarded by an instance flag), not per span, and
      overwritten on a fresh run.
    * Best-effort: write failures (e.g. read-only cwd) are swallowed and never
      propagate to the tracing pipeline. Attribution is never fatal to tracing.
"""

import os
import threading
from typing import Optional

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor

# Span attribute that carries the conversation id the Data API keys on.
CONVERSATION_ID_ATTR = "moda.conversation_id"

# Directory (relative to cwd) and filename for the attribution channel.
_MODA_DIR = ".moda"
_LAST_CONVERSATION_FILE = "last-conversation"


def _read_conversation_id(span: ReadableSpan) -> Optional[str]:
    """Return a non-empty ``moda.conversation_id`` from a span, or ``None``."""
    attributes = getattr(span, "attributes", None) or {}
    value = attributes.get(CONVERSATION_ID_ATTR)
    if not isinstance(value, str):
        return None
    trimmed = value.strip()
    return trimmed or None


def write_last_conversation(conversation_id: str, cwd: Optional[str] = None) -> None:
    """Best-effort write of ``<cwd>/.moda/last-conversation``.

    Creates ``.moda/`` if needed and overwrites the file with a single line.
    All errors are swallowed: a read-only cwd must never break tracing.
    """
    if cwd is None:
        cwd = os.getcwd()
    try:
        moda_dir = os.path.join(cwd, _MODA_DIR)
        os.makedirs(moda_dir, exist_ok=True)
        with open(
            os.path.join(moda_dir, _LAST_CONVERSATION_FILE), "w", encoding="utf-8"
        ) as handle:
            handle.write(f"{conversation_id}\n")
    except Exception:  # noqa: BLE001 - best-effort attribution, never fatal
        pass


class LastConversationWriterSpanProcessor(SpanProcessor):
    """A SpanProcessor that writes the first ``moda.conversation_id`` it sees.

    Watches spans for the first ``moda.conversation_id`` and writes it to
    ``<cwd>/.moda/last-conversation`` exactly once per process.

    This is intentionally a SEPARATE processor from the OTLP exporter processor
    so it works identically regardless of how the tracer provider is
    configured. The write is driven purely by ``on_end``, which fires when a
    span completes (attributes are final by then).
    """

    def __init__(self, cwd: Optional[str] = None) -> None:
        self._cwd = cwd if cwd is not None else os.getcwd()
        self._written = False
        self._lock = threading.Lock()

    def on_start(
        self, span: Span, parent_context: Optional[Context] = None
    ) -> None:
        # no-op: attribution is decided at span end when attributes are final.
        pass

    def on_end(self, span: ReadableSpan) -> None:
        # Write once per process. Cheap flag check first so we never touch the
        # filesystem (or take the lock) per span once we've written.
        if self._written:
            return
        conversation_id = _read_conversation_id(span)
        if not conversation_id:
            return
        with self._lock:
            if self._written:
                return
            # Flip the flag before writing so a (defensively) throwing write
            # cannot cause repeated attempts on every subsequent span.
            self._written = True
        write_last_conversation(conversation_id, self._cwd)

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        # Nothing buffered -- the write is synchronous in on_end.
        return True

    def shutdown(self) -> None:
        # Nothing to release.
        pass
