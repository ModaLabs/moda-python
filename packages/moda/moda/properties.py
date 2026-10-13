"""End-user and account properties: identify, group and subscription events.

Calls validate their arguments synchronously (raising ``ValueError`` /
``TypeError`` for programmer errors) and then enqueue the payload for
non-blocking delivery. A daemon background thread batches items per endpoint
and POSTs them to the Moda ingest worker:

- ``POST /v1/identify``
- ``POST /v1/group``
- ``POST /v1/subscription-events``

Delivery never raises into application code. Call ``moda.flush()`` to block
until queued items have been sent (this also happens automatically at
interpreter exit).
"""

from __future__ import annotations

import atexit
import logging
import math
import os
import queue
import random
import re
import threading
import time
import uuid
from datetime import date, datetime, timezone
from typing import Any, Callable, Iterable, Mapping, Optional, Union
from urllib.parse import urlparse

import httpx

logger = logging.getLogger("moda.properties")

DEFAULT_ENDPOINT = "https://moda-ingest.modas.workers.dev/v1/traces"

IDENTIFY_PATH = "/v1/identify"
GROUP_PATH = "/v1/group"
SUBSCRIPTION_PATH = "/v1/subscription-events"

MAX_BATCH_SIZE = 100
MAX_QUEUE_SIZE = 10_000
FLUSH_INTERVAL_SECONDS = 1.0
REQUEST_TIMEOUT_SECONDS = 10.0
MAX_ATTEMPTS = 3
BACKOFF_BASE_SECONDS = 0.5
BACKOFF_MAX_SECONDS = 8.0
DEFAULT_FLUSH_TIMEOUT_SECONDS = 30.0

MAX_ID_LENGTH = 256
MAX_TRAITS = 100
MAX_STRING_VALUE_LENGTH = 1024

SUBSCRIPTION_EVENT_TYPES = frozenset(
    {
        "trial_started",
        "trial_converted",
        "started",
        "renewed",
        "upgraded",
        "downgraded",
        "cancelled",
        "reactivated",
        "churned",
    }
)
MEMBERSHIP_VALUES = frozenset({"active", "removed"})

_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.\-]{0,127}$")
_RETRYABLE_STATUS = frozenset({429}) | frozenset(range(500, 600))

Scalar = Union[str, int, float, bool, None]
Timestamp = Union[datetime, str]

# ============================================================
# Validation
# ============================================================


def _validate_id(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, got {type(value).__name__}")
    if not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    if len(value) > MAX_ID_LENGTH:
        raise ValueError(f"{name} must be at most {MAX_ID_LENGTH} characters, got {len(value)}")
    return value


def _validate_key(key: Any, name: str) -> str:
    if not isinstance(key, str):
        raise TypeError(f"{name} keys must be strings, got {type(key).__name__}")
    if key.startswith("$"):
        raise ValueError(f"{name} key {key!r} is invalid: keys starting with '$' are reserved")
    if not _KEY_RE.match(key):
        raise ValueError(
            f"{name} key {key!r} is invalid: keys must match ^[A-Za-z_][A-Za-z0-9_.\\-]{{0,127}}$"
        )
    return key


def _validate_scalar(value: Any, name: str) -> Scalar:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        if len(value) > MAX_STRING_VALUE_LENGTH:
            raise ValueError(
                f"{name} string values must be at most {MAX_STRING_VALUE_LENGTH} characters, got {len(value)}"
            )
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{name} numeric values must be finite, got {value!r}")
        return value
    if isinstance(value, datetime):
        return _format_timestamp(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (Mapping, list, tuple, set, frozenset)):
        raise TypeError(f"{name} values must be scalars (str, int, float, bool, None); nested {type(value).__name__} "
                        "values are not supported")
    raise TypeError(f"{name} values must be scalars (str, int, float, bool, None), got {type(value).__name__}")


def _validate_properties(props: Any, name: str) -> Optional[dict]:
    if props is None:
        return None
    if not isinstance(props, Mapping):
        raise TypeError(f"{name} must be a dict, got {type(props).__name__}")
    if len(props) > MAX_TRAITS:
        raise ValueError(f"{name} must have at most {MAX_TRAITS} keys, got {len(props)}")
    out = {}
    for key, value in props.items():
        _validate_key(key, name)
        out[key] = _validate_scalar(value, f"{name}[{key!r}]")
    return out


def _validate_unset(unset: Any) -> Optional[list]:
    if unset is None:
        return None
    if isinstance(unset, (str, bytes)) or not isinstance(unset, Iterable) or isinstance(unset, Mapping):
        raise TypeError("unset must be a list of trait keys")
    keys = [_validate_key(k, "unset") for k in unset]
    if len(keys) > MAX_TRAITS:
        raise ValueError(f"unset must have at most {MAX_TRAITS} keys, got {len(keys)}")
    return keys


def _format_timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    value = value.astimezone(timezone.utc)
    return value.strftime("%Y-%m-%dT%H:%M:%S.") + f"{value.microsecond // 1000:03d}Z"


def _validate_timestamp(value: Any, name: str) -> str:
    """Normalize a datetime or ISO-8601 string to UTC ISO-8601 with milliseconds.

    Naive datetimes (and ISO strings without an offset) are treated as UTC.
    ``None`` means "now".
    """
    if value is None:
        return _format_timestamp(datetime.now(timezone.utc))
    if isinstance(value, datetime):
        return _format_timestamp(value)
    if isinstance(value, str):
        text = value.strip()
        if text.endswith(("Z", "z")):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            raise ValueError(f"{name} must be an ISO-8601 timestamp, got {value!r}") from None
        return _format_timestamp(parsed)
    raise TypeError(f"{name} must be a datetime or ISO-8601 string, got {type(value).__name__}")


def _validate_number(value: Any, name: str) -> Optional[Union[int, float]]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return value


def _validate_optional_str(value: Any, name: str) -> Optional[str]:
    if value is None:
        return None
    return _validate_id(value, name)


# ============================================================
# Delivery
# ============================================================


def _resolve_base_url(endpoint: str) -> Optional[str]:
    """Derive the ingest base URL from the configured traces endpoint.

    ``https://host/v1/traces`` -> ``https://host``. Any path prefix in front of
    the OTLP signal suffix (e.g. a reverse proxy mount) is preserved.
    """
    parsed = urlparse(endpoint.strip())
    if parsed.scheme.lower() not in ("http", "https") or not parsed.netloc:
        return None
    path = parsed.path.rstrip("/")
    for suffix in ("/v1/traces", "/v1/metrics", "/v1/logs"):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break
    return f"{parsed.scheme}://{parsed.netloc}{path}"


class _FlushRequest:
    __slots__ = ("done",)

    def __init__(self) -> None:
        self.done = threading.Event()


class _Stop:
    pass


class PropertiesClient:
    """Background batching sender for identify/group/subscription payloads."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        *,
        transport: Optional[httpx.BaseTransport] = None,
        flush_interval: float = FLUSH_INTERVAL_SECONDS,
        max_batch_size: int = MAX_BATCH_SIZE,
        max_queue_size: int = MAX_QUEUE_SIZE,
        max_attempts: int = MAX_ATTEMPTS,
        backoff_base: float = BACKOFF_BASE_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._transport = transport
        self._flush_interval = flush_interval
        self._max_batch_size = max_batch_size
        self._max_attempts = max_attempts
        self._backoff_base = backoff_base
        self._sleep = sleep
        self._queue: "queue.Queue[Any]" = queue.Queue(maxsize=max_queue_size)
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._http: Optional[httpx.Client] = None
        self._closed = False
        self._dropped = 0
        self._pending_flushes: list = []

    # ---------------------------------------------------------------- public

    def enqueue(self, path: str, item: dict) -> bool:
        if self._closed:
            logger.debug("Moda properties client is shut down; dropping %s item", path)
            return False
        try:
            self._queue.put_nowait((path, item))
        except queue.Full:
            self._dropped += 1
            logger.warning(
                "Moda properties queue is full (%d items); dropping %s item (dropped so far: %d)",
                self._queue.maxsize,
                path,
                self._dropped,
            )
            return False
        self._ensure_thread()
        return True

    def flush(self, timeout: Optional[float] = DEFAULT_FLUSH_TIMEOUT_SECONDS) -> bool:
        """Block until everything enqueued before this call has been sent (or dropped)."""
        thread = self._thread
        if thread is None or not thread.is_alive():
            return self._queue.empty()
        request = _FlushRequest()
        try:
            self._queue.put(request, timeout=timeout)
        except queue.Full:
            logger.warning("Moda properties flush timed out waiting for queue space")
            return False
        if not request.done.wait(timeout):
            logger.warning("Moda properties flush timed out after %ss", timeout)
            return False
        return True

    def shutdown(self, timeout: Optional[float] = DEFAULT_FLUSH_TIMEOUT_SECONDS) -> None:
        if self._closed:
            return
        self.flush(timeout)
        self._closed = True
        thread = self._thread
        if thread is not None and thread.is_alive():
            try:
                self._queue.put(_Stop(), timeout=1.0)
            except queue.Full:
                pass
            thread.join(timeout=1.0)
        if self._http is not None:
            try:
                self._http.close()
            except Exception:  # pragma: no cover - defensive
                pass

    # -------------------------------------------------------------- internal

    def _ensure_thread(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(target=self._run, name="moda-properties", daemon=True)
            self._thread.start()

    def _client(self) -> httpx.Client:
        if self._http is None:
            self._http = httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS, transport=self._transport)
        return self._http

    def _run(self) -> None:
        buffers: dict[str, list] = {}
        last_flush = time.monotonic()
        while True:
            try:
                remaining = max(0.0, self._flush_interval - (time.monotonic() - last_flush))
                try:
                    entry = self._queue.get(timeout=remaining)
                except queue.Empty:
                    entry = None

                if isinstance(entry, _Stop):
                    self._send_all(buffers)
                    return
                if isinstance(entry, _FlushRequest):
                    self._drain_nowait(buffers)
                    self._send_all(buffers)
                    last_flush = time.monotonic()
                    entry.done.set()
                    continue
                if entry is not None:
                    path, item = entry
                    buf = buffers.setdefault(path, [])
                    buf.append(item)
                    if len(buf) >= self._max_batch_size:
                        buffers[path] = []
                        self._send(path, buf)

                if time.monotonic() - last_flush >= self._flush_interval:
                    self._send_all(buffers)
                    last_flush = time.monotonic()
            except Exception:  # never let the worker die
                logger.exception("Moda properties worker error")

    def _drain_nowait(self, buffers: dict) -> None:
        """Pull items already queued (ahead of a flush request) into buffers."""
        while True:
            try:
                entry = self._queue.get_nowait()
            except queue.Empty:
                return
            if isinstance(entry, _FlushRequest):
                # Will be satisfied by the send that follows.
                self._pending_flushes.append(entry)
                continue
            if isinstance(entry, _Stop):
                # Re-queue so the loop exits after this flush.
                self._queue.put_nowait(entry)
                return
            path, item = entry
            buffers.setdefault(path, []).append(item)

    def _send_all(self, buffers: dict) -> None:
        for path in list(buffers):
            items = buffers.pop(path)
            for i in range(0, len(items), self._max_batch_size):
                self._send(path, items[i : i + self._max_batch_size])
        pending = self._pending_flushes
        while pending:
            pending.pop().done.set()

    def _send(self, path: str, items: list) -> None:
        if not items:
            return
        url = f"{self.base_url}{path}"
        body = {"batch": items}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        for attempt in range(1, self._max_attempts + 1):
            retry_after: Optional[float] = None
            try:
                response = self._client().post(url, json=body, headers=headers)
            except httpx.TransportError as exc:  # network errors and timeouts
                reason = f"{type(exc).__name__}: {exc}"
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Moda properties: dropping %d item(s) for %s: %s", len(items), path, exc)
                return
            else:
                status = response.status_code
                if 200 <= status < 300:
                    self._log_warnings(path, response)
                    return
                if status not in _RETRYABLE_STATUS:
                    logger.warning(
                        "Moda properties: %s rejected %d item(s) with HTTP %d (not retried): %s",
                        path,
                        len(items),
                        status,
                        _short_body(response),
                    )
                    return
                reason = f"HTTP {status}"
                retry_after = _parse_retry_after(response)

            if attempt >= self._max_attempts:
                logger.warning(
                    "Moda properties: giving up on %d item(s) for %s after %d attempts (%s)",
                    len(items),
                    path,
                    attempt,
                    reason,
                )
                return
            delay = min(BACKOFF_MAX_SECONDS, self._backoff_base * (2 ** (attempt - 1)))
            delay = delay * (0.5 + random.random() / 2)
            if retry_after is not None:
                delay = min(BACKOFF_MAX_SECONDS, max(delay, retry_after))
            logger.debug("Moda properties: retrying %s in %.2fs (%s)", path, delay, reason)
            self._sleep(delay)

    @staticmethod
    def _log_warnings(path: str, response: httpx.Response) -> None:
        try:
            data = response.json()
        except Exception:
            return
        if isinstance(data, dict):
            for warning in data.get("warnings") or []:
                logger.debug("Moda properties: %s warning: %s", path, warning)


def _short_body(response: httpx.Response) -> str:
    try:
        text = response.text
    except Exception:  # pragma: no cover - defensive
        return ""
    return text[:500]


def _parse_retry_after(response: httpx.Response) -> Optional[float]:
    value = response.headers.get("retry-after")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


# ============================================================
# Module-level configuration
# ============================================================

_client: Optional[PropertiesClient] = None
_client_lock = threading.Lock()
_atexit_registered = False


def configure(
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None,
    **client_kwargs: Any,
) -> Optional[PropertiesClient]:
    """Configure delivery. Called by ``moda.init``; uses the same resolution order.

    Environment variables take precedence over arguments, matching ``moda.init``:
    ``MODA_API_KEY`` / ``TRACELOOP_API_KEY`` then ``api_key``;
    ``MODA_BASE_URL`` / ``TRACELOOP_BASE_URL`` then ``endpoint`` then the default.
    """
    global _client, _atexit_registered
    resolved_key = os.getenv("MODA_API_KEY") or os.getenv("TRACELOOP_API_KEY") or api_key
    resolved_endpoint = os.getenv("MODA_BASE_URL") or os.getenv("TRACELOOP_BASE_URL") or endpoint or DEFAULT_ENDPOINT

    with _client_lock:
        old = _client
        _client = None
        if old is not None:
            old.shutdown(timeout=5.0)

        if not resolved_key:
            logger.debug("Moda properties disabled: no API key configured")
            return None
        base_url = _resolve_base_url(resolved_endpoint)
        if base_url is None:
            logger.warning(
                "Moda properties disabled: endpoint %r is not an http(s) URL", resolved_endpoint
            )
            return None

        _client = PropertiesClient(resolved_key, base_url, **client_kwargs)
        if not _atexit_registered:
            atexit.register(_shutdown_at_exit)
            _atexit_registered = True
        return _client


def flush(timeout: Optional[float] = DEFAULT_FLUSH_TIMEOUT_SECONDS) -> bool:
    """Block until queued property payloads are delivered. Returns False on timeout."""
    client = _client
    if client is None:
        return True
    return client.flush(timeout)


def reset() -> None:
    """Shut down and discard the current client (mainly for tests)."""
    global _client
    with _client_lock:
        old = _client
        _client = None
    if old is not None:
        old.shutdown(timeout=5.0)


def _shutdown_at_exit() -> None:
    client = _client
    if client is None:
        return
    try:
        client.shutdown(timeout=DEFAULT_FLUSH_TIMEOUT_SECONDS)
    except Exception:  # pragma: no cover - never raise at exit
        pass


def _enqueue(path: str, item: dict) -> None:
    client = _client
    if client is None:
        logger.debug("Moda properties: %s called before moda.init() (or without an API key); ignoring", path)
        return
    client.enqueue(path, item)


# ============================================================
# Public API
# ============================================================


def identify(
    user_id: str,
    traits: Optional[Mapping[str, Scalar]] = None,
    *,
    unset: Optional[Iterable[str]] = None,
    account_id: Optional[str] = None,
    timestamp: Optional[Timestamp] = None,
) -> None:
    """Record traits for an end user of your product.

    Args:
        user_id: Your stable, opaque identifier for the user (1-256 chars).
        traits: Flat dict of scalar values (str, int, float, bool). ``None``
            values unset the trait. At most 100 keys.
        unset: Trait keys to remove.
        account_id: Optional account the user belongs to.
        timestamp: When the traits were true (datetime or ISO-8601 string).
            Defaults to now.

    Raises:
        TypeError / ValueError: for invalid arguments. Delivery errors are
        never raised; they are logged.
    """
    item: dict = {"user_id": _validate_id(user_id, "user_id")}
    validated_traits = _validate_properties(traits, "traits")
    if validated_traits is not None:
        item["traits"] = validated_traits
    validated_unset = _validate_unset(unset)
    if validated_unset is not None:
        item["unset"] = validated_unset
    if account_id is not None:
        item["account_id"] = _validate_id(account_id, "account_id")
    item["timestamp"] = _validate_timestamp(timestamp, "timestamp")
    item["message_id"] = str(uuid.uuid4())
    _enqueue(IDENTIFY_PATH, item)


def group(
    account_id: str,
    traits: Optional[Mapping[str, Scalar]] = None,
    *,
    unset: Optional[Iterable[str]] = None,
    user_id: Optional[str] = None,
    membership: Optional[str] = None,
    timestamp: Optional[Timestamp] = None,
) -> None:
    """Record traits for an account (company, workspace, team).

    Args:
        account_id: Your stable, opaque identifier for the account (1-256 chars).
        traits: Flat dict of scalar values. ``None`` values unset the trait.
        unset: Trait keys to remove.
        user_id: Optional user whose membership in this account is recorded.
        membership: ``"active"`` (server default when ``user_id`` is set) or
            ``"removed"``. Requires ``user_id``.
        timestamp: When the traits were true. Defaults to now.
    """
    item: dict = {"account_id": _validate_id(account_id, "account_id")}
    validated_traits = _validate_properties(traits, "traits")
    if validated_traits is not None:
        item["traits"] = validated_traits
    validated_unset = _validate_unset(unset)
    if validated_unset is not None:
        item["unset"] = validated_unset
    if user_id is not None:
        item["user_id"] = _validate_id(user_id, "user_id")
    if membership is not None:
        if not isinstance(membership, str):
            raise TypeError(f"membership must be a string, got {type(membership).__name__}")
        if membership not in MEMBERSHIP_VALUES:
            raise ValueError(f"membership must be one of {sorted(MEMBERSHIP_VALUES)}, got {membership!r}")
        if user_id is None:
            raise ValueError("membership requires user_id")
        item["membership"] = membership
    item["timestamp"] = _validate_timestamp(timestamp, "timestamp")
    item["message_id"] = str(uuid.uuid4())
    _enqueue(GROUP_PATH, item)


def track_subscription(
    type: str,
    *,
    event_id: str,
    account_id: Optional[str] = None,
    user_id: Optional[str] = None,
    subscription_id: Optional[str] = None,
    plan: Optional[str] = None,
    mrr: Optional[Union[int, float]] = None,
    currency: Optional[str] = None,
    occurred_at: Optional[Timestamp] = None,
    properties: Optional[Mapping[str, Scalar]] = None,
) -> None:
    """Record a subscription lifecycle event for an account or a user.

    Args:
        type: One of ``trial_started``, ``trial_converted``, ``started``,
            ``renewed``, ``upgraded``, ``downgraded``, ``cancelled``,
            ``reactivated``, ``churned``.
        event_id: Caller-provided idempotency key (e.g. your billing
            provider's event id). Re-sending the same id is deduplicated.
        account_id / user_id: Exactly one must be provided.
        subscription_id, plan, mrr, currency: Optional subscription details.
        occurred_at: When the event happened. Defaults to now.
        properties: Flat dict of additional scalar values.
    """
    if not isinstance(type, str):
        raise TypeError(f"type must be a string, got {type.__class__.__name__}")
    if type not in SUBSCRIPTION_EVENT_TYPES:
        raise ValueError(f"type must be one of {sorted(SUBSCRIPTION_EVENT_TYPES)}, got {type!r}")
    item: dict = {"event_id": _validate_id(event_id, "event_id"), "type": type}
    if (account_id is None) == (user_id is None):
        raise ValueError("track_subscription requires exactly one of account_id or user_id")
    if account_id is not None:
        item["account_id"] = _validate_id(account_id, "account_id")
    else:
        item["user_id"] = _validate_id(user_id, "user_id")
    if subscription_id is not None:
        item["subscription_id"] = _validate_id(subscription_id, "subscription_id")
    if plan is not None:
        item["plan"] = _validate_optional_str(plan, "plan")
    validated_mrr = _validate_number(mrr, "mrr")
    if validated_mrr is not None:
        item["mrr"] = validated_mrr
    if currency is not None:
        item["currency"] = _validate_optional_str(currency, "currency")
    item["occurred_at"] = _validate_timestamp(occurred_at, "occurred_at")
    validated_props = _validate_properties(properties, "properties")
    if validated_props is not None:
        item["properties"] = validated_props
    _enqueue(SUBSCRIPTION_PATH, item)


__all__ = ["identify", "group", "track_subscription", "flush", "configure", "reset", "PropertiesClient"]
