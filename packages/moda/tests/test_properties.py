"""Tests for moda.identify / moda.group / moda.track_subscription.

All HTTP is served by httpx.MockTransport; no network access.
"""

import json
import threading
from datetime import datetime, timedelta, timezone

import httpx
import pytest

import moda
from moda import properties


class Recorder:
    """MockTransport handler that records requests and replays scripted responses."""

    def __init__(self, responses=None):
        self.requests = []
        self.responses = list(responses or [])
        self.lock = threading.Lock()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        with self.lock:
            self.requests.append(request)
            step = self.responses.pop(0) if self.responses else 200
        if isinstance(step, Exception):
            raise step
        return httpx.Response(step, json={"success": 200 <= step < 300, "count": 1, "requestId": "r"})

    def bodies(self, path=None):
        return [json.loads(r.content) for r in self.requests if path is None or r.url.path == path]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("MODA_API_KEY", "TRACELOOP_API_KEY", "MODA_BASE_URL", "TRACELOOP_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    properties.reset()
    yield
    properties.reset()


@pytest.fixture
def recorder():
    return Recorder()


@pytest.fixture
def configured(recorder):
    sleeps = []
    properties.configure(
        api_key="moda_test_key",
        endpoint="https://ingest.example.com/v1/traces",
        transport=httpx.MockTransport(recorder),
        flush_interval=60.0,  # only flush explicitly or on batch size
        sleep=sleeps.append,
    )
    recorder.sleeps = sleeps
    return recorder


# ------------------------------------------------------------------ payloads


def test_identify_payload_shape(configured):
    moda.identify(
        "u_123",
        {"plan": "pro", "mrr": 49, "beta": True, "legacy": None},
        unset=["legacy_flag"],
        account_id="acct_9",
        timestamp=datetime(2026, 10, 13, 12, 0, 0, 123456, tzinfo=timezone.utc),
    )
    assert moda.flush() is None
    [req] = configured.requests
    assert req.method == "POST"
    assert str(req.url) == "https://ingest.example.com/v1/identify"
    assert req.headers["authorization"] == "Bearer moda_test_key"
    body = json.loads(req.content)
    [item] = body["batch"]
    assert set(body) == {"batch"}
    message_id = item.pop("message_id")
    assert len(message_id) == 36
    assert item == {
        "user_id": "u_123",
        "traits": {"plan": "pro", "mrr": 49, "beta": True, "legacy": None},
        "unset": ["legacy_flag"],
        "account_id": "acct_9",
        "timestamp": "2026-10-13T12:00:00.123Z",
    }


def test_identify_minimal_always_sends_timestamp(configured):
    before = datetime.now(timezone.utc) - timedelta(seconds=1)
    moda.identify("u_1")
    moda.flush()
    [item] = configured.bodies("/v1/identify")[0]["batch"]
    assert set(item) == {"user_id", "timestamp", "message_id"}
    ts = datetime.fromisoformat(item["timestamp"].replace("Z", "+00:00"))
    assert ts >= before
    assert item["timestamp"].endswith("Z") and len(item["timestamp"]) == 24


def test_group_payload_shape(configured):
    moda.group(
        "acct_9",
        {"name": "Acme", "mrr": 1200.5, "currency": "USD"},
        unset=[],
        user_id="u_123",
        membership="removed",
        timestamp="2026-01-02T03:04:05+02:00",
    )
    moda.flush()
    [req] = configured.requests
    assert req.url.path == "/v1/group"
    [item] = json.loads(req.content)["batch"]
    item.pop("message_id")
    assert item == {
        "account_id": "acct_9",
        "traits": {"name": "Acme", "mrr": 1200.5, "currency": "USD"},
        "unset": [],
        "user_id": "u_123",
        "membership": "removed",
        "timestamp": "2026-01-02T01:04:05.000Z",
    }


def test_subscription_payload_shape(configured):
    moda.track_subscription(
        "upgraded",
        event_id="evt_1",
        account_id="acct_9",
        subscription_id="sub_1",
        plan="enterprise",
        mrr=1200,
        currency="USD",
        occurred_at="2026-10-13T12:00:00Z",
        properties={"seats": 25, "source": "stripe"},
    )
    moda.flush()
    [req] = configured.requests
    assert req.url.path == "/v1/subscription-events"
    [item] = json.loads(req.content)["batch"]
    assert item == {
        "event_id": "evt_1",
        "type": "upgraded",
        "account_id": "acct_9",
        "subscription_id": "sub_1",
        "plan": "enterprise",
        "mrr": 1200,
        "currency": "USD",
        "occurred_at": "2026-10-13T12:00:00.000Z",
        "properties": {"seats": 25, "source": "stripe"},
    }


def test_subscription_for_user(configured):
    moda.track_subscription("trial_started", event_id="evt_2", user_id="u_1")
    moda.flush()
    [item] = configured.bodies()[0]["batch"]
    assert item["user_id"] == "u_1" and "account_id" not in item


def test_endpoint_derivation_and_env_precedence(monkeypatch, recorder):
    monkeypatch.setenv("MODA_API_KEY", "env_key")
    monkeypatch.setenv("MODA_BASE_URL", "http://localhost:8787/v1/traces/")
    properties.configure(api_key="arg_key", endpoint="https://ignored.example.com",
                         transport=httpx.MockTransport(recorder), flush_interval=60.0)
    moda.identify("u_1")
    moda.flush()
    [req] = recorder.requests
    assert str(req.url) == "http://localhost:8787/v1/identify"
    assert req.headers["authorization"] == "Bearer env_key"


def test_default_endpoint():
    client = properties.configure(api_key="k")
    assert client.base_url == "https://moda-ingest.modas.workers.dev"


# ------------------------------------------------------------------ batching


def test_batches_per_endpoint(configured):
    for i in range(3):
        moda.identify(f"u_{i}")
    moda.group("acct_1")
    moda.flush()
    identify_bodies = configured.bodies("/v1/identify")
    group_bodies = configured.bodies("/v1/group")
    assert len(identify_bodies) == 1 and len(identify_bodies[0]["batch"]) == 3
    assert len(group_bodies) == 1 and len(group_bodies[0]["batch"]) == 1
    assert [i["user_id"] for i in identify_bodies[0]["batch"]] == ["u_0", "u_1", "u_2"]


def test_batch_size_capped_at_100(configured):
    for i in range(250):
        moda.identify(f"u_{i}")
    moda.flush()
    sizes = [len(b["batch"]) for b in configured.bodies("/v1/identify")]
    assert sizes == [100, 100, 50]
    ids = [item["message_id"] for b in configured.bodies() for item in b["batch"]]
    assert len(set(ids)) == 250


def test_interval_flush_without_explicit_flush(recorder):
    properties.configure(api_key="k", endpoint="https://x.example.com/v1/traces",
                         transport=httpx.MockTransport(recorder), flush_interval=0.05)
    moda.identify("u_1")
    deadline = datetime.now() + timedelta(seconds=5)
    while not recorder.requests and datetime.now() < deadline:
        threading.Event().wait(0.02)
    assert len(recorder.requests) == 1


def test_queue_overflow_drops_with_warning(recorder, caplog):
    client = properties.PropertiesClient("k", "https://x.example.com", transport=httpx.MockTransport(recorder),
                                         max_queue_size=2, flush_interval=60.0)
    # Fill the queue without starting the worker thread.
    client._ensure_thread = lambda: None
    assert client.enqueue("/v1/identify", {"user_id": "a"})
    assert client.enqueue("/v1/identify", {"user_id": "b"})
    with caplog.at_level("WARNING", logger="moda.properties"):
        assert client.enqueue("/v1/identify", {"user_id": "c"}) is False
    assert "queue is full" in caplog.text


# ------------------------------------------------------------------ retries


@pytest.mark.parametrize("first", [500, 503, 429, httpx.ConnectError("boom"), httpx.ReadTimeout("slow")])
def test_retries_then_succeeds_with_stable_message_id(configured, first):
    configured.responses = [first, first, 200]
    moda.identify("u_1")
    moda.flush()
    assert len(configured.requests) == 3
    ids = {b["batch"][0]["message_id"] for b in configured.bodies()}
    assert len(ids) == 1
    assert len(configured.sleeps) == 2
    assert configured.sleeps[1] > 0


def test_gives_up_after_three_attempts(configured, caplog):
    configured.responses = [500, 500, 500, 500]
    with caplog.at_level("WARNING", logger="moda.properties"):
        moda.group("acct_1")
        moda.flush()
    assert len(configured.requests) == 3
    assert "giving up" in caplog.text


@pytest.mark.parametrize("status", [400, 401, 413])
def test_no_retry_on_client_errors(configured, status, caplog):
    configured.responses = [status]
    with caplog.at_level("WARNING", logger="moda.properties"):
        moda.identify("u_1")
        moda.flush()
    assert len(configured.requests) == 1
    assert configured.sleeps == []
    assert f"HTTP {status}" in caplog.text


def test_worker_survives_errors(configured):
    configured.responses = [400]
    moda.identify("u_1")
    moda.flush()
    moda.identify("u_2")
    moda.flush()
    assert [b["batch"][0]["user_id"] for b in configured.bodies()] == ["u_1", "u_2"]


# ------------------------------------------------------------------ flush / lifecycle


def test_flush_drains_everything(configured):
    for i in range(20):
        moda.identify(f"u_{i}")
        moda.track_subscription("renewed", event_id=f"e_{i}", account_id="a")
    moda.flush()
    assert properties._client._queue.empty()
    assert sum(len(b["batch"]) for b in configured.bodies()) == 40


def test_flush_is_safe_with_nothing_queued(configured):
    moda.flush()
    assert configured.requests == []


def test_noop_before_init(caplog):
    with caplog.at_level("DEBUG", logger="moda.properties"):
        moda.identify("u_1", {"plan": "pro"})
        moda.group("acct_1")
        moda.track_subscription("started", event_id="e", account_id="a")
        moda.flush()
    assert properties._client is None
    assert "before moda.init()" in caplog.text


def test_noop_without_api_key():
    assert properties.configure(api_key=None) is None
    moda.identify("u_1")
    assert properties._client is None


def test_validation_still_runs_before_init():
    with pytest.raises(ValueError):
        moda.identify("")


def test_moda_init_configures_properties(monkeypatch):
    calls = {}
    monkeypatch.setattr(moda.Moda, "init", lambda self, **kw: calls.setdefault("init", kw), raising=False)
    monkeypatch.setattr(properties, "configure", lambda **kw: calls.setdefault("configure", kw))
    moda.init(api_key="k", endpoint="https://x.example.com/v1/traces")
    assert calls["configure"] == {"api_key": "k", "endpoint": "https://x.example.com/v1/traces"}


# ------------------------------------------------------------------ validation


@pytest.mark.parametrize(
    "call, exc",
    [
        (lambda: moda.identify(None), TypeError),
        (lambda: moda.identify(""), ValueError),
        (lambda: moda.identify("x" * 257), ValueError),
        (lambda: moda.identify("u", {"nested": {"a": 1}}), TypeError),
        (lambda: moda.identify("u", {"list": [1, 2]}), TypeError),
        (lambda: moda.identify("u", {"$email": "a"}), ValueError),
        (lambda: moda.identify("u", {"1bad": "a"}), ValueError),
        (lambda: moda.identify("u", {"has space": "a"}), ValueError),
        (lambda: moda.identify("u", {"k" * 129: "a"}), ValueError),
        (lambda: moda.identify("u", {"nan": float("nan")}), ValueError),
        (lambda: moda.identify("u", {"long": "x" * 1025}), ValueError),
        (lambda: moda.identify("u", {f"k{i}": i for i in range(101)}), ValueError),
        (lambda: moda.identify("u", ["not", "a", "dict"]), TypeError),
        (lambda: moda.identify("u", unset="legacy"), TypeError),
        (lambda: moda.identify("u", unset=["$reserved"]), ValueError),
        (lambda: moda.identify("u", timestamp="yesterday"), ValueError),
        (lambda: moda.identify("u", timestamp=12345), TypeError),
        (lambda: moda.group(""), ValueError),
        (lambda: moda.group("a", membership="active"), ValueError),
        (lambda: moda.group("a", user_id="u", membership="pending"), ValueError),
        (lambda: moda.track_subscription("started", event_id="e"), ValueError),
        (lambda: moda.track_subscription("started", event_id="e", account_id="a", user_id="u"), ValueError),
        (lambda: moda.track_subscription("bogus", event_id="e", account_id="a"), ValueError),
        (lambda: moda.track_subscription("started", event_id="", account_id="a"), ValueError),
        (lambda: moda.track_subscription("started", event_id="e", account_id="a", mrr="10"), TypeError),
        (lambda: moda.track_subscription("started", event_id="e", account_id="a", mrr=True), TypeError),
        (lambda: moda.track_subscription("started", event_id="e", account_id="a", properties={"x": {}}), TypeError),
    ],
)
def test_validation_errors(call, exc, configured):
    with pytest.raises(exc):
        call()
    moda.flush()
    assert configured.requests == []


def test_valid_key_shapes_and_datetime_values(configured):
    moda.identify(
        "u",
        {"_private": 1, "a.b-c_d": "x", "k" * 128: 1, "signup_at": datetime(2026, 1, 2, tzinfo=timezone.utc)},
    )
    moda.flush()
    [item] = configured.bodies()[0]["batch"]
    assert item["traits"]["signup_at"] == "2026-01-02T00:00:00.000Z"
