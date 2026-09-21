import json

import httpx
import pytest
from opentelemetry import trace
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

from moda import ArtifactClient, ArtifactError, ArtifactInteractionError


def fake_service(calls, *, fail_link=False, evil_upload=False):
    def handle(request):
        calls.append(request)
        assert request.headers["authorization"] == "Bearer isolated-key"
        path = request.url.path
        if path.endswith("/uploads"):
            return httpx.Response(
                200,
                json={
                    "artifactId": "a",
                    "versionId": "v",
                    "uploadMethod": "PUT",
                    "uploadUrl": "https://evil.test/steal"
                    if evil_upload
                    else "https://local.test/v1/artifacts/v/content",
                },
            )
        if path.endswith("/complete"):
            return httpx.Response(200, json={"artifactId": "a", "versionId": "v", "uploadState": "uploaded"})
        if path.endswith("/interactions") and fail_link:
            return httpx.Response(409, json={"error": "conflict"})
        return httpx.Response(200, json={})

    return httpx.MockTransport(handle)


def test_upload_checksum_completion_and_stable_retry_key():
    calls = []
    client = ArtifactClient("isolated-key", "https://local.test/v1/traces", transport=fake_service(calls))
    for _ in range(2):
        assert (
            client.upload_artifact(
                b"hello", file_name="notes.txt", content_type="text/plain", idempotency_key="stable"
            )["versionId"]
            == "v"
        )
    assert [request.method for request in calls] == ["POST", "PUT", "POST"] * 2
    assert calls[0].headers["idempotency-key"] == calls[3].headers["idempotency-key"] == "stable"
    assert json.loads(calls[0].content)["sha256"] == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


def test_link_failure_retains_uploaded_reference():
    calls = []
    client = ArtifactClient("isolated-key", "https://local.test", transport=fake_service(calls, fail_link=True))
    with pytest.raises(ArtifactInteractionError) as error:
        client.upload_artifact(
            b"hello",
            file_name="notes.txt",
            content_type="text/plain",
            idempotency_key="stable",
            interaction={
                "event_id": "event",
                "occurred_at": "2026-09-21T00:00:00Z",
                "actor": "user",
                "relation": "attached",
                "conversation_id": "c",
            },
        )
    assert error.value.artifact["versionId"] == "v"
    assert error.value.status == 409
    assert len(calls) == 4


def test_never_sends_credentials_to_unexpected_upload_target():
    calls = []
    client = ArtifactClient("isolated-key", "https://local.test", transport=fake_service(calls, evil_upload=True))
    with pytest.raises(ArtifactError, match="Unexpected upload target"):
        client.upload_artifact(b"x", file_name="x", content_type="text/plain", idempotency_key="stable")
    assert len(calls) == 1


def test_active_span_context_and_explicit_trace_isolation():
    calls = []
    client = ArtifactClient("isolated-key", "https://local.test", transport=fake_service(calls))
    span = NonRecordingSpan(SpanContext(trace_id=123, span_id=456, is_remote=False, trace_flags=TraceFlags(1)))
    kwargs = dict(
        artifact_id="a",
        version_id="v",
        event_id="e",
        occurred_at="2026-09-21T00:00:00Z",
        actor="agent",
        relation="read",
    )
    with trace.use_span(span):
        client.record_interaction(**kwargs)
        client.record_interaction(**kwargs, trace_id="other")
    body = json.loads(calls[0].content)
    assert body["traceId"] == format(123, "032x")
    assert body["spanId"] == format(456, "016x")
    assert "spanId" not in json.loads(calls[1].content)


def test_transient_retry_uses_same_event_body(monkeypatch):
    monkeypatch.setattr("moda.artifacts.time.sleep", lambda _: None)
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(503 if len(calls) == 1 else 200, json={})

    client = ArtifactClient("isolated-key", "https://local.test", transport=httpx.MockTransport(handle))
    client.record_interaction(
        artifact_id="a",
        version_id="v",
        event_id="e",
        occurred_at="2026-09-21T00:00:00Z",
        actor="agent",
        relation="read",
        conversation_id="c",
    )
    assert len(calls) == 2
    assert calls[0].content == calls[1].content


@pytest.mark.asyncio
async def test_async_upload_awaits_completion():
    calls = []
    client = ArtifactClient("isolated-key", "https://local.test", transport=fake_service(calls))
    result = await client.upload_artifact_async(
        b"x", file_name="x", content_type="text/plain", idempotency_key="stable"
    )
    assert result["uploadState"] == "uploaded"
    assert len(calls) == 3


def test_freezes_context_before_upload_and_preserves_exact_link_retry(monkeypatch):
    conversation = {"id": "original"}
    monkeypatch.setattr("moda.artifacts.get_conversation_id", lambda: conversation["id"])
    calls = []
    service = fake_service(calls, fail_link=True)

    def handle(request):
        if request.url.path.endswith("/uploads"):
            conversation["id"] = "changed-during-upload"
        return service.handle_request(request)

    client = ArtifactClient("isolated-key", "https://local.test", transport=httpx.MockTransport(handle))
    with pytest.raises(ArtifactInteractionError) as error:
        client.upload_artifact(
            b"hello",
            file_name="file.txt",
            content_type="text/plain",
            idempotency_key="stable",
            interaction={
                "event_id": "e",
                "occurred_at": "2026-01-01T00:00:00Z",
                "actor": "user",
                "relation": "attached",
            },
        )
    assert json.loads(calls[-1].content)["conversationId"] == "original"
    assert error.value.interaction["conversation_id"] == "original"
    assert error.value.interaction["use_active_context"] is False
    new_client = ArtifactClient("isolated-key", "https://local.test", transport=fake_service(calls))
    span = NonRecordingSpan(SpanContext(trace_id=987, span_id=654, is_remote=False, trace_flags=TraceFlags(1)))
    with trace.use_span(span):
        new_client.record_interaction(**error.value.interaction)
    assert calls[-1].content == calls[-2].content
    assert "traceId" not in json.loads(calls[-1].content)


def test_invalid_interaction_is_rejected_before_upload():
    calls = []
    client = ArtifactClient("isolated-key", "https://local.test", transport=fake_service(calls))
    with pytest.raises(TypeError):
        client.upload_artifact(
            b"x", file_name="x", content_type="text/plain", idempotency_key="key", interaction={"typo": "value"}
        )
    with pytest.raises(ValueError, match="actor"):
        client.upload_artifact(
            b"x",
            file_name="x",
            content_type="text/plain",
            idempotency_key="key",
            interaction={
                "event_id": "e",
                "occurred_at": "2026-01-01T00:00:00Z",
                "actor": "invalid",
                "relation": "read",
                "conversation_id": "c",
            },
        )
    assert calls == []


def test_invalid_server_metadata_has_explicit_error():
    client = ArtifactClient(
        "key", "https://local.test", transport=httpx.MockTransport(lambda _: httpx.Response(200, json={}))
    )
    with pytest.raises(ArtifactError, match="invalid registration"):
        client.upload_artifact(b"x", file_name="x", content_type="text/plain", idempotency_key="key")


def test_reconfiguration_clears_old_client_when_new_endpoint_is_invalid(monkeypatch):
    from moda.artifacts import _client, configure_artifacts

    for key in ("MODA_API_KEY", "TRACELOOP_API_KEY", "MODA_BASE_URL", "TRACELOOP_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    configure_artifacts("old-key", "https://old.test/v1/traces")
    assert _client().base_url == "https://old.test"
    configure_artifacts("new-key", "https://new.test/v1/artifacts")
    assert _client().base_url == "https://new.test"
    with pytest.raises(ValueError):
        configure_artifacts("third-key", "invalid")
    with pytest.raises(ArtifactError, match="moda.init"):
        _client()


def test_explicit_span_does_not_inherit_a_different_active_trace():
    calls = []
    client = ArtifactClient("isolated-key", "https://local.test", transport=fake_service(calls))
    span = NonRecordingSpan(SpanContext(trace_id=123, span_id=456, is_remote=False, trace_flags=TraceFlags(1)))
    with trace.use_span(span):
        client.record_interaction(
            artifact_id="a",
            version_id="v",
            event_id="e",
            occurred_at="2026-01-01T00:00:00Z",
            actor="tool",
            relation="read",
            span_id="other-span",
        )
    assert "traceId" not in json.loads(calls[0].content)
