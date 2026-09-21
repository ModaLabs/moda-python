"""Explicit artifact uploads. Files are never captured or fetched implicitly."""

import asyncio
import hashlib
import os
import time
from datetime import datetime
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit

import httpx
from opentelemetry import trace
from traceloop.sdk.conversation import get_conversation_id

MAX_ARTIFACT_BYTES = 16 * 1024 * 1024
_default_client = None


class ArtifactError(RuntimeError):
    """Failed operation; status is None for transport errors."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class ArtifactInteractionError(ArtifactError):
    """Upload succeeded; retry linking using the retained artifact reference."""

    def __init__(self, artifact: dict[str, Any], interaction: dict[str, Any], cause: Exception):
        super().__init__("Artifact uploaded, but interaction could not be recorded", getattr(cause, "status", None))
        self.artifact = artifact
        self.interaction = interaction


def _service_url(endpoint: str) -> str:
    url = urlsplit(endpoint)
    if (
        url.scheme not in ("http", "https")
        or not url.netloc
        or url.username
        or url.password
        or url.query
        or url.fragment
    ):
        raise ValueError("Artifact endpoint must be an HTTP(S) URL without credentials or query")
    if url.scheme == "http" and url.hostname not in ("localhost", "127.0.0.1", "::1"):
        raise ValueError("Artifact endpoints require HTTPS except on loopback")
    path = url.path.rstrip("/")
    if path.endswith(("/v1/traces", "/v1/artifacts")):
        path = path.rsplit("/v1/", 1)[0]
    return urlunsplit((url.scheme, url.netloc, path, "", ""))


class ArtifactClient:
    """Bounded client; reuse idempotency keys on retries. Methods finish before returning.

    Explicit conversation IDs are recommended. Active OTel trace/span and Moda
    conversation context are used when omitted. No telemetry flush is required.
    """

    def __init__(
        self, api_key: str, endpoint: str, *, transport: httpx.BaseTransport | None = None, max_attempts: int = 3
    ):
        if not api_key:
            raise ValueError("Artifact API key is required")
        if not 1 <= max_attempts <= 5:
            raise ValueError("max_attempts must be between 1 and 5")
        self.base_url = _service_url(endpoint)
        self._key, self._transport, self._attempts = api_key, transport, max_attempts

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        content: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        with httpx.Client(transport=self._transport, timeout=30, follow_redirects=False) as client:
            for attempt in range(self._attempts):
                try:
                    response = client.request(
                        method,
                        self.base_url + path,
                        json=json,
                        content=content,
                        headers={**(headers or {}), "Authorization": f"Bearer {self._key}"},
                    )
                except httpx.TransportError as exc:
                    if attempt + 1 == self._attempts:
                        raise ArtifactError("Artifact service request failed") from exc
                else:
                    if response.is_success:
                        try:
                            payload = response.json() if response.content else {}
                        except ValueError as exc:
                            raise ArtifactError("Artifact service returned invalid JSON") from exc
                        if not isinstance(payload, dict):
                            raise ArtifactError("Artifact service returned an invalid object")
                        return payload
                    if response.status_code not in (408, 429, 500, 502, 503, 504) or attempt + 1 == self._attempts:
                        raise ArtifactError(
                            f"Artifact service returned HTTP {response.status_code}", response.status_code
                        )
                time.sleep(min(0.1 * 2**attempt, 1))
        raise ArtifactError("Artifact service request failed")

    @staticmethod
    def _prepare_interaction(
        *,
        artifact_id: str,
        version_id: str,
        event_id: str,
        occurred_at: str,
        actor: str,
        relation: str,
        conversation_id: str | None = None,
        trace_id: str | None = None,
        span_id: str | None = None,
        message_id: str | None = None,
        use_active_context: bool = True,
    ) -> dict[str, Any]:
        """Validate and freeze source identity before any upload/network operation."""
        if use_active_context:
            context = trace.get_current_span().get_span_context()
            conversation_id = conversation_id or get_conversation_id()
            if context.is_valid:
                active_trace, active_span = format(context.trace_id, "032x"), format(context.span_id, "016x")
                if trace_id is None and (span_id is None or span_id == active_span):
                    trace_id = active_trace
                if span_id is None and trace_id == active_trace:
                    span_id = active_span
        fields = dict(
            artifact_id=artifact_id,
            version_id=version_id,
            event_id=event_id,
            conversation_id=conversation_id,
            trace_id=trace_id,
            span_id=span_id,
            message_id=message_id,
        )
        for key, value in fields.items():
            if value is not None and (
                not isinstance(value, str)
                or not 1 <= len(value) <= 256
                or any(ord(character) < 32 for character in value)
            ):
                raise ValueError(f"Invalid artifact interaction {key}")
        if not artifact_id or not version_id or not event_id:
            raise ValueError("artifact_id, version_id and event_id are required")
        if actor not in ("user", "agent", "tool"):
            raise ValueError("Invalid artifact actor")
        if relation not in ("attached", "input", "read", "created", "output", "updated", "referenced"):
            raise ValueError("Invalid artifact relation")
        try:
            moment = datetime.fromisoformat(occurred_at.replace("Z", "+00:00"))
            if moment.tzinfo is None:
                raise ValueError("Timezone required")
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("occurred_at must be a timezone-qualified ISO timestamp") from exc
        if not any((conversation_id, trace_id, span_id, message_id)):
            raise ValueError("Artifact interaction requires conversation, trace, span or message context")
        return {**fields, "occurred_at": occurred_at, "actor": actor, "relation": relation, "use_active_context": False}

    def _send_interaction(self, prepared: dict[str, Any]) -> dict[str, Any]:
        wire_names = {
            "artifact_id": "artifactId",
            "version_id": "versionId",
            "event_id": "eventId",
            "occurred_at": "occurredAt",
            "conversation_id": "conversationId",
            "trace_id": "traceId",
            "span_id": "spanId",
            "message_id": "messageId",
            "actor": "actor",
            "relation": "relation",
        }
        body = {wire_names[key]: value for key, value in prepared.items() if key in wire_names and value is not None}
        return self._request("POST", "/v1/artifacts/interactions", json=body)

    def record_interaction(
        self,
        *,
        artifact_id: str,
        version_id: str,
        event_id: str,
        occurred_at: str,
        actor: str,
        relation: str,
        conversation_id: str | None = None,
        trace_id: str | None = None,
        span_id: str | None = None,
        message_id: str | None = None,
        use_active_context: bool = True,
    ) -> dict[str, Any]:
        """Use stable event ID/time. Pass saved error.interaction to retry exact source context."""
        prepared = self._prepare_interaction(
            artifact_id=artifact_id,
            version_id=version_id,
            event_id=event_id,
            occurred_at=occurred_at,
            actor=actor,
            relation=relation,
            conversation_id=conversation_id,
            trace_id=trace_id,
            span_id=span_id,
            message_id=message_id,
            use_active_context=use_active_context,
        )
        return self._send_interaction(prepared)

    def upload_artifact(
        self,
        data: bytes,
        *,
        file_name: str,
        content_type: str,
        idempotency_key: str,
        artifact_id: str | None = None,
        interaction: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Upload bytes and optionally link them. Linking failure preserves reference.

        interaction uses record_interaction kwargs excluding artifact_id/version_id.
        """
        if not isinstance(data, bytes) or len(data) > MAX_ARTIFACT_BYTES:
            raise ValueError("Artifact data must be bytes and at most 16 MiB")
        for key, value in (
            ("idempotency_key", idempotency_key),
            ("file_name", file_name),
            ("content_type", content_type),
        ):
            if not isinstance(value, str) or not 1 <= len(value) <= 256 or any(ord(c) < 32 for c in value):
                raise ValueError(f"Invalid artifact {key}")
        prepared = None
        if interaction is not None:
            if not isinstance(interaction, dict) or "artifact_id" in interaction or "version_id" in interaction:
                raise ValueError("interaction must omit artifact_id and version_id")
            prepared = self._prepare_interaction(artifact_id="pending", version_id="pending", **interaction)
        registration: dict[str, Any] = {
            "fileName": file_name,
            "contentType": content_type,
            "sizeBytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        if artifact_id:
            registration["artifactId"] = artifact_id
        upload = self._request(
            "POST", "/v1/artifacts/uploads", json=registration, headers={"Idempotency-Key": idempotency_key}
        )
        if not all(
            isinstance(upload.get(key), str) and upload[key] for key in ("artifactId", "versionId", "uploadUrl")
        ):
            raise ArtifactError("Artifact service returned an invalid registration")
        version_id = quote(upload["versionId"], safe="")
        if (
            upload["uploadUrl"] != self.base_url + f"/v1/artifacts/{version_id}/content"
            or upload.get("uploadMethod") != "PUT"
        ):
            raise ArtifactError("Unexpected upload target; refusing to forward credentials")
        self._request(
            "PUT", f"/v1/artifacts/{version_id}/content", content=data, headers={"Content-Type": content_type}
        )
        result = self._request("POST", f"/v1/artifacts/{version_id}/complete", json={})
        if (
            result.get("artifactId") != upload["artifactId"]
            or result.get("versionId") != upload["versionId"]
            or result.get("uploadState") != "uploaded"
        ):
            raise ArtifactError("Artifact service returned an invalid completion")
        if prepared is not None:
            prepared.update(artifact_id=upload["artifactId"], version_id=upload["versionId"])
            try:
                self._send_interaction(prepared)
            except ArtifactError as exc:
                raise ArtifactInteractionError(result, prepared, exc) from exc
        return result

    async def upload_artifact_async(self, data: bytes, **kwargs: Any) -> dict[str, Any]:
        """to_thread preserves active conversation/OTel context."""
        return await asyncio.to_thread(self.upload_artifact, data, **kwargs)

    async def record_interaction_async(self, **kwargs: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self.record_interaction, **kwargs)


def configure_artifacts(api_key: str | None, endpoint: str | None) -> None:
    global _default_client
    _default_client = None
    key = os.getenv("MODA_API_KEY") or os.getenv("TRACELOOP_API_KEY") or api_key
    url = (
        os.getenv("MODA_BASE_URL")
        or os.getenv("TRACELOOP_BASE_URL")
        or endpoint
        or "https://moda-ingest.modas.workers.dev/v1/traces"
    )
    _default_client = ArtifactClient(key, url) if key else None


def _client() -> ArtifactClient:
    if _default_client is None:
        raise ArtifactError("Call moda.init with an API key, or construct ArtifactClient explicitly")
    return _default_client


def upload_artifact(data: bytes, **kwargs: Any) -> dict[str, Any]:
    return _client().upload_artifact(data, **kwargs)


def record_artifact_interaction(**kwargs: Any) -> dict[str, Any]:
    return _client().record_interaction(**kwargs)


async def upload_artifact_async(data: bytes, **kwargs: Any) -> dict[str, Any]:
    return await _client().upload_artifact_async(data, **kwargs)


async def record_artifact_interaction_async(**kwargs: Any) -> dict[str, Any]:
    return await _client().record_interaction_async(**kwargs)
