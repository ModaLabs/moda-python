# Explicit artifact ingestion

The upload API must be enabled on the target Moda ingest service. These calls do not automatically read user files or download provider URLs. The first release accepts bytes up to 16 MiB; text/JSON/XML preview extraction has a separate 2 MiB limit. Files remain downloadable when preview extraction is unsupported.

```python
import moda

moda.init(api_key="...", endpoint="https://your-ingest/v1/traces")
moda.conversation_id = "session-123"

artifact = moda.upload_artifact(
    b"Quarterly results", file_name="results.txt", content_type="text/plain",
    idempotency_key="session-123-upload-1",
)
moda.record_artifact_interaction(
    artifact_id=artifact["artifactId"], version_id=artifact["versionId"],
    event_id="session-123-attachment-1", occurred_at="2026-09-21T10:00:00Z",
    actor="user", relation="attached", conversation_id="session-123",
)
```

Active OpenTelemetry trace/span context is included when available. Explicit conversation IDs make files discoverable before spans/messages arrive. Supported actors: `user`, `agent`, `tool`; relations: `attached`, `input`, `read`, `created`, `output`, `updated`, `referenced`.

Use the same upload idempotency key and identical bytes/metadata when retrying. Use the same interaction event ID, time and context when retrying a link. HTTP transient failures retry up to three times; authentication, validation and conflicts do not. Calls finish before returning, and do not depend on `moda.flush()`.

`upload_artifact_async` and `record_artifact_interaction_async` run the same bounded operations without blocking the caller's event loop. An explicit `ArtifactClient(api_key, endpoint)` can be used without initializing telemetry. No package publication or service rollout is performed by this change.

An optional `interaction={...}` passed to upload uses the keyword arguments of `record_interaction`, excluding artifact and version IDs. If linking fails after successful upload, `ArtifactInteractionError.artifact` retains the uploaded reference and `.interaction` retains the exact validated context and source event. Retry with `client.record_interaction(**error.interaction)`; context inheritance is disabled on this saved payload, so another active span cannot change the event. Retry the interaction; do not mint a new upload key. The client only sends upload credentials to the expected service URL and refuses redirects.
