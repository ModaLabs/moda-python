from __future__ import annotations

import os
import subprocess
from contextlib import contextmanager
from typing import Dict, Iterator, Mapping, Optional, Sequence, Union

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode, Span, Tracer

from traceloop.sdk.conversation import get_conversation_id, get_user_id
from traceloop.sdk.tracing.context_manager import get_tracer

OpenClawAttrValue = Union[str, int, float, bool]
DEFAULT_MODA_ENDPOINT = "https://moda-ingest.modas.workers.dev/v1/traces"


def _strip_otlp_signal_path(endpoint: str) -> str:
    cleaned = endpoint.strip().rstrip("/")
    for suffix in ("/v1/traces", "/v1/metrics", "/v1/logs"):
        if cleaned.endswith(suffix):
            return cleaned[: -len(suffix)]
    return cleaned


def _resolve_api_key(api_key: Optional[str]) -> str:
    resolved = api_key or os.getenv("MODA_API_KEY") or os.getenv("TRACELOOP_API_KEY")
    if not resolved:
        raise ValueError(
            "MODA_API_KEY is required for OpenClaw telemetry helpers "
            "(pass api_key or set MODA_API_KEY)"
        )
    return resolved


def _resolve_endpoint(endpoint: Optional[str]) -> str:
    resolved = (
        endpoint
        or os.getenv("MODA_BASE_URL")
        or os.getenv("TRACELOOP_BASE_URL")
        or DEFAULT_MODA_ENDPOINT
    )
    return _strip_otlp_signal_path(resolved)


def _encode_otel_headers(headers: Mapping[str, str]) -> str:
    return ",".join(f"{key}={value}" for key, value in headers.items())


@contextmanager
def _openclaw_tracer() -> Iterator[Tracer]:
    fallback_tracer = trace.get_tracer("moda.openclaw")
    try:
        tracer_context = get_tracer()
    except Exception:
        # Fallback for callers that use OpenClaw helpers before moda.init().
        yield fallback_tracer
        return

    try:
        moda_tracer = tracer_context.__enter__()
    except Exception:
        # Fallback when entering the Moda tracer context fails.
        yield fallback_tracer
        return

    try:
        yield moda_tracer
    except BaseException as exc:
        suppress = tracer_context.__exit__(type(exc), exc, exc.__traceback__)
        if not suppress:
            raise
    else:
        tracer_context.__exit__(None, None, None)


def get_openclaw_otel_config(
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None,
    service_name: str = "openclaw",
    enable_traces: bool = True,
    enable_metrics: bool = True,
    enable_logs: bool = True,
    additional_headers: Optional[Mapping[str, str]] = None,
) -> Dict[str, object]:
    """Build OpenClaw diagnostics config that exports OTLP data to Moda."""
    resolved_api_key = _resolve_api_key(api_key)
    resolved_endpoint = _resolve_endpoint(endpoint)

    headers: Dict[str, str] = {
        "Authorization": f"Bearer {resolved_api_key}",
        "Content-Type": "application/x-protobuf",
    }
    if additional_headers:
        headers.update(additional_headers)

    return {
        "plugins": {
            "allow": ["diagnostics-otel"],
            "entries": {
                "diagnostics-otel": {"enabled": True},
            },
        },
        "diagnostics": {
            "enabled": True,
            "otel": {
                "enabled": True,
                "endpoint": resolved_endpoint,
                "protocol": "http/protobuf",
                "serviceName": service_name,
                "traces": enable_traces,
                "metrics": enable_metrics,
                "logs": enable_logs,
                "headers": headers,
            },
        },
    }


def get_openclaw_env(
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None,
    service_name: str = "openclaw",
    enable_traces: bool = True,
    enable_metrics: bool = True,
    enable_logs: bool = True,
    additional_headers: Optional[Mapping[str, str]] = None,
    extra_env: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    """Build OTEL environment variables for OpenClaw runtime processes."""
    config = get_openclaw_otel_config(
        api_key=api_key,
        endpoint=endpoint,
        service_name=service_name,
        enable_traces=enable_traces,
        enable_metrics=enable_metrics,
        enable_logs=enable_logs,
        additional_headers=additional_headers,
    )
    otel = config["diagnostics"]["otel"]  # type: ignore[index]
    headers = otel["headers"]

    env = {
        "OTEL_EXPORTER_OTLP_ENDPOINT": str(otel["endpoint"]),
        "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf",
        "OTEL_EXPORTER_OTLP_HEADERS": _encode_otel_headers(headers),
        "OTEL_SERVICE_NAME": str(otel["serviceName"]),
        "OTEL_TRACES_EXPORTER": "otlp" if bool(otel["traces"]) else "none",
        "OTEL_METRICS_EXPORTER": "otlp" if bool(otel["metrics"]) else "none",
        "OTEL_LOGS_EXPORTER": "otlp" if bool(otel["logs"]) else "none",
    }

    if extra_env:
        env.update(extra_env)

    return env


@contextmanager
def trace_openclaw_operation(
    operation: str,
    *,
    conversation_id: Optional[str] = None,
    user_id: Optional[str] = None,
    attributes: Optional[Mapping[str, OpenClawAttrValue]] = None,
) -> Iterator[Span]:
    """Create a Moda span around an OpenClaw operation."""
    with _openclaw_tracer() as tracer:
        span = tracer.start_span(name=f"openclaw.{operation}")
        span.set_attribute("llm.vendor", "openclaw")
        span.set_attribute("llm.request.type", operation)

        resolved_conversation_id = conversation_id or get_conversation_id()
        resolved_user_id = user_id or get_user_id()
        if resolved_conversation_id:
            span.set_attribute("moda.conversation_id", resolved_conversation_id)
        if resolved_user_id:
            span.set_attribute("moda.user_id", resolved_user_id)

        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)

        try:
            yield span
            span.set_status(Status(StatusCode.OK))
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise
        finally:
            span.end()


def run_openclaw_cli(
    args: Sequence[str],
    *,
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None,
    service_name: str = "openclaw",
    conversation_id: Optional[str] = None,
    user_id: Optional[str] = None,
    operation: str = "cli.run",
    cwd: Optional[str] = None,
    timeout: Optional[float] = None,
    check: bool = False,
    capture_output: bool = True,
    text: bool = True,
    env: Optional[Mapping[str, str]] = None,
    command_prefix: Optional[Sequence[str]] = None,
) -> subprocess.CompletedProcess:
    """Run the `openclaw` CLI with Moda OTEL env and traced operation span."""
    if not args:
        raise ValueError("args must contain at least one OpenClaw CLI argument")

    prefix = list(command_prefix) if command_prefix is not None else ["openclaw"]
    command = [*prefix, *args]
    process_env = dict(os.environ)
    process_env.update(
        get_openclaw_env(
            api_key=api_key,
            endpoint=endpoint,
            service_name=service_name,
        )
    )
    if env:
        process_env.update(env)

    with trace_openclaw_operation(
        operation,
        conversation_id=conversation_id,
        user_id=user_id,
        attributes={
            "openclaw.mode": "cli",
            "openclaw.command": " ".join(command),
        },
    ) as span:
        completed = subprocess.run(
            command,
            cwd=cwd,
            timeout=timeout,
            check=check,
            capture_output=capture_output,
            text=text,
            env=process_env,
        )
        span.set_attribute("openclaw.exit_code", completed.returncode)
        if capture_output and isinstance(completed.stdout, str):
            span.set_attribute("openclaw.stdout.length", len(completed.stdout))
        if capture_output and isinstance(completed.stderr, str):
            span.set_attribute("openclaw.stderr.length", len(completed.stderr))
        return completed
