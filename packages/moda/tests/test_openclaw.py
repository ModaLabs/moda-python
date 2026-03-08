from __future__ import annotations

from contextlib import contextmanager
import subprocess

from opentelemetry import trace
from opentelemetry.trace import StatusCode

from traceloop.sdk.openclaw import (
    get_openclaw_env,
    get_openclaw_otel_config,
    run_openclaw_cli,
    trace_openclaw_operation,
)


class _FakeSpan:
    def __init__(self, name: str):
        self.name = name
        self.attributes = {}
        self.status = None
        self.exceptions = []
        self.ended = False

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def set_status(self, status):
        self.status = status

    def record_exception(self, exc):
        self.exceptions.append(exc)

    def end(self):
        self.ended = True


class _FakeTracer:
    def __init__(self):
        self.spans = []

    def start_span(self, name: str):
        span = _FakeSpan(name)
        self.spans.append(span)
        return span


def test_get_openclaw_otel_config():
    config = get_openclaw_otel_config(
        api_key="moda_test_key",
        endpoint="https://moda-ingest.modas.workers.dev/v1/traces",
        service_name="openclaw-gateway",
    )

    assert config["plugins"]["allow"] == ["diagnostics-otel"]
    assert config["plugins"]["entries"]["diagnostics-otel"]["enabled"] is True
    assert config["diagnostics"]["otel"]["endpoint"] == "https://moda-ingest.modas.workers.dev"
    assert config["diagnostics"]["otel"]["headers"]["Authorization"] == "Bearer moda_test_key"
    assert config["diagnostics"]["otel"]["serviceName"] == "openclaw-gateway"


def test_get_openclaw_env():
    env = get_openclaw_env(
        api_key="moda_test_key",
        endpoint="https://moda-ingest.modas.workers.dev/v1/traces",
    )

    assert env["OTEL_EXPORTER_OTLP_ENDPOINT"] == "https://moda-ingest.modas.workers.dev"
    assert env["OTEL_EXPORTER_OTLP_PROTOCOL"] == "http/protobuf"
    assert "Authorization=Bearer moda_test_key" in env["OTEL_EXPORTER_OTLP_HEADERS"]
    assert env["OTEL_TRACES_EXPORTER"] == "otlp"
    assert env["OTEL_METRICS_EXPORTER"] == "otlp"
    assert env["OTEL_LOGS_EXPORTER"] == "otlp"


def test_trace_openclaw_operation(monkeypatch):
    tracer = _FakeTracer()

    @contextmanager
    def _fake_get_tracer():
        yield tracer

    monkeypatch.setattr("traceloop.sdk.openclaw.get_tracer", _fake_get_tracer)
    monkeypatch.setattr("traceloop.sdk.openclaw.get_conversation_id", lambda: "conv-global")
    monkeypatch.setattr("traceloop.sdk.openclaw.get_user_id", lambda: "user-global")

    with trace_openclaw_operation(
        "gateway.request",
        attributes={"openclaw.transport": "http"},
    ):
        pass

    span = tracer.spans[0]
    assert span.name == "openclaw.gateway.request"
    assert span.attributes["llm.vendor"] == "openclaw"
    assert span.attributes["llm.request.type"] == "gateway.request"
    assert span.attributes["moda.conversation_id"] == "conv-global"
    assert span.attributes["moda.user_id"] == "user-global"
    assert span.attributes["openclaw.transport"] == "http"
    assert span.status.status_code == StatusCode.OK
    assert span.ended is True


def test_run_openclaw_cli(monkeypatch):
    tracer = _FakeTracer()
    run_calls = {}

    @contextmanager
    def _fake_get_tracer():
        yield tracer

    def _fake_run(command, **kwargs):
        run_calls["command"] = command
        run_calls["kwargs"] = kwargs
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr("traceloop.sdk.openclaw.get_tracer", _fake_get_tracer)
    monkeypatch.setattr("traceloop.sdk.openclaw.subprocess.run", _fake_run)

    result = run_openclaw_cli(
        ["status"],
        api_key="moda_test_key",
        endpoint="https://moda-ingest.modas.workers.dev/v1/traces",
    )

    assert result.returncode == 0
    assert run_calls["command"][0] == "openclaw"
    assert run_calls["kwargs"]["env"]["OTEL_EXPORTER_OTLP_ENDPOINT"] == "https://moda-ingest.modas.workers.dev"
    assert tracer.spans[0].attributes["openclaw.exit_code"] == 0


def test_trace_openclaw_operation_falls_back_when_moda_not_initialized(monkeypatch):
    tracer = _FakeTracer()

    monkeypatch.setattr(
        "traceloop.sdk.openclaw.get_tracer",
        lambda: (_ for _ in ()).throw(RuntimeError("not initialized")),
    )
    monkeypatch.setattr(trace, "get_tracer", lambda _: tracer)

    with trace_openclaw_operation("gateway.request"):
        pass

    assert tracer.spans[0].name == "openclaw.gateway.request"
