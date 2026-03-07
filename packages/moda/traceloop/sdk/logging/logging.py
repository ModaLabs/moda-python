import logging
from typing import Dict, Optional, Any, cast
from urllib.parse import urlparse

from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
    OTLPLogExporter as GRPCExporter,
)
from opentelemetry.exporter.otlp.proto.http._log_exporter import (
    OTLPLogExporter as HTTPExporter,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk._logs.export import LogExporter, BatchLogRecordProcessor
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler

from opentelemetry.instrumentation.logging import LoggingInstrumentor


class LoggerWrapper(object):
    resource_attributes: Dict[Any, Any] = {}
    endpoint: Optional[str] = None
    headers: Dict[str, str] = {}
    __logging_exporter: Optional[LogExporter] = None
    __logging_provider: Optional[LoggerProvider] = None

    def __new__(cls, exporter: Optional[LogExporter] = None) -> "LoggerWrapper":
        if not hasattr(cls, "instance"):
            obj = cls.instance = super(LoggerWrapper, cls).__new__(cls)
            if not LoggerWrapper.endpoint:
                return obj

            obj.__logging_exporter = (
                exporter
                if exporter
                else init_logging_exporter(LoggerWrapper.endpoint, LoggerWrapper.headers)
            )
            obj.__logging_provider = init_logging_provider(
                obj.__logging_exporter, LoggerWrapper.resource_attributes
            )
            LoggingInstrumentor().instrument(set_logging_format=True)

        return cls.instance

    @staticmethod
    def set_static_params(
        resource_attributes: dict,
        endpoint: str,
        headers: Dict[str, str],
    ) -> None:
        LoggerWrapper.resource_attributes = resource_attributes
        LoggerWrapper.endpoint = endpoint
        LoggerWrapper.headers = headers


def init_logging_exporter(endpoint: str, headers: Dict[str, str]) -> LogExporter:
    parsed = urlparse(endpoint.strip())
    if parsed.scheme.lower() in {"http", "https"}:
        return cast(
            LogExporter,
            HTTPExporter(
                endpoint=_normalize_http_signal_endpoint(endpoint, "logs"),
                headers=headers,
            ),
        )
    else:
        return cast(LogExporter, GRPCExporter(endpoint=endpoint.strip(), headers=headers))


def _normalize_http_signal_endpoint(endpoint: str, signal: str) -> str:
    """Normalize HTTP OTLP endpoints to avoid double-appending /v1/<signal>."""
    parsed = urlparse(endpoint.strip().rstrip("/"))
    path = parsed.path.rstrip("/")

    for suffix in ("/v1/traces", "/v1/metrics", "/v1/logs"):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break

    normalized_path = f"{path}/v1/{signal}" if path else f"/v1/{signal}"
    return parsed._replace(path=normalized_path).geturl()


def init_logging_provider(
    exporter: LogExporter, resource_attributes: Optional[Dict[Any, Any]] = None
) -> LoggerProvider:
    resource = (
        Resource.create(resource_attributes)
        if resource_attributes
        else Resource.create()
    )

    logger_provider = LoggerProvider(resource=resource)
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(exporter))

    logging_handler = LoggingHandler(level=logging.NOTSET, logger_provider=logger_provider)
    logging.basicConfig(level=logging.INFO, handlers=[logging_handler])

    return logger_provider
