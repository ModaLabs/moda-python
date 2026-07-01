"""Moda SDK - LLM Observability with Automatic Conversation Threading.

This SDK provides automatic instrumentation for LLM calls with conversation
threading support.

Example:
    import moda
    moda.init("moda_xxx")
    # All LLM calls are now automatically tracked with conversation threading
"""

import os
import sys
from pathlib import Path

from typing import Callable, Dict, List, Optional, Set, Union
from colorama import Fore
from opentelemetry.sdk.trace import SpanProcessor, ReadableSpan
from opentelemetry.sdk.trace.sampling import Sampler
from opentelemetry.sdk.trace.export import SpanExporter
from opentelemetry.sdk.metrics.export import MetricExporter
from opentelemetry.sdk._logs.export import LogExporter
from opentelemetry.sdk.resources import SERVICE_NAME
from opentelemetry.propagators.textmap import TextMapPropagator
from opentelemetry.util.re import parse_env_headers

from traceloop.sdk.images.image_uploader import ImageUploader
from traceloop.sdk.metrics.metrics import MetricsWrapper
from traceloop.sdk.logging.logging import LoggerWrapper
from traceloop.sdk.instruments import Instruments
from traceloop.sdk.config import (
    is_content_tracing_enabled,
    is_tracing_enabled,
    is_metrics_enabled,
    is_logging_enabled,
)
from traceloop.sdk.fetcher import Fetcher
from traceloop.sdk.tracing.tracing import (
    TracerWrapper,
    set_association_properties,
    set_external_prompt_tracing_context,
)
from traceloop.sdk.client.client import Client
from traceloop.sdk.associations.associations import AssociationProperty as AssociationProperty
from traceloop.sdk.openclaw import (
    get_openclaw_env as _get_openclaw_env,
    get_openclaw_otel_config as _get_openclaw_otel_config,
    run_openclaw_cli as _run_openclaw_cli,
    trace_openclaw_operation,
)

# Import conversation and context modules
from traceloop.sdk.context import (
    set_conversation_id,
    set_user_id,
    set_environment,
    set_conversation_id_value,
    set_user_id_value,
    set_environment_value,
)
from traceloop.sdk.conversation import (
    compute_conversation_id,
    get_conversation_id,
    get_user_id,
    get_environment,
)

# Re-export for convenience
__all__ = [
    "Moda",
    "init",
    "flush",
    "set_conversation_id",
    "set_user_id",
    "set_environment",
    "set_conversation_id_value",
    "set_user_id_value",
    "set_environment_value",
    "get_conversation_id",
    "get_user_id",
    "get_environment",
    "compute_conversation_id",
    "set_association_properties",
    "AssociationProperty",
    "Instruments",
    "get_openclaw_otel_config",
    "get_openclaw_env",
    "trace_openclaw_operation",
    "run_openclaw_cli",
]

# Default Moda endpoint
DEFAULT_ENDPOINT = "https://moda-ingest.modas.workers.dev/v1/traces"


class Moda:
    """Moda SDK for LLM observability with automatic conversation threading."""

    AUTO_CREATED_KEY_PATH = str(
        Path.home() / ".cache" / "moda" / "auto_created_key"
    )
    AUTO_CREATED_URL = str(Path.home() / ".cache" / "moda" / "auto_created_url")

    __tracer_wrapper: TracerWrapper
    __fetcher: Optional[Fetcher] = None
    __app_name: Optional[str] = None
    __client: Optional[Client] = None

    @staticmethod
    def init(
        api_key: Optional[str] = None,
        app_name: str = sys.argv[0],
        api_endpoint: str = DEFAULT_ENDPOINT,
        enabled: bool = True,
        headers: Dict[str, str] = {},
        disable_batch=False,
        exporter: Optional[SpanExporter] = None,
        metrics_exporter: MetricExporter = None,
        metrics_headers: Dict[str, str] = None,
        logging_exporter: LogExporter = None,
        logging_headers: Dict[str, str] = None,
        processor: Optional[Union[SpanProcessor, List[SpanProcessor]]] = None,
        propagator: TextMapPropagator = None,
        sampler: Optional[Sampler] = None,
        should_enrich_metrics: bool = True,
        resource_attributes: dict = {},
        environment: Optional[str] = None,
        instruments: Optional[Set[Instruments]] = None,
        block_instruments: Optional[Set[Instruments]] = None,
        image_uploader: Optional[ImageUploader] = None,
        span_postprocess_callback: Optional[Callable[[ReadableSpan], None]] = None,
    ) -> Optional[Client]:
        """Initialize Moda SDK.

        Args:
            api_key: Your Moda API key (or set MODA_API_KEY env var).
            app_name: Name of your application for identification.
            api_endpoint: Moda ingest endpoint (default: https://moda-ingest.modas.workers.dev/v1/traces).
            enabled: Whether to enable instrumentation.
            headers: Additional headers for the exporter.
            disable_batch: If True, send spans immediately instead of batching.
            exporter: Custom span exporter.
            metrics_exporter: Custom metrics exporter.
            metrics_headers: Headers for metrics exporter.
            logging_exporter: Custom logging exporter.
            logging_headers: Headers for logging exporter.
            processor: Custom span processor(s).
            propagator: Custom trace context propagator.
            sampler: Custom sampler.
            should_enrich_metrics: Whether to enrich metrics with additional data.
            resource_attributes: Additional resource attributes.
            environment: Deployment environment name (e.g. 'development', 'staging',
                'production'). Resolved as explicit arg > MODA_ENVIRONMENT env var >
                'production'. Stamped on the resource as both 'moda.environment' and
                'deployment.environment'.
            instruments: Set of instruments to enable.
            block_instruments: Set of instruments to disable.
            image_uploader: Custom image uploader.
            span_postprocess_callback: Callback for post-processing spans.

        Returns:
            Client instance if using Moda cloud, None otherwise.
        """
        if not enabled:
            TracerWrapper.set_disabled(True)
            print(
                Fore.YELLOW
                + "Moda instrumentation is disabled via init flag"
                + Fore.RESET
            )
            return

        # Check environment variables (MODA_ takes precedence, fall back to TRACELOOP_)
        api_endpoint = (
            os.getenv("MODA_BASE_URL")
            or os.getenv("TRACELOOP_BASE_URL")
            or api_endpoint
        )
        api_key = (
            os.getenv("MODA_API_KEY")
            or os.getenv("TRACELOOP_API_KEY")
            or api_key
        )
        Moda.__app_name = app_name

        if not is_tracing_enabled():
            print(Fore.YELLOW + "Tracing is disabled" + Fore.RESET)
            return

        enable_content_tracing = is_content_tracing_enabled()

        if exporter or processor:
            print(Fore.GREEN + "Moda exporting traces to a custom exporter")

        headers = (
            os.getenv("MODA_HEADERS")
            or os.getenv("TRACELOOP_HEADERS")
            or headers
        )

        if isinstance(headers, str):
            headers = parse_env_headers(headers)

        if (
            not exporter
            and not processor
            and api_endpoint == DEFAULT_ENDPOINT
            and not api_key
        ):
            print(
                Fore.RED
                + "Error: Missing Moda API key."
                + " Set the MODA_API_KEY environment variable or pass api_key to init()"
            )
            print(Fore.RESET)
            return

        if not exporter and not processor and headers:
            print(
                Fore.GREEN
                + f"Moda exporting traces to {api_endpoint}, authenticating with custom headers"
            )

        if api_key and not exporter and not processor and not headers:
            print(
                Fore.GREEN
                + f"Moda exporting traces to {api_endpoint}"
            )
            headers = {
                "Authorization": f"Bearer {api_key}",
            }

        print(Fore.RESET)

        # Tracer init
        resource_attributes.update({SERVICE_NAME: app_name})

        # Resolve environment. Precedence, mirroring the Node SDK (which
        # defaults to 'production' and stamps the explicit value verbatim):
        #   explicit arg > MODA_ENVIRONMENT env var
        #     > caller-provided resource_attributes > 'production'
        # "Explicit arg" means the caller passed anything other than None. An
        # explicitly-provided value always wins and is stamped verbatim — even
        # an empty string — so it never silently inherits MODA_ENVIRONMENT or
        # the default (the explicit argument has priority). Only when the arg is
        # None do the ambient sources apply, where a blank value means "unset"
        # (os.getenv returns None when unset; empty strings are falsy). Both
        # 'moda.environment' and 'deployment.environment' are stamped so the
        # backend maps them identically.
        if environment is not None:
            resolved_environment = environment
        else:
            resolved_environment = (
                os.getenv("MODA_ENVIRONMENT")
                or resource_attributes.get("moda.environment")
                or resource_attributes.get("deployment.environment")
                or "production"
            )
        resource_attributes.update(
            {
                "moda.environment": resolved_environment,
                "deployment.environment": resolved_environment,
            }
        )

        TracerWrapper.set_static_params(
            resource_attributes, enable_content_tracing, api_endpoint, headers
        )
        Moda.__tracer_wrapper = TracerWrapper(
            disable_batch=disable_batch,
            processor=processor,
            propagator=propagator,
            exporter=exporter,
            sampler=sampler,
            should_enrich_metrics=should_enrich_metrics,
            image_uploader=image_uploader or ImageUploader(api_endpoint, api_key),
            instruments=instruments,
            block_instruments=block_instruments,
            span_postprocess_callback=span_postprocess_callback,
        )

        metrics_disabled_by_config = not is_metrics_enabled()
        has_custom_spans_pipeline = processor or exporter
        custom_trace_without_custom_metrics = has_custom_spans_pipeline and not metrics_exporter
        explicit_metrics_endpoint = (
            os.getenv("MODA_METRICS_ENDPOINT")
            or os.getenv("TRACELOOP_METRICS_ENDPOINT")
        )
        metrics_disabled_for_default_trace_endpoint = (
            api_endpoint.strip().rstrip("/") == DEFAULT_ENDPOINT.rstrip("/")
            and not explicit_metrics_endpoint
            and not metrics_exporter
        )

        if (
            metrics_disabled_by_config
            or custom_trace_without_custom_metrics
            or metrics_disabled_for_default_trace_endpoint
        ):
            print(Fore.YELLOW + "Metrics are disabled" + Fore.RESET)
        else:
            metrics_endpoint = (
                explicit_metrics_endpoint
                or api_endpoint
            )
            metrics_headers = (
                os.getenv("MODA_METRICS_HEADERS")
                or os.getenv("TRACELOOP_METRICS_HEADERS")
                or metrics_headers
                or headers
            )
            if metrics_exporter or processor:
                print(Fore.GREEN + "Moda exporting metrics to a custom exporter")

            MetricsWrapper.set_static_params(
                resource_attributes, metrics_endpoint, metrics_headers
            )
            Moda.__metrics_wrapper = MetricsWrapper(exporter=metrics_exporter)

        if is_logging_enabled() and (logging_exporter or not exporter):
            logging_endpoint = (
                os.getenv("MODA_LOGGING_ENDPOINT")
                or os.getenv("TRACELOOP_LOGGING_ENDPOINT")
                or api_endpoint
            )
            logging_headers = (
                os.getenv("MODA_LOGGING_HEADERS")
                or os.getenv("TRACELOOP_LOGGING_HEADERS")
                or logging_headers
                or headers
            )
            if logging_exporter or processor:
                print(Fore.GREEN + "Moda exporting logs to a custom exporter")

            LoggerWrapper.set_static_params(
                resource_attributes, logging_endpoint, logging_headers
            )
            Moda.__logger_wrapper = LoggerWrapper(exporter=logging_exporter)

        # Store client reference for flush
        Moda.__client = Client(
            api_key=api_key, app_name=app_name, api_endpoint=api_endpoint
        ) if api_key else None

        return Moda.__client

    @staticmethod
    def set_association_properties(properties: dict) -> None:
        """Set association properties for the current context."""
        set_association_properties(properties)

    @staticmethod
    def set_prompt(template: str, variables: dict, version: int):
        """Set external prompt tracing context."""
        set_external_prompt_tracing_context(template, variables, version)

    @staticmethod
    def flush() -> None:
        """Force flush all pending spans."""
        if hasattr(Moda, "_Moda__tracer_wrapper") and Moda.__tracer_wrapper:
            Moda.__tracer_wrapper.flush()

    @staticmethod
    def get_openclaw_otel_config(
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        service_name: str = "openclaw",
        enable_traces: bool = True,
        enable_metrics: bool = True,
        enable_logs: bool = True,
        additional_headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, object]:
        """Build OpenClaw diagnostics config for OTLP export to Moda."""
        return _get_openclaw_otel_config(
            api_key=api_key,
            endpoint=endpoint,
            service_name=service_name,
            enable_traces=enable_traces,
            enable_metrics=enable_metrics,
            enable_logs=enable_logs,
            additional_headers=additional_headers,
        )

    @staticmethod
    def get_openclaw_env(
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        service_name: str = "openclaw",
        enable_traces: bool = True,
        enable_metrics: bool = True,
        enable_logs: bool = True,
        additional_headers: Optional[Dict[str, str]] = None,
        extra_env: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """Build OTEL environment variables for OpenClaw runtime processes."""
        return _get_openclaw_env(
            api_key=api_key,
            endpoint=endpoint,
            service_name=service_name,
            enable_traces=enable_traces,
            enable_metrics=enable_metrics,
            enable_logs=enable_logs,
            additional_headers=additional_headers,
            extra_env=extra_env,
        )

    @staticmethod
    def run_openclaw_cli(
        args: List[str],
        **kwargs,
    ) -> object:
        """Run OpenClaw CLI with Moda OTEL env and tracing span."""
        return _run_openclaw_cli(args, **kwargs)

    @staticmethod
    def get_default_span_processor(
        disable_batch: bool = False,
        api_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        exporter: Optional[SpanExporter] = None
    ) -> SpanProcessor:
        """Create and return the default Moda span processor.

        This allows combining the default processor with custom processors.

        Args:
            disable_batch: If True, uses SimpleSpanProcessor, otherwise BatchSpanProcessor.
            api_endpoint: The endpoint URL for the exporter.
            headers: Headers for the exporter.
            exporter: Custom exporter to use.

        Returns:
            SpanProcessor: The default Moda span processor.
        """
        from traceloop.sdk.tracing.tracing import get_default_span_processor
        if headers is None:
            if api_key is None:
                api_key = os.getenv("MODA_API_KEY") or os.getenv("TRACELOOP_API_KEY")
            headers = {
                "Authorization": f"Bearer {api_key}",
            }
        if api_endpoint is None:
            api_endpoint = (
                os.getenv("MODA_BASE_URL")
                or os.getenv("TRACELOOP_BASE_URL")
                or DEFAULT_ENDPOINT
            )
        return get_default_span_processor(disable_batch, api_endpoint, headers, exporter)

    @staticmethod
    def get():
        """Return the shared SDK client instance.

        Returns:
            Client: The Moda client instance.

        Raises:
            Exception: If init() has not been called.
        """
        if not Moda.__client:
            raise Exception(
                "Client not initialized, you should call moda.init() first. "
                "Make sure you have provided an API key."
            )
        return Moda.__client


# Convenience function for simpler API
def init(
    api_key: Optional[str] = None,
    app_name: str = sys.argv[0],
    endpoint: Optional[str] = None,
    **kwargs
) -> Optional[Client]:
    """Initialize Moda SDK.

    This is a convenience wrapper around Moda.init().

    Example:
        import moda
        moda.init("moda_xxx")

    Args:
        api_key: Your Moda API key.
        app_name: Name of your application.
        endpoint: Custom endpoint (optional).
        **kwargs: Additional arguments passed to Moda.init().

    Returns:
        Client instance if successful.
    """
    if endpoint:
        kwargs["api_endpoint"] = endpoint
    return Moda.init(api_key=api_key, app_name=app_name, **kwargs)


def flush() -> None:
    """Force flush all pending spans.

    Example:
        moda.flush()  # Ensure all spans are sent before exit
    """
    Moda.flush()


def get_openclaw_otel_config(
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None,
    service_name: str = "openclaw",
    enable_traces: bool = True,
    enable_metrics: bool = True,
    enable_logs: bool = True,
    additional_headers: Optional[Dict[str, str]] = None,
) -> Dict[str, object]:
    """Build OpenClaw diagnostics config for OTLP export to Moda."""
    return Moda.get_openclaw_otel_config(
        api_key=api_key,
        endpoint=endpoint,
        service_name=service_name,
        enable_traces=enable_traces,
        enable_metrics=enable_metrics,
        enable_logs=enable_logs,
        additional_headers=additional_headers,
    )


def get_openclaw_env(
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None,
    service_name: str = "openclaw",
    enable_traces: bool = True,
    enable_metrics: bool = True,
    enable_logs: bool = True,
    additional_headers: Optional[Dict[str, str]] = None,
    extra_env: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Build OTEL environment variables for OpenClaw runtime processes."""
    return Moda.get_openclaw_env(
        api_key=api_key,
        endpoint=endpoint,
        service_name=service_name,
        enable_traces=enable_traces,
        enable_metrics=enable_metrics,
        enable_logs=enable_logs,
        additional_headers=additional_headers,
        extra_env=extra_env,
    )


def run_openclaw_cli(
    args: List[str],
    **kwargs,
) -> object:
    """Run OpenClaw CLI with Moda OTEL env and tracing span."""
    return Moda.run_openclaw_cli(args, **kwargs)


# Keep backward compatibility with Traceloop
Traceloop = Moda
