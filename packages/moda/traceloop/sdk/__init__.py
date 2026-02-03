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

# Import conversation and context modules
from traceloop.sdk.context import (
    set_conversation_id,
    set_user_id,
    set_conversation_id_value,
    set_user_id_value,
)
from traceloop.sdk.conversation import (
    compute_conversation_id,
    get_conversation_id,
    get_user_id,
)

# Re-export for convenience
__all__ = [
    "Moda",
    "init",
    "flush",
    "set_conversation_id",
    "set_user_id",
    "set_conversation_id_value",
    "set_user_id_value",
    "get_conversation_id",
    "get_user_id",
    "compute_conversation_id",
    "set_association_properties",
    "AssociationProperty",
    "Instruments",
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

        if metrics_disabled_by_config or custom_trace_without_custom_metrics:
            print(Fore.YELLOW + "Metrics are disabled" + Fore.RESET)
        else:
            metrics_endpoint = (
                os.getenv("MODA_METRICS_ENDPOINT")
                or os.getenv("TRACELOOP_METRICS_ENDPOINT")
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


# Keep backward compatibility with Traceloop
Traceloop = Moda
