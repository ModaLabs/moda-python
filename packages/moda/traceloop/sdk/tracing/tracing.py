import atexit
import logging
import os
from urllib.parse import urlparse


from colorama import Fore
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter as HTTPExporter,
)
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter as GRPCExporter,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider, SpanProcessor, ReadableSpan
from opentelemetry.sdk.trace.sampling import Sampler
from opentelemetry.propagators.textmap import TextMapPropagator
from opentelemetry.propagate import set_global_textmap
from opentelemetry.sdk.trace.export import (
    SpanExporter,
    SimpleSpanProcessor,
    BatchSpanProcessor,
)
from opentelemetry.context import Context

from opentelemetry.trace import get_tracer_provider, ProxyTracerProvider, Span
from opentelemetry.context import get_value, attach, set_value
from opentelemetry.instrumentation.threading import ThreadingInstrumentor

from opentelemetry.semconv_ai import SpanAttributes
from traceloop.sdk.images.image_uploader import ImageUploader
from traceloop.sdk.instruments import Instruments
from traceloop.sdk.tracing.content_allow_list import ContentAllowList
from traceloop.sdk.utils import is_notebook
from traceloop.sdk.utils.package_check import is_package_installed
from typing import Callable, Dict, List, Optional, Set, Union
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_AGENT_NAME,
)


TRACER_NAME = "traceloop.tracer"
EXCLUDED_URLS = """
    iam.cloud.ibm.com,
    dataplatform.cloud.ibm.com,
    ml.cloud.ibm.com,
    api.openai.com,
    openai.azure.com,
    api.anthropic.com,
    api.cohere.ai,
    pinecone.io,
    traceloop.com,
    posthog.com,
    sentry.io,
    bedrock-runtime,
    sagemaker-runtime,
    googleapis.com,
    githubusercontent.com,
    openaipublic.blob.core.windows.net"""


# Attribute keys used to tag every span with the deployment environment. This
# mirrors the moda-node tee-in fix: the ingest worker reads span-level
# `moda.environment` (overriding the resource-level `deployment.environment`)
# and, critically, defaults to "production" when neither is present. When Moda
# tees its SpanProcessor into a *host* provider, that host provider's Resource
# does not carry Moda's `deployment.environment`, so without a span-level stamp
# every tee-in trace silently normalizes to production. We therefore stamp
# `moda.environment` at the span level, always, on the shared on_start path.
MODA_ENVIRONMENT_ATTRIBUTE = "moda.environment"
DEPLOYMENT_ENVIRONMENT_ATTRIBUTE = "deployment.environment"


class TracerProviderCoexistenceError(RuntimeError):
    """Raised when Moda cannot safely tee its SpanProcessor into the active
    global TracerProvider.

    Per the loud-fail contract, an unsafe tee-in must surface to the caller
    rather than being swallowed by a library-level ``logging.error`` + silent
    ``return``. The most common trigger is a host that registered a custom
    global provider which does not expose ``add_span_processor`` (e.g. a
    read-only or NoOp-style provider), meaning Moda has no seam to attach to
    without registering a forbidden second global provider.
    """


def resolve_moda_environment() -> Optional[str]:
    """Resolve the deployment environment to stamp on every span.

    Resolution order (first non-empty wins):
      1. The ``deployment.environment`` static resource attribute configured on
         ``TracerWrapper`` (set from ``resource_attributes`` at init).
      2. The ``MODA_ENVIRONMENT`` environment variable.
      3. The ``TRACELOOP_ENVIRONMENT`` environment variable (back-compat).

    Returns ``None`` when nothing is configured so callers can decide whether to
    stamp. We intentionally do NOT fabricate a "production" default here: the
    whole point of the span-level stamp is to stop tee-in traces from being
    *defaulted* to production in ingest. If the caller has not declared an
    environment, we leave the span unstamped and let ingest apply its documented
    default rather than lying about the environment.
    """
    resource_attributes = getattr(TracerWrapper, "resource_attributes", None) or {}
    env = resource_attributes.get(DEPLOYMENT_ENVIRONMENT_ATTRIBUTE)
    if not env:
        env = os.getenv("MODA_ENVIRONMENT") or os.getenv("TRACELOOP_ENVIRONMENT")
    if env is None:
        return None
    env = str(env).strip()
    return env or None


class TracerWrapper(object):
    resource_attributes: dict = {}
    enable_content_tracing: bool = True
    endpoint: str = None
    headers: Dict[str, str] = {}
    __tracer_provider: TracerProvider = None
    __image_uploader: ImageUploader = None
    __disabled: bool = False

    def __new__(
        cls,
        disable_batch=False,
        processor: Optional[Union[SpanProcessor, List[SpanProcessor]]] = None,
        propagator: TextMapPropagator = None,
        exporter: SpanExporter = None,
        sampler: Optional[Sampler] = None,
        should_enrich_metrics: bool = True,
        instruments: Optional[Set[Instruments]] = None,
        block_instruments: Optional[Set[Instruments]] = None,
        image_uploader: ImageUploader = None,
        span_postprocess_callback: Optional[Callable[[ReadableSpan], None]] = None,
    ) -> "TracerWrapper":
        if not hasattr(cls, "instance"):
            obj = cls.instance = super(TracerWrapper, cls).__new__(cls)
            if not TracerWrapper.endpoint:
                return obj

            obj.__image_uploader = image_uploader
            obj.__resource = Resource.create(TracerWrapper.resource_attributes)
            try:
                obj.__tracer_provider = init_tracer_provider(
                    resource=obj.__resource, sampler=sampler
                )
            except TracerProviderCoexistenceError:
                # Loud-fail contract: do not leave a half-built singleton behind
                # after an unsafe tee-in. Drop the instance so a fixed re-init
                # can succeed, then re-raise to the caller (Traceloop.init).
                if hasattr(cls, "instance"):
                    del cls.instance
                raise

            # Track Moda's own processors and the provider they are attached to,
            # so the init-order race (a real host provider registering AFTER
            # Moda) can be reconciled on the first span rather than snapshotted
            # once at init. See _reattach_to_host_provider_if_changed.
            obj.__moda_processors: List[SpanProcessor] = []
            obj.__attached_provider = obj.__tracer_provider
            obj.__reattach_checked = False

            # Handle multiple processors case
            if processor is not None and isinstance(processor, list):
                obj.__spans_processors = []
                for proc in processor:
                    original_on_start = proc.on_start

                    def chained_on_start(
                        span, parent_context=None, orig=original_on_start
                    ):
                        if orig:
                            orig(span, parent_context)
                        obj._span_processor_on_start(span, parent_context)

                    proc.on_start = chained_on_start
                    obj.__spans_processors.append(proc)
                    obj.__moda_processors.append(proc)

                    obj.__tracer_provider.add_span_processor(proc)

            # Handle single processor case (backward compatibility)
            elif processor is not None:
                obj.__spans_processor: SpanProcessor = processor
                original_on_start = obj.__spans_processor.on_start

                def chained_on_start(span, parent_context=None, orig=original_on_start):
                    obj._span_processor_on_start(span, parent_context)
                    if orig:
                        orig(span, parent_context)

                obj.__spans_processor.on_start = chained_on_start

                obj.__moda_processors.append(obj.__spans_processor)
                obj.__tracer_provider.add_span_processor(obj.__spans_processor)

            # Handle default processor case
            else:
                obj.__spans_processor = get_default_span_processor(
                    disable_batch=disable_batch, exporter=exporter
                )

                if span_postprocess_callback:
                    # Create a wrapper that calls both the custom and original methods
                    original_on_end = obj.__spans_processor.on_end

                    def wrapped_on_end(span):
                        # Call the custom on_end first
                        span_postprocess_callback(span)
                        # Then call the original to ensure normal processing
                        original_on_end(span)

                    obj.__spans_processor.on_end = wrapped_on_end

                obj.__spans_processor.on_start = obj._span_processor_on_start
                obj.__moda_processors.append(obj.__spans_processor)
                obj.__tracer_provider.add_span_processor(obj.__spans_processor)

            if propagator:
                set_global_textmap(propagator)

            # this makes sure otel context is propagated so we always want it
            ThreadingInstrumentor().instrument()

            init_instrumentations(
                should_enrich_metrics,
                image_uploader.aupload_base64_image,
                instruments,
                block_instruments,
            )

            obj.__content_allow_list = ContentAllowList()

            # Force flushes for debug environments (e.g. local development)
            atexit.register(obj.exit_handler)

        return cls.instance

    def exit_handler(self):
        self.flush()

    def _reattach_to_host_provider_if_changed(self) -> None:
        """Reconcile the init-order race: if a real host provider registered
        after Moda initialized, move Moda's processors onto it.

        Runs at most once (on the first span). This covers silent mode (1): when
        Moda initializes before the host observability SDK, Moda first creates
        its own provider; once the host registers its real global provider, we
        detach nothing from our own provider (spans in flight keep exporting) but
        ADD our processors to the now-active host provider so subsequent host
        spans are also tee'd. We never call set_tracer_provider here — we only
        ever add_span_processor onto the host's existing global.
        """
        if self.__reattach_checked:
            return
        self.__reattach_checked = True

        try:
            host_provider = resolve_host_provider()
        except Exception:  # pragma: no cover - defensive; never break span start
            return

        if host_provider is None:
            return
        if host_provider is self.__attached_provider:
            return

        # A real host provider appeared after init. Tee our processors into it
        # so host-created spans are captured too (single global, no second set).
        for proc in self.__moda_processors:
            try:
                host_provider.add_span_processor(proc)
            except Exception:  # pragma: no cover - host provider misbehaving
                continue
        self.__attached_provider = host_provider

    def _span_processor_on_start(self, span, parent_context):
        # Re-check for a late-registered host provider (init-order race) before
        # anything else, so host spans start being tee'd as early as possible.
        self._reattach_to_host_provider_if_changed()

        default_span_processor_on_start(span, parent_context)

        # TODO: this is here only because we need self to be able to access the content allow list
        # we should refactor this to not need self
        association_properties = get_value("association_properties")
        if association_properties is not None:
            if not self.enable_content_tracing:
                if self.__content_allow_list.is_allowed(association_properties):
                    attach(set_value("override_enable_content_tracing", True))
                else:
                    attach(set_value("override_enable_content_tracing", False))

    @staticmethod
    def set_static_params(
        resource_attributes: dict,
        enable_content_tracing: bool,
        endpoint: str,
        headers: Dict[str, str],
    ) -> None:
        TracerWrapper.resource_attributes = resource_attributes
        TracerWrapper.enable_content_tracing = enable_content_tracing
        TracerWrapper.endpoint = endpoint
        TracerWrapper.headers = headers

    @classmethod
    def verify_initialized(cls) -> bool:
        if cls.__disabled:
            return False

        if hasattr(cls, "instance"):
            return True

        if (os.getenv("TRACELOOP_SUPPRESS_WARNINGS") or "false").lower() == "true":
            return False

        print(
            Fore.RED
            + "Warning: Traceloop not initialized, make sure you call Traceloop.init()"
        )
        print(Fore.RESET)
        return False

    @classmethod
    def set_disabled(cls, disabled: bool) -> None:
        cls.__disabled = disabled

    def flush(self):
        if hasattr(self, "_TracerWrapper__spans_processor"):
            self.__spans_processor.force_flush()
        elif hasattr(self, "_TracerWrapper__spans_processors"):
            for processor in self.__spans_processors:
                processor.force_flush()

    def get_tracer(self):
        return self.__tracer_provider.get_tracer(TRACER_NAME)


def set_association_properties(properties: dict) -> None:
    attach(set_value("association_properties", properties))

    # Attach association properties to the current span, if it's a workflow or a task
    span = trace.get_current_span()
    if get_value("workflow_name") is not None or get_value("entity_name") is not None:
        _set_association_properties_attributes(span, properties)


def _set_association_properties_attributes(span, properties: dict) -> None:
    for key, value in properties.items():
        span.set_attribute(
            f"{SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES}.{key}", value
        )


def set_workflow_name(workflow_name: str) -> None:
    attach(set_value("workflow_name", workflow_name))


def set_agent_name(agent_name: str) -> None:
    attach(set_value("agent_name", agent_name))


def set_entity_path(entity_path: str) -> None:
    attach(set_value("entity_path", entity_path))


def get_chained_entity_path(entity_name: str) -> str:
    parent = get_value("entity_path")
    if parent is None:
        return entity_name
    else:
        return f"{parent}.{entity_name}"


def set_managed_prompt_tracing_context(
    key: str,
    version: int,
    version_name: str,
    version_hash: str,
    template_variables: dict,
) -> None:
    attach(set_value("managed_prompt", True))
    attach(set_value("prompt_key", key))
    attach(set_value("prompt_version", version))
    attach(set_value("prompt_version_name", version_name))
    attach(set_value("prompt_version_hash", version_hash))
    attach(set_value("prompt_template_variables", template_variables))


def set_external_prompt_tracing_context(
    template: str, variables: dict, version: int
) -> None:
    attach(set_value("managed_prompt", False))
    attach(set_value("prompt_version", version))
    attach(set_value("prompt_template", template))
    attach(set_value("prompt_template_variables", variables))


def is_llm_span(span) -> bool:
    return span.attributes.get(SpanAttributes.LLM_REQUEST_TYPE) is not None


def init_spans_exporter(api_endpoint: str, headers: Dict[str, str]) -> SpanExporter:
    """
    Initialize a span exporter based on the endpoint URL scheme.

    Supported schemes:
        - http:// or https:// → HTTP exporter
        - grpc:// → gRPC exporter (insecure)
        - grpcs:// → gRPC exporter (secure/TLS)
        - No scheme → gRPC exporter (insecure, for backward compatibility)

    Args:
        api_endpoint: The endpoint URL (with or without scheme)
        headers: Headers to include with the exporter requests

    Returns:
        SpanExporter: Configured HTTP or gRPC exporter
    """
    parsed = urlparse(api_endpoint.strip())

    match parsed.scheme.lower():
        case "http" | "https":
            base_url = api_endpoint.strip().rstrip('/')
            if not base_url.endswith('/v1/traces'):
                endpoint = f"{base_url}/v1/traces"
            else:
                endpoint = base_url
            return HTTPExporter(endpoint=endpoint, headers=headers)
        case "grpc":
            return GRPCExporter(
                endpoint=parsed.netloc, headers=headers, insecure=True
            )
        case "grpcs":
            return GRPCExporter(
                endpoint=parsed.netloc, headers=headers, insecure=False
            )
        case _:
            # No scheme → default to insecure gRPC for backward compatibility
            return GRPCExporter(
                endpoint=api_endpoint.strip(), headers=headers, insecure=True
            )


def default_span_processor_on_start(span: Span, parent_context: Context | None = None):
    """
    Same as _span_processor_on_start but without the usage of self which comes from the sdk, good for standalone usage.
    """
    # Stamp moda.environment at the SPAN level, always. On the tee-in path the
    # host provider's Resource carries the HOST's deployment.environment (or
    # none), so ingest would otherwise default Moda's tee-in traces to
    # production. Mirroring the moda-node fix, we set the environment on every
    # span; ingest reads span-level moda.environment with priority over the
    # resource-level attribute. Only stamp when an environment is actually
    # configured — we never fabricate "production".
    environment = resolve_moda_environment()
    if environment is not None:
        span.set_attribute(MODA_ENVIRONMENT_ATTRIBUTE, environment)

    workflow_name = get_value("workflow_name")
    if workflow_name is not None:
        span.set_attribute(SpanAttributes.TRACELOOP_WORKFLOW_NAME, str(workflow_name))

    agent_name = get_value("agent_name")
    if agent_name is not None:
        span.set_attribute(GEN_AI_AGENT_NAME, str(agent_name))

    entity_path = get_value("entity_path")
    if entity_path is not None:
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_PATH, str(entity_path))

    association_properties = get_value("association_properties")
    if association_properties is not None and isinstance(association_properties, dict):
        _set_association_properties_attributes(span, association_properties)

    if is_llm_span(span):
        managed_prompt = get_value("managed_prompt")
        if managed_prompt is not None:
            span.set_attribute(
                SpanAttributes.TRACELOOP_PROMPT_MANAGED, str(managed_prompt)
            )

        prompt_key = get_value("prompt_key")
        if prompt_key is not None:
            span.set_attribute(SpanAttributes.TRACELOOP_PROMPT_KEY, str(prompt_key))

        prompt_version = get_value("prompt_version")
        if prompt_version is not None:
            span.set_attribute(
                SpanAttributes.TRACELOOP_PROMPT_VERSION, str(prompt_version)
            )

        prompt_version_name = get_value("prompt_version_name")
        if prompt_version_name is not None:
            span.set_attribute(
                SpanAttributes.TRACELOOP_PROMPT_VERSION_NAME, str(prompt_version_name)
            )

        prompt_version_hash = get_value("prompt_version_hash")
        if prompt_version_hash is not None:
            span.set_attribute(
                SpanAttributes.TRACELOOP_PROMPT_VERSION_HASH, str(prompt_version_hash)
            )

        prompt_template = get_value("prompt_template")
        if prompt_template is not None:
            span.set_attribute(
                SpanAttributes.TRACELOOP_PROMPT_TEMPLATE, str(prompt_template)
            )

        prompt_template_variables = get_value("prompt_template_variables")
        if prompt_template_variables is not None and isinstance(
            prompt_template_variables, dict
        ):
            for key, value in prompt_template_variables.items():
                span.set_attribute(
                    f"{SpanAttributes.TRACELOOP_PROMPT_TEMPLATE_VARIABLES}.{key}",
                    value,
                )


def get_default_span_processor(
    disable_batch: bool = False,
    api_endpoint: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    exporter: Optional[SpanExporter] = None,
) -> SpanProcessor:
    """
    Creates and returns the default Traceloop span processor.

    Args:
        disable_batch: If True, uses SimpleSpanProcessor, otherwise BatchSpanProcessor
        api_endpoint: The endpoint URL for the exporter (uses TracerWrapper.endpoint if None)
        headers: Headers for the exporter (uses TracerWrapper.headers if None)
        exporter: Custom exporter to use (creates default if None)

    Returns:
        SpanProcessor: The default Traceloop span processor
    """
    endpoint = api_endpoint or TracerWrapper.endpoint
    request_headers = headers or TracerWrapper.headers

    spans_exporter = exporter or init_spans_exporter(endpoint, request_headers)

    if disable_batch or is_notebook():
        processor = SimpleSpanProcessor(spans_exporter)
    else:
        processor = BatchSpanProcessor(spans_exporter)

    setattr(processor, "_traceloop_processor", True)
    processor.on_start = default_span_processor_on_start
    return processor


def _unwrap_proxy_provider(provider: TracerProvider) -> TracerProvider:
    """Return the real delegate behind a ProxyTracerProvider, if any.

    Host SDKs (Sentry, some Datadog setups) may leave a ``ProxyTracerProvider``
    registered as the global while wiring a real provider behind it as the
    delegate. Mirroring the moda-node fix, we unwrap so we attach to the real,
    ``add_span_processor``-capable provider rather than the proxy shell. If the
    delegate is itself a proxy (i.e. no real provider yet), we return the
    original proxy so callers treat it as "no real host provider yet".
    """
    if not isinstance(provider, ProxyTracerProvider):
        return provider

    delegate = getattr(provider, "_delegate", None)
    if delegate is not None and not isinstance(delegate, ProxyTracerProvider):
        return delegate
    return provider


def resolve_host_provider() -> Optional[TracerProvider]:
    """Resolve the currently-active, tee-able host provider, or ``None``.

    Re-reads the global on every call (never snapshot-once): the init-order
    race means a real host provider may register *after* Moda initializes, and
    we must still land Moda's processor on it. Returns a provider that exposes
    ``add_span_processor`` and is not merely a ProxyTracerProvider shell, or
    ``None`` when no real host provider is registered yet.
    """
    current = _unwrap_proxy_provider(get_tracer_provider())
    if isinstance(current, ProxyTracerProvider):
        return None
    if not hasattr(current, "add_span_processor"):
        return None
    return current


def init_tracer_provider(
    resource: Resource, sampler: Optional[Sampler] = None
) -> TracerProvider:
    """Resolve the TracerProvider Moda will attach its SpanProcessor to.

    LOCKED coexistence rule: tee a SpanProcessor into the host provider; never
    register a second global provider when a real host provider already exists.

    Handles the four silent modes:

      (1) Init-order race (Moda before host). Callers must not snapshot the
          returned provider as final — ``resolve_host_provider`` re-reads the
          global so that if a real host provider registers later, Moda's
          processor still lands on it (see ``TracerWrapper`` attach path).
      (2) Env-only OTLP. When OTEL is configured purely via env vars, the global
          may still be a ProxyTracerProvider at Moda-init time; we do NOT treat
          that as a real host provider and instead create+register Moda's own
          (single) global, while the re-check path picks up a real provider if
          one materializes.
      (3) Non-global provider. A host provider handed to Moda out-of-band is
          honored via the re-check; we never blindly ``set_tracer_provider`` on
          top of a real provider.
      (4) Snapshot-once. Avoided: attachment is re-evaluated, not frozen at
          init.

    Loud-fail contract: if the active global is a real, non-proxy provider that
    lacks ``add_span_processor``, we cannot tee without registering a forbidden
    second global. That surfaces loudly to the caller as
    ``TracerProviderCoexistenceError`` instead of a silent ``logging.error`` +
    ``return None``.
    """
    default_provider: TracerProvider = get_tracer_provider()
    unwrapped = _unwrap_proxy_provider(default_provider)

    # No real host provider yet (bare ProxyTracerProvider, incl. env-only OTLP
    # mode where nothing concrete is registered). Create + register exactly one
    # global provider for Moda. This is the ONLY set_tracer_provider Moda ever
    # performs, and only when no real host provider exists.
    if isinstance(unwrapped, ProxyTracerProvider):
        if sampler is not None:
            provider = TracerProvider(resource=resource, sampler=sampler)
        else:
            provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(provider)
        return provider

    # A real host provider is registered. If it cannot accept a span processor,
    # we have no safe seam — fail loudly rather than silently dropping Moda's
    # telemetry or clobbering the host's global provider.
    if not hasattr(unwrapped, "add_span_processor"):
        raise TracerProviderCoexistenceError(
            "Moda cannot tee its SpanProcessor into the active global "
            f"TracerProvider ({type(unwrapped).__name__}): it does not expose "
            "add_span_processor(). Refusing to register a second global "
            "provider. Configure Moda before the host observability SDK, or "
            "use an SDK TracerProvider that supports add_span_processor()."
        )

    # Real, tee-able host provider: reuse it (never a second global).
    return unwrapped


def init_instrumentations(
    should_enrich_metrics: bool,
    base64_image_uploader: Callable[[str, str, str, str], str],
    instruments: Optional[Set[Instruments]] = None,
    block_instruments: Optional[Set[Instruments]] = None,
):
    block_instruments = block_instruments or set()
    # explictly test for None since empty set is a False value
    instruments = instruments if instruments is not None else set(Instruments)

    # Remove any instruments that were explicitly blocked
    instruments = instruments - block_instruments

    instrument_set = False
    for instrument in instruments:
        if instrument == Instruments.AGNO:
            if init_agno_instrumentor():
                instrument_set = True
        elif instrument == Instruments.ALEPHALPHA:
            if init_alephalpha_instrumentor():
                instrument_set = True
        elif instrument == Instruments.ANTHROPIC:
            if init_anthropic_instrumentor(
                should_enrich_metrics, base64_image_uploader
            ):
                instrument_set = True
        elif instrument == Instruments.BEDROCK:
            if init_bedrock_instrumentor(should_enrich_metrics):
                instrument_set = True
        elif instrument == Instruments.CHROMA:
            if init_chroma_instrumentor():
                instrument_set = True
        elif instrument == Instruments.CLAUDE_AGENT_SDK:
            if init_claude_agent_sdk_instrumentor():
                instrument_set = True
        elif instrument == Instruments.COHERE:
            if init_cohere_instrumentor():
                instrument_set = True
        elif instrument == Instruments.CREWAI:
            if init_crewai_instrumentor():
                instrument_set = True
        elif instrument == Instruments.GOOGLE_GENERATIVEAI:
            if init_google_generativeai_instrumentor(
                should_enrich_metrics, base64_image_uploader
            ):
                instrument_set = True
        elif instrument == Instruments.GROQ:
            if init_groq_instrumentor():
                instrument_set = True
        elif instrument == Instruments.HAYSTACK:
            if init_haystack_instrumentor():
                instrument_set = True
        elif instrument == Instruments.LANCEDB:
            if init_lancedb_instrumentor():
                instrument_set = True
        elif instrument == Instruments.LANGCHAIN:
            if init_langchain_instrumentor():
                instrument_set = True
        elif instrument == Instruments.LLAMA_INDEX:
            if init_llama_index_instrumentor():
                instrument_set = True
        elif instrument == Instruments.MARQO:
            if init_marqo_instrumentor():
                instrument_set = True
        elif instrument == Instruments.MCP:
            if init_mcp_instrumentor():
                instrument_set = True
        elif instrument == Instruments.MILVUS:
            if init_milvus_instrumentor():
                instrument_set = True
        elif instrument == Instruments.MISTRAL:
            if init_mistralai_instrumentor():
                instrument_set = True
        elif instrument == Instruments.OLLAMA:
            if init_ollama_instrumentor():
                instrument_set = True
        elif instrument == Instruments.OPENAI:
            if init_openai_instrumentor(should_enrich_metrics, base64_image_uploader):
                instrument_set = True
        elif instrument == Instruments.OPENAI_AGENTS:
            if init_openai_agents_instrumentor():
                instrument_set = True
        elif instrument == Instruments.PINECONE:
            if init_pinecone_instrumentor():
                instrument_set = True
        elif instrument == Instruments.PYMYSQL:
            if init_pymysql_instrumentor():
                instrument_set = True
        elif instrument == Instruments.QDRANT:
            if init_qdrant_instrumentor():
                instrument_set = True
        elif instrument == Instruments.REDIS:
            if init_redis_instrumentor():
                instrument_set = True
        elif instrument == Instruments.REPLICATE:
            if init_replicate_instrumentor():
                instrument_set = True
        elif instrument == Instruments.REQUESTS:
            if init_requests_instrumentor():
                instrument_set = True
        elif instrument == Instruments.SAGEMAKER:
            if init_sagemaker_instrumentor(should_enrich_metrics):
                instrument_set = True
        elif instrument == Instruments.TOGETHER:
            if init_together_instrumentor():
                instrument_set = True
        elif instrument == Instruments.TRANSFORMERS:
            if init_transformers_instrumentor():
                instrument_set = True
        elif instrument == Instruments.URLLIB3:
            if init_urllib3_instrumentor():
                instrument_set = True
        elif instrument == Instruments.VERTEXAI:
            if init_vertexai_instrumentor(should_enrich_metrics, base64_image_uploader):
                instrument_set = True
        elif instrument == Instruments.WATSONX:
            if init_watsonx_instrumentor():
                instrument_set = True
        elif instrument == Instruments.WEAVIATE:
            if init_weaviate_instrumentor():
                instrument_set = True
        elif instrument == Instruments.WRITER:
            if init_writer_instrumentor():
                instrument_set = True
        else:
            print(Fore.RED + f"Warning: {instrument} instrumentation does not exist.")
            print(
                "Usage:\n"
                "from traceloop.sdk.instruments import Instruments\n"
                "Traceloop.init(app_name='...', instruments=set([Instruments.OPENAI]))"
            )
            print(Fore.RESET)

    if not instrument_set:
        print(
            Fore.RED
            + "Warning: No valid instruments set. "
            + "Ensure the instrumented libraries are installed, specify valid instruments, "
            + "or remove 'instruments' argument to use all instruments."
        )
        print(Fore.RESET)

    return instrument_set


def init_openai_instrumentor(
    should_enrich_metrics: bool,
    base64_image_uploader: Callable[[str, str, str, str], str],
):
    try:
        if is_package_installed("openai"):
            from opentelemetry.instrumentation.openai import OpenAIInstrumentor

            instrumentor = OpenAIInstrumentor(
                enrich_assistant=should_enrich_metrics,
                get_common_metrics_attributes=metrics_common_attributes,
                upload_base64_image=base64_image_uploader,
            )
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
            return True

    except Exception as e:
        logging.error(f"Error initializing OpenAI instrumentor: {e}")
    return False


def init_anthropic_instrumentor(
    should_enrich_metrics: bool,
    base64_image_uploader: Callable[[str, str, str, str], str],
):
    try:
        if is_package_installed("anthropic"):
            from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor

            instrumentor = AnthropicInstrumentor(
                enrich_token_usage=should_enrich_metrics,
                get_common_metrics_attributes=metrics_common_attributes,
                upload_base64_image=base64_image_uploader,
            )
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
            return True
    except Exception as e:
        logging.error(f"Error initializing Anthropic instrumentor: {e}")
    return False


def init_cohere_instrumentor():
    try:
        if is_package_installed("cohere"):
            from opentelemetry.instrumentation.cohere import CohereInstrumentor

            instrumentor = CohereInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
            return True
    except Exception as e:
        logging.error(f"Error initializing Cohere instrumentor: {e}")
    return False


def init_pinecone_instrumentor():
    try:
        if is_package_installed("pinecone"):
            from opentelemetry.instrumentation.pinecone import PineconeInstrumentor

            instrumentor = PineconeInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
            return True
    except Exception as e:
        logging.error(f"Error initializing Pinecone instrumentor: {e}")
    return False


def init_qdrant_instrumentor():
    try:
        if is_package_installed("qdrant_client") or is_package_installed(
            "qdrant-client"
        ):
            from opentelemetry.instrumentation.qdrant import QdrantInstrumentor

            instrumentor = QdrantInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
            return True
    except Exception as e:
        logging.error(f"Error initializing Qdrant instrumentor: {e}")
    return False


def init_chroma_instrumentor():
    try:
        if is_package_installed("chromadb"):
            from opentelemetry.instrumentation.chromadb import ChromaInstrumentor

            instrumentor = ChromaInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
            return True
    except Exception as e:
        logging.error(f"Error initializing Chroma instrumentor: {e}")
    return False


def init_google_generativeai_instrumentor(
    should_enrich_metrics: bool,
    base64_image_uploader: Callable[[str, str, str, str], str],
):
    try:
        if is_package_installed("google-generativeai") or is_package_installed(
            "google-genai"
        ):
            from opentelemetry.instrumentation.google_generativeai import (
                GoogleGenerativeAiInstrumentor,
            )

            instrumentor = GoogleGenerativeAiInstrumentor(
                upload_base64_image=base64_image_uploader,
            )
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
            return True
    except Exception as e:
        logging.error(f"Error initializing Gemini instrumentor: {e}")
    return False


def init_haystack_instrumentor():
    try:
        if is_package_installed("haystack"):
            from opentelemetry.instrumentation.haystack import HaystackInstrumentor

            instrumentor = HaystackInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
            return True
    except Exception as e:
        logging.error(f"Error initializing Haystack instrumentor: {e}")
    return False


def init_langchain_instrumentor():
    try:
        if is_package_installed("langchain") or is_package_installed("langgraph"):
            from opentelemetry.instrumentation.langchain import LangchainInstrumentor

            instrumentor = LangchainInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
            return True
    except Exception as e:
        logging.error(f"Error initializing LangChain instrumentor: {e}")
    return False


def init_mistralai_instrumentor():
    try:
        if is_package_installed("mistralai"):
            from opentelemetry.instrumentation.mistralai import MistralAiInstrumentor

            instrumentor = MistralAiInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
            return True
    except Exception as e:
        logging.error(f"Error initializing MistralAI instrumentor: {e}")
    return False


def init_ollama_instrumentor():
    try:
        if is_package_installed("ollama"):
            from opentelemetry.instrumentation.ollama import OllamaInstrumentor

            instrumentor = OllamaInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
            return True
    except Exception as e:
        logging.error(f"Error initializing Ollama instrumentor: {e}")
    return False


def init_transformers_instrumentor():
    try:
        if is_package_installed("transformers"):
            from opentelemetry.instrumentation.transformers import (
                TransformersInstrumentor,
            )

            instrumentor = TransformersInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
            return True
    except Exception as e:
        logging.error(f"Error initializing Transformers instrumentor: {e}")
    return False


def init_together_instrumentor():
    try:
        if is_package_installed("together"):
            from opentelemetry.instrumentation.together import TogetherAiInstrumentor

            instrumentor = TogetherAiInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
            return True
    except Exception as e:
        logging.error(f"Error initializing TogetherAI instrumentor: {e}")
    return False


def init_llama_index_instrumentor():
    try:
        if is_package_installed("llama-index") or is_package_installed("llama_index"):
            from opentelemetry.instrumentation.llamaindex import LlamaIndexInstrumentor

            instrumentor = LlamaIndexInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
            return True
    except Exception as e:
        logging.error(f"Error initializing LlamaIndex instrumentor: {e}")
    return False


def init_milvus_instrumentor():
    try:
        if is_package_installed("pymilvus"):
            from opentelemetry.instrumentation.milvus import MilvusInstrumentor

            instrumentor = MilvusInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
            return True
    except Exception as e:
        logging.error(f"Error initializing Milvus instrumentor: {e}")
    return False


def init_requests_instrumentor():
    try:
        if is_package_installed("requests"):
            from opentelemetry.instrumentation.requests import RequestsInstrumentor

            instrumentor = RequestsInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument(excluded_urls=EXCLUDED_URLS)
            return True
    except Exception as e:
        logging.error(f"Error initializing Requests instrumentor: {e}")
    return False


def init_urllib3_instrumentor():
    try:
        if is_package_installed("urllib3"):
            from opentelemetry.instrumentation.urllib3 import URLLib3Instrumentor

            instrumentor = URLLib3Instrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument(excluded_urls=EXCLUDED_URLS)
            return True
    except Exception as e:
        logging.error(f"Error initializing urllib3 instrumentor: {e}")
    return False


def init_pymysql_instrumentor():
    try:
        if is_package_installed("sqlalchemy"):
            from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

            instrumentor = SQLAlchemyInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
            return True
    except Exception as e:
        logging.error(f"Error initializing SQLAlchemy instrumentor: {e}")
    return False


def init_bedrock_instrumentor(should_enrich_metrics: bool):
    if is_package_installed("boto3"):
        from opentelemetry.instrumentation.bedrock import BedrockInstrumentor

        instrumentor = BedrockInstrumentor(
            enrich_token_usage=should_enrich_metrics,
        )
        if not instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.instrument()
        return True
    return False


def init_sagemaker_instrumentor(should_enrich_metrics: bool):
    try:
        if is_package_installed("boto3"):
            from opentelemetry.instrumentation.sagemaker import SageMakerInstrumentor

            instrumentor = SageMakerInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
            return True
    except Exception as e:
        logging.error(f"Error initializing SageMaker instrumentor: {e}")
    return False


def init_replicate_instrumentor():
    try:
        if is_package_installed("replicate"):
            from opentelemetry.instrumentation.replicate import ReplicateInstrumentor

            instrumentor = ReplicateInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
            return True
    except Exception as e:
        logging.error(f"Error initializing Replicate instrumentor: {e}")
    return False


def init_vertexai_instrumentor(
    should_enrich_metrics: bool,
    base64_image_uploader: Callable[[str, str, str, str], str],
):
    try:
        if is_package_installed("google-cloud-aiplatform"):
            from opentelemetry.instrumentation.vertexai import VertexAIInstrumentor

            instrumentor = VertexAIInstrumentor(
                upload_base64_image=base64_image_uploader,
            )
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
            return True
    except Exception as e:
        logging.warning(f"Error initializing Vertex AI instrumentor: {e}")
    return False


def init_watsonx_instrumentor():
    try:
        if is_package_installed("ibm-watsonx-ai") or is_package_installed(
            "ibm-watson-machine-learning"
        ):
            from opentelemetry.instrumentation.watsonx import WatsonxInstrumentor

            instrumentor = WatsonxInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
            return True
    except Exception as e:
        logging.warning(f"Error initializing Watsonx instrumentor: {e}")
    return False


def init_weaviate_instrumentor():
    try:
        if is_package_installed("weaviate"):
            from opentelemetry.instrumentation.weaviate import WeaviateInstrumentor

            instrumentor = WeaviateInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
            return True
    except Exception as e:
        logging.warning(f"Error initializing Weaviate instrumentor: {e}")
    return False


def init_writer_instrumentor():
    try:
        if is_package_installed("writer-sdk"):
            from opentelemetry.instrumentation.writer import WriterInstrumentor

            instrumentor = WriterInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
            return True
    except Exception as e:
        logging.error(f"Error initializing Writer instrumentor: {e}")
    return False


def init_agno_instrumentor():
    try:
        if is_package_installed("agno"):
            from opentelemetry.instrumentation.agno import AgnoInstrumentor

            instrumentor = AgnoInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
            return True
    except Exception as e:
        logging.error(f"Error initializing Agno instrumentor: {e}")
    return False


def init_alephalpha_instrumentor():
    try:
        if is_package_installed("aleph_alpha_client"):
            from opentelemetry.instrumentation.alephalpha import AlephAlphaInstrumentor

            instrumentor = AlephAlphaInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
            return True
    except Exception as e:
        logging.error(f"Error initializing Aleph Alpha instrumentor: {e}")
    return False


def init_marqo_instrumentor():
    try:
        if is_package_installed("marqo"):
            from opentelemetry.instrumentation.marqo import MarqoInstrumentor

            instrumentor = MarqoInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
            return True
    except Exception as e:
        logging.error(f"Error initializing marqo instrumentor: {e}")
    return False


def init_lancedb_instrumentor():
    try:
        if is_package_installed("lancedb"):
            from opentelemetry.instrumentation.lancedb import LanceInstrumentor

            instrumentor = LanceInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
            return True
    except Exception as e:
        logging.error(f"Error initializing LanceDB instrumentor: {e}")
    return False


def init_redis_instrumentor():
    try:
        if is_package_installed("redis"):
            from opentelemetry.instrumentation.redis import RedisInstrumentor

            instrumentor = RedisInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument(excluded_urls=EXCLUDED_URLS)
            return True
    except Exception as e:
        logging.error(f"Error initializing redis instrumentor: {e}")
    return False


def init_groq_instrumentor():
    try:
        if is_package_installed("groq"):
            from opentelemetry.instrumentation.groq import GroqInstrumentor

            instrumentor = GroqInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
            return True
    except Exception as e:
        logging.error(f"Error initializing Groq instrumentor: {e}")
    return False


def init_crewai_instrumentor():
    try:
        if is_package_installed("crewai"):
            os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
            from opentelemetry.instrumentation.crewai import CrewAIInstrumentor

            instrumentor = CrewAIInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
            return True
    except Exception as e:
        logging.error(f"Error initializing CrewAI instrumentor: {e}")
    return False


def init_mcp_instrumentor():
    try:
        if is_package_installed("mcp"):
            from opentelemetry.instrumentation.mcp import McpInstrumentor

            instrumentor = McpInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
            return True
    except Exception as e:
        logging.error(f"Error initializing MCP instrumentor: {e}")
    return False


def init_openai_agents_instrumentor():
    try:
        if is_package_installed("openai-agents"):
            from opentelemetry.instrumentation.openai_agents import (
                OpenAIAgentsInstrumentor,
            )

            instrumentor = OpenAIAgentsInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
            return True
    except Exception as e:
        logging.error(f"Error initializing OpenAI Agents instrumentor: {e}")
    return False


def init_claude_agent_sdk_instrumentor():
    try:
        if is_package_installed("claude-agent-sdk"):
            from opentelemetry.instrumentation.claude_agent_sdk import (
                ClaudeAgentSDKInstrumentor,
            )

            instrumentor = ClaudeAgentSDKInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
            return True
    except Exception as e:
        logging.error(f"Error initializing Claude Agent SDK instrumentor: {e}")
    return False


def metrics_common_attributes():
    common_attributes = {}
    workflow_name = get_value("workflow_name")
    if workflow_name is not None:
        common_attributes[SpanAttributes.TRACELOOP_WORKFLOW_NAME] = workflow_name

    entity_name = get_value("entity_name")
    if entity_name is not None:
        common_attributes[SpanAttributes.TRACELOOP_ENTITY_NAME] = entity_name

    association_properties = get_value("association_properties")
    if association_properties is not None:
        for key, value in association_properties.items():
            common_attributes[
                f"{SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES}.{key}"
            ] = value

    return common_attributes
