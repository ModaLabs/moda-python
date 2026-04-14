import json
import logging
import types
import openai
import pydantic
from importlib.metadata import version

from opentelemetry.instrumentation.openai.shared.config import Config
from opentelemetry.instrumentation.openai.utils import (
    dont_throw,
    is_openai_v1,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
    openai_attributes as OpenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace.propagation import set_span_in_context
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

OPENAI_LLM_USAGE_TOKEN_TYPES = ["prompt_tokens", "completion_tokens"]
PROMPT_FILTER_KEY = "prompt_filter_results"
PROMPT_ERROR = "prompt_error"

_PYDANTIC_VERSION = version("pydantic")


logger = logging.getLogger(__name__)


def _set_span_attribute(span, name, value):
    if value is None or value == "":
        return

    if hasattr(openai, "NOT_GIVEN") and value == openai.NOT_GIVEN:
        return

    span.set_attribute(name, value)


def _set_client_attributes(span, instance):
    if not span.is_recording():
        return

    if not is_openai_v1():
        return

    client = instance._client  # pylint: disable=protected-access
    if isinstance(client, (openai.AsyncOpenAI, openai.OpenAI)):
        _set_span_attribute(
            span, SpanAttributes.LLM_OPENAI_API_BASE, str(client.base_url)
        )
    if isinstance(client, (openai.AsyncAzureOpenAI, openai.AzureOpenAI)):
        _set_span_attribute(
            span, SpanAttributes.LLM_OPENAI_API_VERSION, client._api_version
        )  # pylint: disable=protected-access


def _set_api_attributes(span):
    if not span.is_recording():
        return

    if is_openai_v1():
        return

    base_url = openai.base_url if hasattr(openai, "base_url") else openai.api_base

    _set_span_attribute(span, SpanAttributes.LLM_OPENAI_API_BASE, base_url)
    _set_span_attribute(span, SpanAttributes.LLM_OPENAI_API_TYPE, openai.api_type)
    _set_span_attribute(span, SpanAttributes.LLM_OPENAI_API_VERSION, openai.api_version)

    return


def _set_functions_attributes(span, functions):
    if not functions:
        return

    for i, function in enumerate(functions):
        prefix = f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.{i}"
        _set_span_attribute(span, f"{prefix}.name", function.get("name"))
        _set_span_attribute(span, f"{prefix}.description", function.get("description"))
        _set_span_attribute(
            span, f"{prefix}.parameters", json.dumps(function.get("parameters"))
        )


def set_tools_attributes(span, tools):
    if not tools:
        return

    for i, tool in enumerate(tools):
        function = tool.get("function")
        if not function:
            continue

        prefix = f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.{i}"
        _set_span_attribute(span, f"{prefix}.name", function.get("name"))
        _set_span_attribute(span, f"{prefix}.description", function.get("description"))
        _set_span_attribute(
            span, f"{prefix}.parameters", json.dumps(function.get("parameters"))
        )


def _set_request_attributes(span, kwargs, instance=None):
    if not span.is_recording():
        return

    _set_api_attributes(span)

    base_url = _get_openai_base_url(instance) if instance else ""
    vendor = _get_vendor_from_url(base_url)
    _set_span_attribute(span, GenAIAttributes.GEN_AI_SYSTEM, vendor)

    model = kwargs.get("model")
    if vendor == "AWS" and model and "." in model:
        model = _cross_region_check(model)
    elif vendor == "OpenRouter":
        model = _extract_model_name_from_provider_format(model)

    _set_span_attribute(span, GenAIAttributes.GEN_AI_REQUEST_MODEL, model)
    _set_span_attribute(
        span, GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS, kwargs.get("max_tokens")
    )
    _set_span_attribute(
        span, GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE, kwargs.get("temperature")
    )
    _set_span_attribute(span, GenAIAttributes.GEN_AI_REQUEST_TOP_P, kwargs.get("top_p"))
    _set_span_attribute(
        span, SpanAttributes.LLM_FREQUENCY_PENALTY, kwargs.get("frequency_penalty")
    )
    _set_span_attribute(
        span, SpanAttributes.LLM_PRESENCE_PENALTY, kwargs.get("presence_penalty")
    )
    _set_span_attribute(span, SpanAttributes.LLM_USER, kwargs.get("user"))
    _set_span_attribute(span, SpanAttributes.LLM_HEADERS, str(kwargs.get("headers")))
    # The new OpenAI SDK removed the `headers` and create new field called `extra_headers`
    if kwargs.get("extra_headers") is not None:
        _set_span_attribute(
            span, SpanAttributes.LLM_HEADERS, str(kwargs.get("extra_headers"))
        )
    _set_span_attribute(
        span, SpanAttributes.LLM_IS_STREAMING, kwargs.get("stream") or False
    )
    _set_span_attribute(
        span, OpenAIAttributes.OPENAI_REQUEST_SERVICE_TIER, kwargs.get("service_tier")
    )
    if response_format := kwargs.get("response_format"):
        # backward-compatible check for
        # openai.types.shared_params.response_format_json_schema.ResponseFormatJSONSchema
        if (
            isinstance(response_format, dict)
            and response_format.get("type") == "json_schema"
            and response_format.get("json_schema")
        ):
            schema = dict(response_format.get("json_schema")).get("schema")
            if schema:
                _set_span_attribute(
                    span,
                    SpanAttributes.LLM_REQUEST_STRUCTURED_OUTPUT_SCHEMA,
                    json.dumps(schema),
                )
        elif (
            isinstance(response_format, pydantic.BaseModel)
            or (
                hasattr(response_format, "model_json_schema")
                and callable(response_format.model_json_schema)
            )
        ):
            _set_span_attribute(
                span,
                SpanAttributes.LLM_REQUEST_STRUCTURED_OUTPUT_SCHEMA,
                json.dumps(response_format.model_json_schema()),
            )
        else:
            schema = None
            try:
                schema = json.dumps(pydantic.TypeAdapter(response_format).json_schema())
            except Exception:
                try:
                    schema = json.dumps(response_format)
                except Exception:
                    pass

            if schema:
                _set_span_attribute(
                    span,
                    SpanAttributes.LLM_REQUEST_STRUCTURED_OUTPUT_SCHEMA,
                    schema,
                )


@dont_throw
def _set_response_attributes(span, response):
    if not span.is_recording():
        return

    if "error" in response:
        _set_span_attribute(
            span,
            f"{GenAIAttributes.GEN_AI_PROMPT}.{PROMPT_ERROR}",
            json.dumps(response.get("error")),
        )
        return

    response_model = response.get("model")
    if response_model:
        response_model = _extract_model_name_from_provider_format(response_model)
    _set_span_attribute(span, GenAIAttributes.GEN_AI_RESPONSE_MODEL, response_model)
    _set_span_attribute(span, GenAIAttributes.GEN_AI_RESPONSE_ID, response.get("id"))

    _set_span_attribute(
        span,
        SpanAttributes.LLM_OPENAI_RESPONSE_SYSTEM_FINGERPRINT,
        response.get("system_fingerprint"),
    )
    _set_span_attribute(
        span,
        OpenAIAttributes.OPENAI_RESPONSE_SERVICE_TIER,
        response.get("service_tier"),
    )
    _log_prompt_filter(span, response)
    usage = response.get("usage")
    if not usage:
        return

    if is_openai_v1() and not isinstance(usage, dict):
        usage = usage.__dict__

    _set_span_attribute(
        span, SpanAttributes.LLM_USAGE_TOTAL_TOKENS, usage.get("total_tokens")
    )
    _set_span_attribute(
        span,
        GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS,
        usage.get("completion_tokens"),
    )
    _set_span_attribute(
        span, GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS, usage.get("prompt_tokens")
    )
    prompt_tokens_details = dict(usage.get("prompt_tokens_details", {}))
    _set_span_attribute(
        span,
        SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS,
        prompt_tokens_details.get("cached_tokens", 0),
    )
    return


def _log_prompt_filter(span, response_dict):
    if response_dict.get("prompt_filter_results"):
        _set_span_attribute(
            span,
            f"{GenAIAttributes.GEN_AI_PROMPT}.{PROMPT_FILTER_KEY}",
            json.dumps(response_dict.get("prompt_filter_results")),
        )


@dont_throw
def _set_span_stream_usage(span, prompt_tokens, completion_tokens):
    if not span.is_recording():
        return

    if isinstance(completion_tokens, int) and completion_tokens >= 0:
        _set_span_attribute(
            span, GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS, completion_tokens
        )

    if isinstance(prompt_tokens, int) and prompt_tokens >= 0:
        _set_span_attribute(span, GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS, prompt_tokens)

    if (
        isinstance(prompt_tokens, int)
        and isinstance(completion_tokens, int)
        and completion_tokens + prompt_tokens >= 0
    ):
        _set_span_attribute(
            span,
            SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
            completion_tokens + prompt_tokens,
        )


def _get_openai_base_url(instance):
    if hasattr(instance, "_client"):
        client = instance._client  # pylint: disable=protected-access
        if isinstance(client, (openai.AsyncOpenAI, openai.OpenAI)):
            return str(client.base_url)

    return ""


def _get_vendor_from_url(base_url):
    if not base_url:
        return "openai"

    if "openai.azure.com" in base_url:
        return "Azure"
    elif "amazonaws.com" in base_url or "bedrock" in base_url:
        return "AWS"
    elif "googleapis.com" in base_url or "vertex" in base_url:
        return "Google"
    elif "openrouter.ai" in base_url:
        return "OpenRouter"
    elif "groq.com" in base_url:
        return "Groq"
    elif "together.xyz" in base_url or "together.ai" in base_url:
        return "Together"
    elif "mistral.ai" in base_url:
        return "Mistral"
    elif "cohere.com" in base_url or "cohere.ai" in base_url:
        return "Cohere"
    elif "deepseek.com" in base_url:
        return "DeepSeek"
    elif "fireworks.ai" in base_url:
        return "Fireworks"
    elif "litellm" in base_url:
        return "LiteLLM"

    return "openai"


def _cross_region_check(value):
    if not value or "." not in value:
        return value

    prefixes = ["us", "us-gov", "eu", "apac"]
    if any(value.startswith(prefix + ".") for prefix in prefixes):
        parts = value.split(".")
        if len(parts) > 2:
            return parts[2]
        else:
            return value
    else:
        vendor, model = value.split(".", 1)
        return model


def _extract_model_name_from_provider_format(model_name):
    """
    Extract model name from provider/model format.
    E.g., 'openai/gpt-4o' -> 'gpt-4o', 'anthropic/claude-3-sonnet' -> 'claude-3-sonnet'
    """
    if not model_name:
        return model_name

    if "/" in model_name:
        parts = model_name.split("/")
        return parts[-1]  # Return the last part (actual model name)

    return model_name


def _extract_litellm_headers(response):
    """
    Extract LiteLLM metadata from response headers.
    Returns dict with LiteLLM metadata if detected, None otherwise.
    """
    headers = None

    # Try to access headers from OpenAI SDK response object
    if hasattr(response, '_response') and hasattr(response._response, 'headers'):
        headers = response._response.headers
    elif hasattr(response, 'response') and hasattr(response.response, 'headers'):
        headers = response.response.headers
    elif hasattr(response, '_headers'):
        headers = response._headers

    if not headers:
        return None

    # Helper to safely get header
    def get_header(name):
        if hasattr(headers, 'get'):
            return headers.get(name)
        return None

    call_id = get_header('x-litellm-call-id')
    model_id = get_header('x-litellm-model-id')
    model_group = get_header('x-litellm-model-group')
    model_api_base = get_header('x-litellm-model-api-base')
    litellm_version = get_header('x-litellm-version')
    response_cost = get_header('x-litellm-response-cost')
    key_spend = get_header('x-litellm-key-spend')
    response_duration_ms = get_header('x-litellm-response-duration-ms')
    overhead_duration_ms = get_header('x-litellm-overhead-duration-ms')
    attempted_retries = get_header('x-litellm-attempted-retries')
    attempted_fallbacks = get_header('x-litellm-attempted-fallbacks')

    if not any([call_id, model_id, model_group, litellm_version]):
        return None

    return {
        'is_proxy': True,
        'call_id': call_id or None,
        'model_id': model_id or None,
        'model_group': model_group or None,
        'model_api_base': model_api_base or None,
        'version': litellm_version or None,
        'response_cost': float(response_cost) if response_cost else None,
        'key_spend': float(key_spend) if key_spend else None,
        'response_duration_ms': float(response_duration_ms) if response_duration_ms else None,
        'overhead_duration_ms': float(overhead_duration_ms) if overhead_duration_ms else None,
        'attempted_retries': int(attempted_retries) if attempted_retries else None,
        'attempted_fallbacks': int(attempted_fallbacks) if attempted_fallbacks else None,
    }


def _set_litellm_span_attributes(span, metadata):
    """Set LiteLLM-specific span attributes on the span."""
    _set_span_attribute(span, "litellm.is_proxy", True)

    if metadata.get('call_id'):
        _set_span_attribute(span, "litellm.call_id", metadata['call_id'])
    if metadata.get('model_id'):
        _set_span_attribute(span, "litellm.model_id", metadata['model_id'])
    if metadata.get('model_group'):
        _set_span_attribute(span, "litellm.model_group", metadata['model_group'])
        _set_span_attribute(span, "llm.system", metadata['model_group'])
    if metadata.get('model_api_base'):
        _set_span_attribute(span, "litellm.model_api_base", metadata['model_api_base'])
    if metadata.get('version'):
        _set_span_attribute(span, "litellm.version", metadata['version'])
    if metadata.get('response_cost') is not None:
        _set_span_attribute(span, "litellm.response_cost", metadata['response_cost'])
    if metadata.get('key_spend') is not None:
        _set_span_attribute(span, "litellm.key_spend", metadata['key_spend'])
    if metadata.get('response_duration_ms') is not None:
        _set_span_attribute(span, "litellm.response_duration_ms", metadata['response_duration_ms'])
    if metadata.get('overhead_duration_ms') is not None:
        _set_span_attribute(span, "litellm.overhead_duration_ms", metadata['overhead_duration_ms'])
    if metadata.get('attempted_retries') is not None:
        _set_span_attribute(span, "litellm.attempted_retries", metadata['attempted_retries'])
    if metadata.get('attempted_fallbacks') is not None:
        _set_span_attribute(span, "litellm.attempted_fallbacks", metadata['attempted_fallbacks'])


def is_streaming_response(response):
    if is_openai_v1():
        return isinstance(response, openai.Stream) or isinstance(
            response, openai.AsyncStream
        )

    return isinstance(response, types.GeneratorType) or isinstance(
        response, types.AsyncGeneratorType
    )


def model_as_dict(model):
    if isinstance(model, dict):
        return model
    if _PYDANTIC_VERSION < "2.0.0":
        return model.dict()
    if hasattr(model, "model_dump"):
        return model.model_dump()
    elif hasattr(model, "parse"):  # Raw API response
        return model_as_dict(model.parse())
    else:
        return model


def _token_type(token_type: str):
    if token_type == "prompt_tokens":
        return "input"
    elif token_type == "completion_tokens":
        return "output"

    return None


def metric_shared_attributes(
    response_model: str, operation: str, server_address: str, is_streaming: bool = False
):
    attributes = Config.get_common_metrics_attributes()
    vendor = _get_vendor_from_url(server_address)

    return {
        **attributes,
        GenAIAttributes.GEN_AI_SYSTEM: vendor,
        GenAIAttributes.GEN_AI_RESPONSE_MODEL: response_model,
        "gen_ai.operation.name": operation,
        "server.address": server_address,
        "stream": is_streaming,
    }


def propagate_trace_context(span, kwargs):
    if is_openai_v1():
        extra_headers = kwargs.get("extra_headers", {})
        ctx = set_span_in_context(span)
        TraceContextTextMapPropagator().inject(extra_headers, context=ctx)
        kwargs["extra_headers"] = extra_headers
    else:
        headers = kwargs.get("headers", {})
        ctx = set_span_in_context(span)
        TraceContextTextMapPropagator().inject(headers, context=ctx)
        kwargs["headers"] = headers
