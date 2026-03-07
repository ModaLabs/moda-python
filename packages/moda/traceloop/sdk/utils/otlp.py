from urllib.parse import urlparse


def normalize_http_signal_endpoint(endpoint: str, signal: str) -> str:
    """Normalize HTTP OTLP endpoints to avoid double-appending /v1/<signal>."""
    parsed = urlparse(endpoint.strip().rstrip("/"))
    path = parsed.path.rstrip("/")

    for suffix in ("/v1/traces", "/v1/metrics", "/v1/logs"):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break

    normalized_path = f"{path}/v1/{signal}" if path else f"/v1/{signal}"
    return parsed._replace(path=normalized_path).geturl()
