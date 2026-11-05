from typing import Callable, Optional

from opentelemetry._logs import Logger


class Config:
    enrich_assistant = False
    exception_logger = None
    get_common_metrics_attributes: Callable[[], dict] = lambda: {}
    upload_base64_image: Callable[[str, str, str, str], str] = (
        lambda trace_id, span_id, image_name, base64_string: str
    )
    # Uploads a file found in a request or response and returns the block that
    # replaces it in the span (shared/inline_files.py). None leaves files inline.
    upload_file: Optional[Callable[[bytes, str, Optional[str]], Optional[dict]]] = None
    enable_trace_context_propagation: bool = True
    use_legacy_attributes = True
    event_logger: Optional[Logger] = None
