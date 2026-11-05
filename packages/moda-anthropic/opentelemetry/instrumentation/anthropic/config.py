from typing import Callable, Optional

from typing_extensions import Coroutine


class Config:
    enrich_token_usage = False
    exception_logger = None
    get_common_metrics_attributes: Callable[[], dict] = lambda: {}
    upload_base64_image: Optional[
        Callable[[str, str, str, str], Coroutine[None, None, str]]
    ] = None
    # Uploads a file found in a request and returns the block that replaces it
    # in the span (inline_files.py). None leaves files inline.
    upload_file: Optional[Callable[[bytes, str, Optional[str]], Optional[dict]]] = None
    use_legacy_attributes = True
