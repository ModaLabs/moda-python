"""Files inside Anthropic requests, swapped for references.

A screenshot or document sent to the model, including one inside a tool
result (as computer use sends them), is uploaded once by its content hash in
the background and the span carries a small `artifact_ref` block in its place.
Ingest keeps the reference with the message.

`Config.upload_file(data, mime_type, name)` does the upload and returns the
block; without it, blocks pass through unchanged.
"""

import base64
import binascii
import logging
from typing import Any

from opentelemetry.instrumentation.anthropic.config import Config

logger = logging.getLogger(__name__)

FILE_NOT_CAPTURED = "[file not captured]"


def file_uploads_enabled() -> bool:
    return Config.upload_file is not None


def _file_ref(source: dict, name: Any) -> dict:
    try:
        data = base64.b64decode(source.get("data") or "", validate=True)
        ref = Config.upload_file(data, str(source.get("media_type") or "application/octet-stream"),
                                 name if isinstance(name, str) else None)
    except (binascii.Error, ValueError, TypeError):
        ref = None
    except Exception as e:  # never cost the message its other content
        logger.warning("File upload failed; keeping the message without it: %s", e)
        ref = None
    return ref if isinstance(ref, dict) else {"type": "text", "text": FILE_NOT_CAPTURED}


def replace_inline_file(block: Any) -> Any:
    """The block with any base64 image or document swapped for a reference."""
    if not file_uploads_enabled() or not isinstance(block, dict):
        return block
    kind = block.get("type")
    source = block.get("source")
    if kind in ("image", "document") and isinstance(source, dict) and source.get("type") == "base64":
        return _file_ref(source, block.get("title"))
    if kind == "tool_result" and isinstance(block.get("content"), list):
        return {**block, "content": [replace_inline_file(item) for item in block["content"]]}
    return block
