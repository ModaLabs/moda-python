"""Files inside OpenAI requests and responses, swapped for references.

A screenshot, recording or document sent to or from the model is uploaded once
by its content hash (in the background) and the span carries a small
`artifact_ref` block in its place, so a span stays small however many files a
conversation holds. Ingest keeps the reference with the message.

`Config.upload_file(data, mime_type, name)` does the upload and returns the
block; without it, blocks pass through unchanged.
"""

import base64
import binascii
import json
import logging
from typing import Any, Optional, Tuple

from opentelemetry.instrumentation.openai.shared.config import Config

logger = logging.getLogger(__name__)

FILE_NOT_CAPTURED = "[file not captured]"

# input_audio's `format` as a MIME type.
_AUDIO_TYPES = {
    "wav": "audio/wav", "mp3": "audio/mpeg", "flac": "audio/flac", "opus": "audio/ogg", "pcm16": "audio/pcm",
}


def file_uploads_enabled() -> bool:
    return Config.upload_file is not None


def decode_data_url(url: Any) -> Optional[Tuple[bytes, str]]:
    """The bytes and MIME type of a base64 data URL, or None for anything else."""
    if not isinstance(url, str) or not url.startswith("data:"):
        return None
    header, sep, payload = url.partition(",")
    if not sep or not header.endswith(";base64"):
        return None
    mime_type = header[len("data:"):-len(";base64")].split(";")[0].strip().lower()
    return base64.b64decode(payload, validate=True), mime_type or "application/octet-stream"


def file_ref(data: bytes, mime_type: str, name: Optional[str] = None) -> dict:
    """Upload a file in the background and return the block that names it."""
    try:
        ref = Config.upload_file(data, mime_type, name)
    except Exception as e:  # never cost the message its other content
        logger.warning("File upload failed; keeping the message without it: %s", e)
        ref = None
    return ref if isinstance(ref, dict) else {"type": "text", "text": FILE_NOT_CAPTURED}


def _from_data_url(url: Any, name: Optional[str] = None) -> Optional[dict]:
    try:
        decoded = decode_data_url(url)
    except (binascii.Error, ValueError):
        return {"type": "text", "text": FILE_NOT_CAPTURED}
    return file_ref(decoded[0], decoded[1], name) if decoded else None


def _from_base64(data: Any, mime_type: str, name: Optional[str] = None) -> Optional[dict]:
    if not isinstance(data, str) or not data:
        return None
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        return {"type": "text", "text": FILE_NOT_CAPTURED}
    return file_ref(raw, mime_type, name)


def replace_inline_file(block: Any) -> Any:
    """The block with any file it carries inline swapped for a reference.

    Covers chat parts (image_url, input_audio, file), Responses parts
    (input_image, input_file, computer screenshots) and generated images.
    URLs and provider file IDs are left as they are: there are no bytes to keep.
    """
    if not file_uploads_enabled() or not isinstance(block, dict):
        return block
    kind = block.get("type")
    ref = None
    if kind == "image_url":
        image = block.get("image_url")
        ref = _from_data_url(image.get("url") if isinstance(image, dict) else image)
    elif kind in ("input_image", "image", "computer_screenshot"):
        ref = _from_data_url(block.get("image_url"))
    elif kind == "file":
        file = block.get("file") if isinstance(block.get("file"), dict) else block
        ref = _from_data_url(file.get("file_data"), file.get("filename"))
    elif kind == "input_file":
        ref = _from_data_url(block.get("file_data"), block.get("filename"))
    elif kind == "input_audio":
        audio = block.get("input_audio") or {}
        fmt = str(audio.get("format") or "").lower()
        ref = _from_base64(audio.get("data"), _AUDIO_TYPES.get(fmt, f"audio/{fmt}" if fmt else "audio/wav"))
    elif kind == "image_generation_call":
        fmt = str(block.get("output_format") or "png").lower()
        ref = _from_base64(block.get("result"), f"image/{'jpeg' if fmt == 'jpg' else fmt}")
    return ref if ref is not None else block


def sniff_audio_type(data: bytes) -> str:
    """An audio file's type from its first bytes; WAV when unrecognized, OpenAI's default."""
    if data[:4] == b"fLaC":
        return "audio/flac"
    if data[:4] == b"OggS":
        return "audio/ogg"
    if data[:3] == b"ID3" or data[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return "audio/mpeg"
    return "audio/wav"


def audio_output_content(message: dict) -> Any:
    """A chat reply's content, with the audio the model spoke as a file after its text."""
    content = message.get("content")
    audio = message.get("audio")
    if not file_uploads_enabled() or not isinstance(audio, dict) or not isinstance(audio.get("data"), str):
        return content
    try:
        data = base64.b64decode(audio["data"], validate=True)
    except (binascii.Error, ValueError):
        return content
    text = content if isinstance(content, str) and content else audio.get("transcript")
    blocks = [{"type": "text", "text": text}] if isinstance(text, str) and text else []
    blocks.append(file_ref(data, sniff_audio_type(data)))
    return json.dumps(blocks)


def replace_inline_files(content: Any) -> Any:
    """A message's content with every inline file swapped for a reference."""
    if isinstance(content, list):
        return [replace_inline_file(item) for item in content]
    return replace_inline_file(content)
