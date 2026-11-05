import json
from unittest.mock import MagicMock

import pytest

from opentelemetry.instrumentation.anthropic import span_utils
from opentelemetry.instrumentation.anthropic.config import Config

CONTENT = [
    {"type": "text", "text": "What should I click?"},
    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aGk="}},
]


@pytest.mark.asyncio
async def test_failed_image_upload_keeps_the_message(monkeypatch):
    async def fail(*_):
        raise RuntimeError("404 from ingest")

    monkeypatch.setattr(Config, "upload_base64_image", fail)
    assert json.loads(await span_utils._dump_content(0, CONTENT, MagicMock())) == [
        {"type": "text", "text": "What should I click?"},
        {"type": "text", "text": span_utils.IMAGE_NOT_CAPTURED},
    ]


@pytest.mark.asyncio
async def test_uploaded_image_is_replaced_by_its_url(monkeypatch):
    async def upload(*_):
        return "https://ingest.test/image.png"

    monkeypatch.setattr(Config, "upload_base64_image", upload)
    content = json.loads(await span_utils._dump_content(0, CONTENT, MagicMock()))
    assert content[1] == {"type": "image_url", "image_url": {"url": "https://ingest.test/image.png"}}
