import json
from unittest.mock import MagicMock

import pytest

from opentelemetry.instrumentation.openai.shared import chat_wrappers
from opentelemetry.instrumentation.openai.shared.config import Config

MESSAGES = [{"role": "user", "content": [
    {"type": "text", "text": "What should I click?"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGk="}},
]}]


def recorded_prompt(span):
    attributes = {call.args[0]: call.args[1] for call in span.set_attribute.call_args_list}
    return attributes["gen_ai.prompt.0.role"], json.loads(attributes["gen_ai.prompt.0.content"])


@pytest.mark.asyncio
async def test_failed_image_upload_keeps_the_message(monkeypatch):
    async def fail(*_):
        raise RuntimeError("404 from ingest")

    monkeypatch.setattr(Config, "upload_base64_image", fail)
    span = MagicMock()
    span.is_recording.return_value = True
    await chat_wrappers._set_prompts(span, MESSAGES)

    role, content = recorded_prompt(span)
    assert role == "user"
    assert content == [
        {"type": "text", "text": "What should I click?"},
        {"type": "text", "text": chat_wrappers.IMAGE_NOT_CAPTURED},
    ]


@pytest.mark.asyncio
async def test_uploaded_image_is_replaced_by_its_url(monkeypatch):
    async def upload(*_):
        return "https://ingest.test/image.png"

    monkeypatch.setattr(Config, "upload_base64_image", upload)
    span = MagicMock()
    span.is_recording.return_value = True
    await chat_wrappers._set_prompts(span, MESSAGES)

    assert recorded_prompt(span)[1][1] == {"type": "image_url", "image_url": {"url": "https://ingest.test/image.png"}}
