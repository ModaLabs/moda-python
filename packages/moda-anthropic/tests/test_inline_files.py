import base64
import json
from unittest.mock import MagicMock

import pytest

from opentelemetry.instrumentation.anthropic import inline_files, span_utils
from opentelemetry.instrumentation.anthropic.config import Config

PNG = b"\x89PNG\r\n\x1a\n shot"
PDF = b"%PDF-1.7 report"


def b64(data):
    return base64.b64encode(data).decode()


@pytest.fixture
def uploads(monkeypatch):
    sent = []

    def upload(data, mime_type, name):
        sent.append((data, mime_type, name))
        return {"type": "artifact_ref", "sha256": f"sha-{len(sent)}", "mime_type": mime_type}

    monkeypatch.setattr(Config, "upload_file", upload)
    return sent


@pytest.mark.asyncio
async def test_images_documents_and_tool_result_screenshots_become_references(uploads):
    content = [
        {"type": "text", "text": "What changed?"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64(PNG)}},
        {
            "type": "document",
            "title": "report.pdf",
            "source": {"type": "base64", "media_type": "application/pdf", "data": b64(PDF)},
        },
        {
            "type": "tool_result",
            "tool_use_id": "call-1",
            "content": [
                {"type": "text", "text": "clicked"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64(PNG)}},
            ],
        },
        {"type": "image", "source": {"type": "url", "url": "https://example.test/a.png"}},
    ]
    dumped = json.loads(await span_utils._dump_content(0, content, MagicMock()))
    assert uploads == [(PNG, "image/png", None), (PDF, "application/pdf", "report.pdf"), (PNG, "image/png", None)]
    assert dumped[0] == {"type": "text", "text": "What changed?"}
    assert dumped[1] == {"type": "artifact_ref", "sha256": "sha-1", "mime_type": "image/png"}
    assert dumped[2]["mime_type"] == "application/pdf"
    assert dumped[3]["content"][1] == {"type": "artifact_ref", "sha256": "sha-3", "mime_type": "image/png"}
    assert dumped[4] == content[4]
    assert b64(PNG) not in json.dumps(dumped)


@pytest.mark.asyncio
async def test_a_file_that_cannot_be_kept_leaves_a_placeholder(monkeypatch):
    monkeypatch.setattr(Config, "upload_file", lambda *_: None)
    content = [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "not base64!"}},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64(PNG)}},
    ]
    dumped = json.loads(await span_utils._dump_content(0, content, MagicMock()))
    assert dumped == [{"type": "text", "text": inline_files.FILE_NOT_CAPTURED}] * 2


def test_blocks_pass_through_without_an_uploader(monkeypatch):
    monkeypatch.setattr(Config, "upload_file", None)
    block = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64(PNG)}}
    assert inline_files.replace_inline_file(block) is block
