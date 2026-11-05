import base64
import json

import pytest

from opentelemetry.instrumentation.openai.shared import inline_files
from opentelemetry.instrumentation.openai.shared.config import Config

PNG = b"\x89PNG\r\n\x1a\n shot"
WAV = b"RIFF\x00\x00\x00\x00WAVEfmt "
MP3 = b"ID3\x04\x00 song"


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


def test_chat_and_responses_parts_become_references(uploads):
    data_url = f"data:image/png;base64,{b64(PNG)}"
    content = [
        {"type": "text", "text": "Look"},
        {"type": "image_url", "image_url": {"url": data_url}},
        {"type": "input_audio", "input_audio": {"data": b64(WAV), "format": "wav"}},
        {"type": "file", "file": {"file_data": f"data:application/pdf;base64,{b64(b'%PDF')}", "filename": "a.pdf"}},
        {"type": "input_image", "image_url": data_url},
        {"type": "input_file", "file_data": f"data:text/csv;base64,{b64(b'a,b')}", "filename": "t.csv"},
        {"type": "image_url", "image_url": {"url": "https://example.test/a.png"}},
        {"type": "input_image", "file_id": "file-123"},
    ]
    replaced = inline_files.replace_inline_files(content)
    assert [(mime, name) for _, mime, name in uploads] == [
        ("image/png", None), ("audio/wav", None), ("application/pdf", "a.pdf"), ("image/png", None), ("text/csv", "t.csv"),
    ]
    assert [block["type"] for block in replaced] == ["text"] + ["artifact_ref"] * 5 + ["image_url", "input_image"]
    assert b64(PNG) not in json.dumps(replaced)


def test_generated_images_and_spoken_audio_are_kept_as_files(uploads):
    image = inline_files.replace_inline_file({"type": "image_generation_call", "result": b64(PNG), "output_format": "jpg"})
    assert image["type"] == "artifact_ref" and uploads[-1][1] == "image/jpeg"

    spoken = json.loads(inline_files.audio_output_content({"content": None, "audio": {"data": b64(MP3), "transcript": "Hello"}}))
    assert spoken[0] == {"type": "text", "text": "Hello"}
    assert spoken[1]["type"] == "artifact_ref" and uploads[-1][1] == "audio/mpeg"
    assert inline_files.audio_output_content({"content": "plain"}) == "plain"


def test_a_file_that_cannot_be_kept_leaves_a_placeholder(monkeypatch):
    monkeypatch.setattr(Config, "upload_file", lambda *_: None)
    placeholder = {"type": "text", "text": inline_files.FILE_NOT_CAPTURED}
    assert inline_files.replace_inline_file({"type": "image_url", "image_url": {"url": "data:image/png;base64,@@"}}) == placeholder
    assert inline_files.replace_inline_file({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64(PNG)}"}}) == placeholder


def test_blocks_pass_through_without_an_uploader(monkeypatch):
    monkeypatch.setattr(Config, "upload_file", None)
    block = {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64(PNG)}"}}
    assert inline_files.replace_inline_file(block) is block


class RecordingSpan:
    def __init__(self):
        self.attributes = {}

    def is_recording(self):
        return True

    def set_attribute(self, name, value):
        self.attributes[name] = value


@pytest.mark.asyncio
async def test_chat_prompt_attributes_carry_references_not_bytes(uploads):
    from opentelemetry.instrumentation.openai.shared.chat_wrappers import _set_prompts

    span = RecordingSpan()
    await _set_prompts(span, [{"role": "user", "content": [
        {"type": "text", "text": "What is on screen?"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64(PNG)}"}},
    ]}])
    content = json.loads(span.attributes["gen_ai.prompt.0.content"])
    assert content[1] == {"type": "artifact_ref", "sha256": "sha-1", "mime_type": "image/png"}


def test_responses_computer_screenshots_and_generated_images_become_references(uploads):
    from opentelemetry.instrumentation.openai.v1.responses_wrappers import TracedData, set_data_attributes

    traced = TracedData.model_construct(
        start_time=0, response_id="resp", tools=None, instructions=None, usage=None,
        input=[
            {"type": "message", "role": "user", "content": [{"type": "input_image", "image_url": f"data:image/png;base64,{b64(PNG)}"}]},
            {"type": "computer_call_output", "call_id": "c1", "output": {"type": "computer_screenshot", "image_url": f"data:image/png;base64,{b64(PNG)}"}},
        ],
        output_blocks={"img": {"type": "image_generation_call", "id": "img", "result": b64(PNG), "status": "completed"}},
        output_text="Here it is",
        response_model=None, request_reasoning_summary=None, request_reasoning_effort=None,
        response_reasoning_effort=None, response_service_tier=None,
    )
    span = RecordingSpan()
    set_data_attributes(traced, span)
    serialized = json.dumps(span.attributes)
    assert b64(PNG) not in serialized
    assert json.loads(span.attributes["gen_ai.prompt.0.content"])[0]["type"] == "artifact_ref"
    assert json.loads(span.attributes["gen_ai.prompt.1.content"])[0]["type"] == "artifact_ref"
    completion = json.loads(span.attributes["gen_ai.completion.0.content"])
    assert completion[0] == {"type": "text", "text": "Here it is"}
    assert completion[1]["type"] == "artifact_ref"
