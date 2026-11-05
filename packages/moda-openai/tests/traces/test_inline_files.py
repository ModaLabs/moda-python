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
        ("image/png", None),
        ("audio/wav", None),
        ("application/pdf", "a.pdf"),
        ("image/png", None),
        ("text/csv", "t.csv"),
    ]
    assert [block["type"] for block in replaced] == ["text"] + ["artifact_ref"] * 5 + ["image_url", "input_image"]
    assert b64(PNG) not in json.dumps(replaced)


def test_generated_images_and_spoken_audio_are_kept_as_files(uploads):
    image = inline_files.replace_inline_file(
        {"type": "image_generation_call", "result": b64(PNG), "output_format": "jpg"}
    )
    assert image["type"] == "artifact_ref" and uploads[-1][1] == "image/jpeg"

    spoken = json.loads(
        inline_files.audio_output_content({"content": None, "audio": {"data": b64(MP3), "transcript": "Hello"}})
    )
    assert spoken[0] == {"type": "text", "text": "Hello"}
    assert spoken[1]["type"] == "artifact_ref" and uploads[-1][1] == "audio/mpeg"
    assert inline_files.audio_output_content({"content": "plain"}) == "plain"


def test_a_file_that_cannot_be_kept_leaves_a_placeholder(monkeypatch):
    monkeypatch.setattr(Config, "upload_file", lambda *_: None)
    placeholder = {"type": "text", "text": inline_files.FILE_NOT_CAPTURED}
    assert (
        inline_files.replace_inline_file({"type": "image_url", "image_url": {"url": "data:image/png;base64,@@"}})
        == placeholder
    )
    assert (
        inline_files.replace_inline_file(
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64(PNG)}"}}
        )
        == placeholder
    )


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
    await _set_prompts(
        span,
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is on screen?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64(PNG)}"}},
                ],
            }
        ],
    )
    content = json.loads(span.attributes["gen_ai.prompt.0.content"])
    assert content[1] == {"type": "artifact_ref", "sha256": "sha-1", "mime_type": "image/png"}


def test_responses_computer_screenshots_and_generated_images_become_references(uploads):
    from opentelemetry.instrumentation.openai.v1.responses_wrappers import TracedData, set_data_attributes

    traced = TracedData.model_construct(
        start_time=0,
        response_id="resp",
        tools=None,
        instructions=None,
        usage=None,
        input=[
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_image", "image_url": f"data:image/png;base64,{b64(PNG)}"}],
            },
            {
                "type": "computer_call_output",
                "call_id": "c1",
                "output": {"type": "computer_screenshot", "image_url": f"data:image/png;base64,{b64(PNG)}"},
            },
        ],
        output_blocks={
            "img": {"type": "image_generation_call", "id": "img", "result": b64(PNG), "status": "completed"}
        },
        output_text="Here it is",
        response_model=None,
        request_reasoning_summary=None,
        request_reasoning_effort=None,
        response_reasoning_effort=None,
        response_service_tier=None,
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



def _responses_turn():
    from opentelemetry.instrumentation.openai.v1.responses_wrappers import TracedData

    return TracedData.model_construct(
        start_time=0, response_id="resp", tools=None, instructions=None, usage=None,
        input=[{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "What does the report say?"},
            {"type": "input_file", "filename": "r.pdf", "file_data": f"data:application/pdf;base64,{b64(b'%PDF')}"},
        ]}],
        output_blocks={}, output_text="Revenue is up.", response_model=None, request_reasoning_summary=None,
        request_reasoning_effort=None, response_reasoning_effort=None, response_service_tier=None,
    )


def test_responses_input_text_is_recorded_as_text(uploads):
    from opentelemetry.instrumentation.openai.v1.responses_wrappers import set_data_attributes

    span = RecordingSpan()
    set_data_attributes(_responses_turn(), span)
    content = json.loads(span.attributes["gen_ai.prompt.0.content"])
    assert content[0] == {"type": "text", "text": "What does the report say?"}
    assert content[1]["type"] == "artifact_ref"


def test_responses_turns_join_their_conversation(uploads):
    context = pytest.importorskip("traceloop.sdk.context")
    from opentelemetry.instrumentation.openai.v1.responses_wrappers import set_data_attributes

    span = RecordingSpan()
    with context.set_conversation_id("conv-e2e"):
        set_data_attributes(_responses_turn(), span)
    assert span.attributes["moda.conversation_id"] == "conv-e2e"


@pytest.mark.asyncio
@pytest.mark.parametrize("async_stream", [False, True])
@pytest.mark.parametrize("other_user", [False, True])
async def test_response_stream_keeps_request_ids(async_stream, other_user):
    from contextlib import ExitStack
    from unittest.mock import MagicMock
    from opentelemetry.instrumentation.openai.v1.responses_wrappers import ResponseStream

    context = pytest.importorskip("traceloop.sdk.context")

    class EmptyStream:
        def __iter__(self):
            return self

        def __next__(self):
            raise StopIteration

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    span = MagicMock()
    with context.set_user_id("request-user"), context.set_conversation_id("request-conversation"):
        stream = ResponseStream(
            span=span, response=EmptyStream(), start_time=1,
            request_kwargs={"input": "hello", "model": "gpt-4o"},
        )
    with ExitStack() as scopes:
        if other_user:
            scopes.enter_context(context.set_user_id("consumer-user"))
            scopes.enter_context(context.set_conversation_id("consumer-conversation"))
        if async_stream:
            async for _ in stream:
                pass
        else:
            list(stream)
    span.set_attribute.assert_any_call("moda.user_id", "request-user")
    span.set_attribute.assert_any_call("moda.conversation_id", "request-conversation")
    assert not any("consumer-user" in call.args or "consumer-conversation" in call.args
                   for call in span.set_attribute.call_args_list)


def test_responses_without_request_user_do_not_adopt_consumer_ids(uploads):
    context = pytest.importorskip("traceloop.sdk.context")
    from traceloop.sdk.conversation import compute_conversation_id
    from opentelemetry.instrumentation.openai.v1.responses_wrappers import set_data_attributes

    turn = _responses_turn()
    expected = compute_conversation_id([
        {"role": "system", "content": turn.instructions},
        {"role": "user", "content": turn.input[0]["content"]},
    ])
    span = RecordingSpan()
    with context.set_user_id("consumer-user"), context.set_conversation_id("consumer-conversation"):
        set_data_attributes(turn, span)
    assert "moda.user_id" not in span.attributes
    assert span.attributes["moda.conversation_id"] == expected


def test_cached_response_does_not_retain_unrelated_context_values():
    import gc
    import weakref
    from contextvars import ContextVar
    from opentelemetry.instrumentation.openai.v1.responses_wrappers import responses

    pytest.importorskip("traceloop.sdk.context")

    class Request:
        pass

    request_var = ContextVar("unrelated_request")
    request = Request()
    reference = weakref.ref(request)
    token = request_var.set(request)
    try:
        turn = _responses_turn()
        responses["memory-regression"] = turn
    finally:
        request_var.reset(token)
    del request
    try:
        gc.collect()
        assert reference() is None
        assert responses["memory-regression"] is turn
    finally:
        responses.pop("memory-regression", None)
