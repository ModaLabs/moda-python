import asyncio
from time import monotonic

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from traceloop.sdk.images import image_uploader
from traceloop.sdk.images.image_uploader import ImageUploader


async def serve(create_status=200, upload_status=200, hang=False, create_body=None):
    """A stand-in for ingest's two-step image routes."""
    uploads = []

    async def create(request):
        if hang:
            await asyncio.sleep(30)
        assert request.headers["Authorization"] == "Bearer key"
        body = create_body if create_body is not None else {"url": str(server.make_url("/upload/shot.png"))}
        return web.json_response(body, status=create_status)

    async def upload(request):
        uploads.append(await request.json())
        return web.json_response({"success": True}, status=upload_status)

    app = web.Application()
    app.router.add_post("/v2/traces/1/spans/2/images", create)
    app.router.add_post("/upload/shot.png", upload)
    server = TestServer(app)
    await server.start_server()
    return server, uploads


async def run(server):
    uploader = ImageUploader(str(server.make_url("")).rstrip("/"), "key")
    return await uploader.aupload_base64_image("1", "2", "shot.png", "aGk=")


@pytest.mark.asyncio
async def test_uploads_and_returns_the_url():
    server, uploads = await serve()
    try:
        assert (await run(server)).endswith("/upload/shot.png")
        assert uploads == [{"image_data": "aGk="}]
    finally:
        await server.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("options", [
    {"create_status": 503},
    {"create_status": 404},
    {"upload_status": 413},
    {"create_body": {"error": "no url"}},
])
async def test_raises_when_either_step_fails(options):
    server, _ = await serve(**options)
    try:
        with pytest.raises(Exception):
            await run(server)
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_gives_up_on_a_hanging_server(monkeypatch):
    monkeypatch.setattr(image_uploader, "UPLOAD_TIMEOUT_SECONDS", 0.2)
    server, _ = await serve(hang=True)
    async def expect_upload_timeout():
        started = monotonic()
        with pytest.raises(asyncio.TimeoutError):
            await run(server)
        assert 0.2 <= monotonic() - started < 2

    try:
        # The watchdog must fail the test, rather than satisfy pytest.raises.
        await asyncio.wait_for(expect_upload_timeout(), timeout=5)
    finally:
        await server.close()
