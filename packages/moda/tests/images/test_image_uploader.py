import asyncio
import base64
import binascii
import hashlib
from time import monotonic

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from traceloop.sdk.images import image_uploader
from traceloop.sdk.images.image_uploader import ImageUploader

IMAGE = b"\x89PNG\r\n\x1a\n raw image"
ENCODED = base64.b64encode(IMAGE).decode()


async def serve(status=200, hang=False, body=None):
    """A stand-in for ingest's raw image route."""
    uploads = []

    async def upload(request):
        if hang:
            await asyncio.sleep(30)
        uploads.append((dict(request.headers), await request.read()))
        return web.json_response(body if body is not None else {"url": "https://ingest.test/shot.png"}, status=status)

    app = web.Application()
    app.router.add_put("/v1/traces/v2/traces/1/spans/2/images/shot.png", upload)
    server = TestServer(app)
    await server.start_server()
    return server, uploads


async def run(server, image=ENCODED):
    uploader = ImageUploader(str(server.make_url("/v1/traces")), "key")
    return await uploader.aupload_base64_image("1", "2", "shot.png", image)


@pytest.mark.asyncio
async def test_sends_raw_bytes_with_size_and_digest_in_one_request():
    server, uploads = await serve()
    try:
        assert await run(server) == "https://ingest.test/shot.png"
        assert len(uploads) == 1
        headers, received = uploads[0]
        assert received == IMAGE
        assert headers["Authorization"] == "Bearer key"
        assert headers["x-moda-content-size"] == str(len(IMAGE))
        assert headers["x-moda-content-sha256"] == hashlib.sha256(IMAGE).hexdigest()
    finally:
        await server.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("options", [
    {"status": 503},
    {"status": 404},
    {"status": 413},
    {"body": {"error": "no url"}},
])
async def test_raises_when_the_upload_fails(options):
    server, _ = await serve(**options)
    try:
        with pytest.raises(Exception):
            await run(server)
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_raises_on_invalid_base64_without_sending_anything():
    server, uploads = await serve()
    try:
        with pytest.raises(binascii.Error):
            await run(server, "not base64!")
        assert uploads == []
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
