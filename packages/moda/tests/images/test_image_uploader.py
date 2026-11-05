import asyncio
import base64
import concurrent.futures
import hashlib
import logging
import multiprocessing
import threading
from time import monotonic

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from traceloop.sdk.images import image_uploader
from traceloop.sdk.images.image_uploader import ImageUploader

IMAGE = b"\x89PNG\r\n\x1a\n raw image"
ENCODED = base64.b64encode(IMAGE).decode()
PATH = "/v1/traces/v2/traces/1/spans/2/images/shot.png"


async def serve(status=200, release=None):
    """A stand-in for ingest's raw image route. `release` holds requests until it is set."""
    uploads = []

    async def upload(request):
        body = await request.read()
        if release is not None:
            while not release.is_set():
                await asyncio.sleep(0.01)
        uploads.append((dict(request.headers), body))
        return web.json_response({"url": "unused"}, status=status)

    app = web.Application()
    app.router.add_put(PATH, upload)
    server = TestServer(app)
    await server.start_server()
    return server, uploads


def uploader_for(server):
    return ImageUploader(str(server.make_url("/v1/traces")), "key")


async def flush(uploader):
    await asyncio.to_thread(uploader.flush)


@pytest.mark.asyncio
async def test_sends_raw_bytes_with_size_and_digest_in_one_request():
    server, uploads = await serve()
    try:
        uploader = uploader_for(server)
        url = await uploader.aupload_base64_image("1", "2", "shot.png", ENCODED)
        await flush(uploader)
        assert url == str(server.make_url(PATH))
        assert len(uploads) == 1
        headers, received = uploads[0]
        assert received == IMAGE
        assert headers["Authorization"] == "Bearer key"
        assert headers["x-moda-content-size"] == str(len(IMAGE))
        assert headers["x-moda-content-sha256"] == hashlib.sha256(IMAGE).hexdigest()
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_returns_the_url_before_the_upload_finishes_and_flush_waits_for_it():
    release = threading.Event()
    server, uploads = await serve(release=release)
    try:
        uploader = uploader_for(server)
        started = monotonic()
        await uploader.aupload_base64_image("1", "2", "shot.png", ENCODED)
        assert monotonic() - started < 0.5
        assert uploads == []

        waiting = asyncio.create_task(flush(uploader))
        await asyncio.sleep(0.05)
        assert not waiting.done()
        release.set()
        await waiting
        assert len(uploads) == 1
    finally:
        release.set()
        await server.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [503, 404, 413])
async def test_a_failed_upload_never_reaches_the_caller(status, caplog):
    server, _ = await serve(status=status)
    try:
        uploader = uploader_for(server)
        with caplog.at_level(logging.WARNING):
            url = await uploader.aupload_base64_image("1", "2", "shot.png", ENCODED)
            await flush(uploader)
        assert url.endswith(PATH)
        assert "Image upload failed" in caplog.text
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_invalid_base64_is_logged_without_sending_anything(caplog):
    server, uploads = await serve()
    try:
        uploader = uploader_for(server)
        with caplog.at_level(logging.WARNING):
            await uploader.aupload_base64_image("1", "2", "shot.png", "not base64!")
            await flush(uploader)
        assert uploads == []
        assert "Image upload failed" in caplog.text
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_gives_up_on_a_hanging_server(monkeypatch, caplog):
    monkeypatch.setattr(image_uploader, "UPLOAD_TIMEOUT_SECONDS", 0.2)
    release = threading.Event()
    server, _ = await serve(release=release)
    try:
        uploader = uploader_for(server)
        started = monotonic()
        with caplog.at_level(logging.WARNING):
            await uploader.aupload_base64_image("1", "2", "shot.png", ENCODED)
            await flush(uploader)
        assert 0.2 <= monotonic() - started < 2
        assert "Image upload failed" in caplog.text
    finally:
        release.set()
        await server.close()


@pytest.mark.asyncio
async def test_skips_images_past_the_backlog_limit_then_recovers(monkeypatch, caplog):
    monkeypatch.setattr(image_uploader, "MAX_PENDING_BYTES", len(ENCODED))
    release = threading.Event()
    server, uploads = await serve(release=release)
    try:
        uploader = uploader_for(server)
        with caplog.at_level(logging.WARNING):
            await uploader.aupload_base64_image("1", "2", "shot.png", ENCODED)
            await uploader.aupload_base64_image("1", "2", "shot.png", ENCODED)
        assert "Too many images" in caplog.text
        release.set()
        await flush(uploader)
        assert len(uploads) == 1

        await uploader.aupload_base64_image("1", "2", "shot.png", ENCODED)
        await flush(uploader)
        assert len(uploads) == 2
    finally:
        release.set()
        await server.close()


@pytest.mark.asyncio
async def test_a_forked_process_gets_its_own_workers(monkeypatch):
    server, uploads = await serve()
    try:
        uploader = uploader_for(server)
        await uploader.aupload_base64_image("1", "2", "shot.png", ENCODED)
        await flush(uploader)
        before = uploader._executor
        monkeypatch.setattr(image_uploader.os, "getpid", lambda: -1)
        await uploader.aupload_base64_image("1", "2", "shot.png", ENCODED)
        await flush(uploader)
        assert uploader._executor is not before
        assert len(uploads) == 2
    finally:
        await server.close()


@pytest.mark.skipif("fork" not in multiprocessing.get_all_start_methods(), reason="requires fork")
@pytest.mark.parametrize("first_action", ["flush", "submit"])
def test_fork_with_lock_held_resets_upload_state(first_action, monkeypatch):
    uploader = ImageUploader("http://unused", "key")
    before = uploader._workers()
    uploader._pending.add(concurrent.futures.Future())
    uploader._pending_bytes = image_uploader.MAX_PENDING_BYTES
    sent = []

    async def put(url, image_file):
        sent.append((url, image_file))

    monkeypatch.setattr(uploader, "_put", put)
    locked = threading.Event()
    release = threading.Event()

    def hold_lock():
        with uploader._lock:
            locked.set()
            release.wait()

    def child():
        if first_action == "flush":
            uploader.flush(timeout=0)
            assert not uploader._pending
            assert uploader._pending_bytes == 0
        uploader.upload_base64_image("1", "2", "shot.png", ENCODED)
        uploader.flush(timeout=2)
        assert uploader._executor is not before
        assert len(sent) == 1
        assert uploader._pending_bytes == 0
        uploader._executor.shutdown()

    holder = threading.Thread(target=hold_lock)
    holder.start()
    process = multiprocessing.get_context("fork").Process(target=child)
    try:
        assert locked.wait(timeout=2)
        process.start()
        process.join(timeout=5)
        assert process.exitcode == 0, "child upload/flush hung or failed after fork"
        assert uploader._executor is before
        assert uploader._pending_bytes == image_uploader.MAX_PENDING_BYTES
    finally:
        if process.pid is not None:
            if process.is_alive():
                process.terminate()
            process.join(timeout=2)
            process.close()
        release.set()
        holder.join(timeout=2)
        before.shutdown()


async def serve_files(status=200):
    """A stand-in for ingest's file route, PUT /v1/files/:sha256."""
    uploads = []

    async def upload(request):
        uploads.append((request.match_info["sha"], dict(request.headers), await request.read()))
        return web.json_response({}, status=status)

    app = web.Application()
    app.router.add_put("/v1/traces/v1/files/{sha}", upload)
    server = TestServer(app)
    await server.start_server()
    return server, uploads


@pytest.mark.asyncio
async def test_upload_file_returns_the_reference_and_sends_the_bytes_under_their_hash():
    server, uploads = await serve_files()
    try:
        uploader = uploader_for(server)
        video = b"\x00\x00\x00\x18ftypmp42 some video"
        ref = uploader.upload_file(video, "video/mp4", "run.mp4")
        await flush(uploader)
        sha = hashlib.sha256(video).hexdigest()
        assert ref == {
            "type": "artifact_ref",
            "sha256": sha,
            "mime_type": "video/mp4",
            "size_bytes": len(video),
            "name": "run.mp4",
        }
        assert len(uploads) == 1
        path_sha, headers, body = uploads[0]
        assert path_sha == sha and body == video
        assert headers["Content-Type"] == "video/mp4"
        assert headers["x-moda-content-size"] == str(len(video))
        assert headers["Authorization"] == "Bearer key"
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_upload_file_sends_a_repeated_file_once():
    server, uploads = await serve_files()
    try:
        uploader = uploader_for(server)
        for _ in range(5):
            uploader.upload_file(IMAGE, "image/png")
        await flush(uploader)
        assert len(uploads) == 1
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_upload_file_tries_again_after_a_failed_upload(caplog):
    server, uploads = await serve_files(status=503)
    try:
        uploader = uploader_for(server)
        with caplog.at_level(logging.WARNING):
            ref = uploader.upload_file(IMAGE, "image/png")
            await flush(uploader)
            uploader.upload_file(IMAGE, "image/png")
            await flush(uploader)
        assert ref["sha256"] == hashlib.sha256(IMAGE).hexdigest()
        assert len(uploads) == 2
        assert "File upload failed" in caplog.text
    finally:
        await server.close()


def test_upload_file_skips_empty_and_oversized_files(monkeypatch):
    uploader = ImageUploader("http://127.0.0.1:9", "key")
    monkeypatch.setattr(image_uploader, "MAX_FILE_BYTES", 4)
    assert uploader.upload_file(b"", "image/png") is None
    assert uploader.upload_file(b"12345", "image/png") is None


@pytest.mark.asyncio
async def test_upload_file_sends_a_concurrent_repeated_file_once(monkeypatch):
    server, uploads = await serve_files()
    uploader = uploader_for(server)
    callers = 8
    barrier = threading.Barrier(callers)
    submit_file = uploader._submit_file

    def submit_together(*args):
        # Force every caller to reach submission before any hash is reserved.
        barrier.wait(timeout=5)
        return submit_file(*args)

    monkeypatch.setattr(uploader, "_submit_file", submit_together)

    def upload_concurrently():
        with concurrent.futures.ThreadPoolExecutor(max_workers=callers) as pool:
            return list(pool.map(lambda _: uploader.upload_file(IMAGE, "image/png"), range(callers)))

    try:
        refs = await asyncio.to_thread(upload_concurrently)
        await flush(uploader)
        assert len(uploads) == 1
        assert all(ref == refs[0] for ref in refs)
        assert refs[0]["sha256"] == hashlib.sha256(IMAGE).hexdigest()
        assert uploader._pending_bytes == 0
    finally:
        await flush(uploader)
        await server.close()


@pytest.mark.asyncio
async def test_upload_file_queue_rejection_returns_none_and_can_retry(monkeypatch):
    monkeypatch.setattr(image_uploader, "MAX_PENDING_BYTES", len(IMAGE))
    uploader = ImageUploader("http://unused", "key")
    release = threading.Event()
    sent = []

    async def put(url, data, mime_type):
        assert await asyncio.to_thread(release.wait, 5)
        sent.append(data)

    monkeypatch.setattr(uploader, "_put_file", put)
    try:
        ref = uploader.upload_file(IMAGE, "image/png")
        assert uploader.upload_file(IMAGE, "image/png") == ref
        assert uploader.upload_file(b"other", "text/plain") is None
        assert hashlib.sha256(b"other").hexdigest() not in uploader._sent
        release.set()
        await flush(uploader)
        assert uploader.upload_file(b"other", "text/plain") is not None
        await flush(uploader)
        assert sent == [IMAGE, b"other"]
        assert uploader._pending_bytes == 0
    finally:
        release.set()
        await flush(uploader)


def test_file_limit_fits_upload_queue():
    assert image_uploader.MAX_FILE_BYTES <= image_uploader.MAX_PENDING_BYTES


def test_upload_file_worker_rejection_does_not_reserve_hash():
    uploader = ImageUploader("http://unused", "key")
    uploader._workers().shutdown()
    assert uploader.upload_file(IMAGE, "image/png") is None
    assert not uploader._sent
    assert not uploader._pending
    assert uploader._pending_bytes == 0
