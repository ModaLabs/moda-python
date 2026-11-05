import asyncio
import base64
import concurrent.futures
import hashlib
import logging
import os
import threading

import aiohttp

UPLOAD_TIMEOUT_SECONDS = 10
UPLOAD_WORKERS = 4
# Images waiting to upload, at most, so a stalled ingest can't grow memory without limit.
MAX_PENDING_BYTES = 64 * 1024 * 1024
FLUSH_TIMEOUT_SECONDS = 30


class ImageUploader:
    """Uploads the images in a prompt without holding up the model call.

    An upload takes about a second, and it would otherwise run before every model
    call that carries a screenshot. The image's URL doesn't depend on the upload,
    so the span gets it straight away and the upload runs on a worker thread.
    `flush()` waits for what's in progress. A failed upload is logged and never
    reaches the caller, so the message stays in the span; its image is just missing.
    """

    def __init__(self, base_url: str, api_key: str) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.logger = logging.getLogger(__name__)
        self._lock = threading.Lock()
        self._executor = None
        self._executor_pid = None
        self._pending = set()
        self._pending_bytes = 0

    def upload_base64_image(
        self, trace_id: str, span_id: str, image_name: str, image_file: str
    ) -> None:
        asyncio.run(self.aupload_base64_image(trace_id, span_id, image_name, image_file))

    async def aupload_base64_image(
        self, trace_id: str, span_id: str, image_name: str, image_file: str
    ) -> str:
        """Start uploading an image and return the URL it will have."""
        url = f"{self.base_url}/v2/traces/{trace_id}/spans/{span_id}/images/{image_name}"
        self._submit(url, image_file)
        return url

    def flush(self, timeout: float = FLUSH_TIMEOUT_SECONDS) -> None:
        """Wait for uploads in progress, so a short-lived process doesn't lose them."""
        with self._lock:
            pending = list(self._pending)
        if pending:
            concurrent.futures.wait(pending, timeout=timeout)

    def _submit(self, url: str, image_file: str) -> None:
        size = len(image_file)
        with self._lock:
            workers = self._workers()
            if self._pending_bytes + size > MAX_PENDING_BYTES:
                self.logger.warning("Too many images waiting to upload; skipping %s", url)
                return
            self._pending_bytes += size
            future = workers.submit(self._send, url, image_file, size)
            self._pending.add(future)
        future.add_done_callback(self._forget)

    def _forget(self, future: concurrent.futures.Future) -> None:
        with self._lock:
            self._pending.discard(future)

    def _workers(self) -> concurrent.futures.ThreadPoolExecutor:
        # Threads don't survive a fork, so a forked worker process needs its own pool.
        if self._executor is None or self._executor_pid != os.getpid():
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=UPLOAD_WORKERS, thread_name_prefix="moda-image-upload"
            )
            self._executor_pid = os.getpid()
            self._pending = set()
            self._pending_bytes = 0
        return self._executor

    def _send(self, url: str, image_file: str, size: int) -> None:
        try:
            asyncio.run(self._put(url, image_file))
        except Exception as e:
            self.logger.warning("Image upload failed for %s: %r", url, e)
        finally:
            # Here, not in a done callback, so the room is free once flush() returns.
            with self._lock:
                self._pending_bytes -= size

    async def _put(self, url: str, image_file: str) -> None:
        """Send the raw bytes, with their size and SHA-256 up front so ingest can stream them."""
        image = base64.b64decode(image_file, validate=True)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/octet-stream",
            "x-moda-content-size": str(len(image)),
            "x-moda-content-sha256": hashlib.sha256(image).hexdigest(),
        }
        timeout = aiohttp.ClientTimeout(total=UPLOAD_TIMEOUT_SECONDS)
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            async with session.put(url, data=image) as response:
                response.raise_for_status()
        self.logger.info(f"Successfully uploaded image {url}")
