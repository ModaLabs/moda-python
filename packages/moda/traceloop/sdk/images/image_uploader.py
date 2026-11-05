import asyncio
import base64
import concurrent.futures
import hashlib
import logging
import os
import threading
from collections import OrderedDict
from typing import Optional

import aiohttp

UPLOAD_TIMEOUT_SECONDS = 10
UPLOAD_WORKERS = 4
# Images waiting to upload, at most, so a stalled ingest can't grow memory without limit.
MAX_PENDING_BYTES = 64 * 1024 * 1024
FLUSH_TIMEOUT_SECONDS = 30
# Ingest's limit for one file (PUT /v1/files/:sha256).
MAX_FILE_BYTES = 95_000_000
# Hashes this process has sent, so a file repeated in every later prompt's
# history uploads once. Oldest forgotten first.
MAX_REMEMBERED_FILES = 10_000


class ImageUploader:
    """Uploads the files in model calls without holding up the call.

    `upload_file` hashes a file, starts uploading it to {endpoint}/v1/files/{sha256}
    and returns the `artifact_ref` block the span carries in its place. A file
    goes up once per process however many prompts repeat it.
    `aupload_base64_image` is the older per-span image route, kept for
    instrumentations that only know how to put a URL in the span.

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
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._executor_pid = os.getpid()
        self._pending: set[concurrent.futures.Future[None]] = set()
        self._pending_bytes = 0
        self._sent: "OrderedDict[str, None]" = OrderedDict()

    def upload_file(self, data: bytes, mime_type: str, name: Optional[str] = None) -> Optional[dict]:
        """Start uploading a file and return the reference block for the span.

        None when the file can't be kept (too large); the caller puts a
        placeholder in its place.
        """
        if len(data) == 0 or len(data) > MAX_FILE_BYTES:
            self.logger.warning("File of %d bytes is outside the upload limit; skipping it", len(data))
            return None
        sha256 = hashlib.sha256(data).hexdigest()
        ref = {"type": "artifact_ref", "sha256": sha256, "mime_type": mime_type, "size_bytes": len(data)}
        if name:
            ref["name"] = name
        self._reset_after_fork()
        with self._lock:
            sent = sha256 in self._sent
            if sent:
                self._sent.move_to_end(sha256)
        if not sent:
            self._submit_file(sha256, data, mime_type)
        return ref

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
        self._reset_after_fork()
        with self._lock:
            pending = list(self._pending)
        if pending:
            concurrent.futures.wait(pending, timeout=timeout)

    def _submit(self, url: str, image_file: str) -> None:
        self._reset_after_fork()
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

    def _submit_file(self, sha256: str, data: bytes, mime_type: str) -> None:
        url = f"{self.base_url}/v1/files/{sha256}"
        size = len(data)
        with self._lock:
            workers = self._workers()
            if self._pending_bytes + size > MAX_PENDING_BYTES:
                self.logger.warning("Too many files waiting to upload; skipping %s", url)
                return
            # Remembered now so the next prompt doesn't send it again; forgotten if it fails.
            self._remember(sha256)
            self._pending_bytes += size
            future = workers.submit(self._send_file, url, sha256, data, mime_type, size)
            self._pending.add(future)
        future.add_done_callback(self._forget)

    def _remember(self, sha256: str) -> None:
        self._sent[sha256] = None
        while len(self._sent) > MAX_REMEMBERED_FILES:
            self._sent.popitem(last=False)

    def _send_file(self, url: str, sha256: str, data: bytes, mime_type: str, size: int) -> None:
        try:
            asyncio.run(self._put_file(url, data, mime_type))
        except Exception as e:
            self.logger.warning("File upload failed for %s: %r", url, e)
            with self._lock:
                self._sent.pop(sha256, None)
        finally:
            with self._lock:
                self._pending_bytes -= size

    async def _put_file(self, url: str, data: bytes, mime_type: str) -> None:
        """Send the raw bytes, with their size up front so ingest can stream them; the URL holds the hash."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": mime_type,
            "x-moda-content-size": str(len(data)),
        }
        timeout = aiohttp.ClientTimeout(total=UPLOAD_TIMEOUT_SECONDS)
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            async with session.put(url, data=data) as response:
                response.raise_for_status()

    def _forget(self, future: concurrent.futures.Future[None]) -> None:
        with self._lock:
            self._pending.discard(future)

    def _reset_after_fork(self) -> None:
        # Check before acquiring the lock: its owning thread may only exist in
        # the parent. Inherited futures and workers cannot finish in this child.
        if self._executor_pid != os.getpid():
            self._lock = threading.Lock()
            self._executor = None
            self._pending = set()
            self._pending_bytes = 0
            self._sent = OrderedDict()
            self._executor_pid = os.getpid()

    def _workers(self) -> concurrent.futures.ThreadPoolExecutor:
        if self._executor is None:
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
