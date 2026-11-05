import asyncio
import base64
import hashlib
import logging

import aiohttp

# An upload runs inside the customer's model call, so it must never hold it up
# for long.
UPLOAD_TIMEOUT_SECONDS = 10


class ImageUploader:
    def __init__(self, base_url: str, api_key: str) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.logger = logging.getLogger(__name__)

    def upload_base64_image(
        self, trace_id: str, span_id: str, image_name: str, image_file: str
    ) -> None:
        asyncio.run(self.aupload_base64_image(trace_id, span_id, image_name, image_file))

    async def aupload_base64_image(
        self, trace_id: str, span_id: str, image_name: str, image_file: str
    ) -> str:
        """Upload an image and return its URL.

        The raw bytes go up in one request, with their size and SHA-256 up front
        so ingest can stream them into storage. Raises on any failure. Callers
        keep the message and put a placeholder where the image was, so a URL is
        only returned once the image is stored.
        """
        image = base64.b64decode(image_file, validate=True)
        timeout = aiohttp.ClientTimeout(total=UPLOAD_TIMEOUT_SECONDS)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/octet-stream",
            "x-moda-content-size": str(len(image)),
            "x-moda-content-sha256": hashlib.sha256(image).hexdigest(),
        }
        url = f"{self.base_url}/v2/traces/{trace_id}/spans/{span_id}/images/{image_name}"
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            async with session.put(url, data=image) as response:
                response.raise_for_status()
                body = await response.json()
        self.logger.info(f"Successfully uploaded image {body['url']}")
        return body["url"]  # type: ignore[no-any-return]
