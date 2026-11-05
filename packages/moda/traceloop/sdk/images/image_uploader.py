import asyncio
import logging

import aiohttp

# An upload runs inside the customer's model call, so it must never hold it up
# for long. Each of the two requests gets this long.
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

        Raises on any failure. Callers keep the message and put a placeholder
        where the image was, so a URL is only returned once the image is stored.
        """
        timeout = aiohttp.ClientTimeout(total=UPLOAD_TIMEOUT_SECONDS)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            url = await self._get_image_url(session, trace_id, span_id, image_name)
            await self._upload(session, url, image_file)
        return url

    async def _get_image_url(
        self, session: aiohttp.ClientSession, trace_id: str, span_id: str, image_name: str
    ) -> str:
        async with session.post(
            f"{self.base_url}/v2/traces/{trace_id}/spans/{span_id}/images",
            json={"image_name": image_name},
        ) as response:
            response.raise_for_status()
            body = await response.json()
        return body["url"]  # type: ignore[no-any-return]

    async def _upload(self, session: aiohttp.ClientSession, url: str, base64_image: str) -> None:
        async with session.post(url, json={"image_data": base64_image}) as response:
            response.raise_for_status()
        self.logger.info(f"Successfully uploaded image {url}")
