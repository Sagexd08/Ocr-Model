"""
IO conversion utilities for CurioScan worker pipeline.
"""

from typing import List
from PIL import Image
import pdf2image
import mimetypes
import logging

logger = logging.getLogger(__name__)


def detect_mime_type(file_content: bytes, file_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type:
        return mime_type
    if file_content.startswith(b"%PDF"):
        return "application/pdf"
    if file_content.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if file_content.startswith(b"\x89PNG"):
        return "image/png"
    if file_content.startswith(b"PK"):
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    return "application/octet-stream"


def convert_to_images(file_content: bytes, file_path: str) -> List[Image.Image]:
    import io
    mime = detect_mime_type(file_content, file_path)
    try:
        if mime == "application/pdf":
            return pdf2image.convert_from_bytes(file_content, dpi=300)
        if mime.startswith("image/"):
            return [Image.open(io.BytesIO(file_content)).convert("RGB")]
        return [Image.new("RGB", (800, 1000), "white")]
    except Exception as e:
        logger.error(f"Failed to convert file to images: {e}")
        return [Image.new("RGB", (800, 1000), "white")]


def convert_to_image(file_content: bytes, file_path: str) -> Image.Image:
    images = convert_to_images(file_content, file_path)
    return images[0] if images else Image.new("RGB", (100, 100), "white")

