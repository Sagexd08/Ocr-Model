from typing import Any, Dict
from PIL import Image
from pathlib import Path

from ...types import Document, Page
from ...utils.logging import get_logger

logger = get_logger(__name__)


class ImageIngestion:
    """Pipeline processor: load an image from document.metadata['source_path'] into Document.pages[0].image."""

    def __init__(self):
        pass

    def process(self, document: Document) -> Document:
        src = document.metadata.get("source_path")
        if not src:
            logger.warning("ImageIngestion called without source_path")
            return document
        if document.metadata.get("doc_type") != "image":
            # Not an image document
            return document
        try:
            img = Image.open(Path(src)).convert("RGB")
            if not document.pages:
                document.pages.append(Page(page_num=1, image=img, width=img.width, height=img.height))
            else:
                p = document.pages[0]
                p.image = img
                p.width = p.width or img.width
                p.height = p.height or img.height
        except Exception as e:
            logger.warning(f"Failed to ingest image: {e}")
        return document

