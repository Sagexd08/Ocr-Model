"""Preprocess phase."""
from __future__ import annotations
from typing import Dict, Any, List
from PIL import Image

from worker.pipeline.image_preproc import preprocess_image


def preprocess_pages(images: List[Image.Image]) -> List[Dict[str, Any]]:
    out = []
    for i, image in enumerate(images):
        img2 = preprocess_image(image)
        out.append({
            "page": i + 1,
            "image": img2,
            "original_size": image.size,
        })
    return out

