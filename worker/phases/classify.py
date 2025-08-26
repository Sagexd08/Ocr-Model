"""Classification phase (thin wrapper)."""
from __future__ import annotations
from typing import Dict, Any
from PIL import Image


def classify_document(image: Image.Image, metadata: Dict[str, Any], model_manager) -> Dict[str, Any]:
    """Delegate to model_manager.classify_document; stable interface."""
    return model_manager.classify_document(image, metadata)

