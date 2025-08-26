"""
Non-table text region grouping utilities.
"""

from typing import List, Dict, Any
from worker.types import OCRToken


def group_text_regions(tokens: List[OCRToken]) -> List[Dict[str, Any]]:
    """Group tokens into simple line-level regions by y proximity."""
    tokens = sorted(tokens, key=lambda t: (t["bbox"][1], t["bbox"][0]))
    regions: List[Dict[str, Any]] = []
    current: List[OCRToken] = []
    current_y: float | None = None
    for t in tokens:
        cy = (t["bbox"][1] + t["bbox"][3]) / 2
        if current_y is None:
            current_y = cy
        if abs(cy - current_y) > 12:
            if current:
                regions.append({"tokens": current})
            current = [t]
            current_y = cy
        else:
            current.append(t)
    if current:
        regions.append({"tokens": current})
    return regions

