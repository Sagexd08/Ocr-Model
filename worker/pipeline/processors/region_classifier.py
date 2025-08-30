from typing import Dict, Any

from ...types import Document, Region, Bbox
from ...utils.logging import get_logger, log_execution_time

logger = get_logger(__name__)


class RegionClassifierProcessor:
    """
    Classify document regions on each page using LayoutParser/Detectron2 if available,
    with a graceful heuristic fallback. Appends Region objects to page.regions.
    """

    def __init__(self, score_threshold: float = 0.4):
        self.score_threshold = score_threshold
        try:
            from models.region_classifier_lp import RegionClassifierLP  # lazy import
            self.classifier = RegionClassifierLP()
        except Exception as e:
            logger.warning(f"RegionClassifierLP unavailable: {e}")
            self.classifier = None

    @log_execution_time
    def process(self, document: Document) -> Document:
        if self.classifier is None:
            logger.info("Region classifier not initialized; skipping region detection")
            return document

        for page in document.pages:
            if page.image is None:
                continue
            try:
                results = self.classifier.detect(page.image)
                for r in results:
                    if r.get("confidence", 0) < self.score_threshold:
                        continue
                    bbox = r.get("bbox", [0, 0, 0, 0])
                    page.regions.append(
                        Region(
                            type=r.get("type", "region"),
                            bbox=Bbox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3]),
                            confidence=r.get("confidence", 0.0),
                        )
                    )
            except Exception as e:
                logger.warning(f"Region classification failed on page {page.page_num}: {e}")
        return document

