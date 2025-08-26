from ..types import Document
from ..pipeline.tables import extract_table_content
from ..table_reconstructor import TableReconstructor
from ...common.feature_flags import LOAD_TABLE_DETECTOR
from PIL import Image


def crop_table(image, bbox):
    return image.crop(tuple(map(int, bbox)))


def postprocess(document: Document) -> Document:
    # Optionally use heavy table detector (can be slow on CPU). Controlled by flag.
    detector = None
    if LOAD_TABLE_DETECTOR:
        try:
            from ...models.table_detector import TableDetector
            detector = TableDetector()
        except Exception:
            detector = None

    for page in document.pages:
        if not getattr(page, "image", None):
            continue

        page.tables = []

        if detector is not None:
            try:
                table_bboxes = detector.detect(page.image)
            except Exception:
                table_bboxes = []
        else:
            table_bboxes = []

        # Convert Pydantic tokens to lightweight dicts for table extraction
        dtokens = [
            {"text": t.text, "bbox": [int(t.bbox.x1), int(t.bbox.y1), int(t.bbox.x2), int(t.bbox.y2)], "confidence": float(t.confidence)}
            for t in getattr(page, "tokens", [])
        ]

        for bbox in table_bboxes:
            # Structured extraction from tokens inside bbox
            content = extract_table_content(tuple(map(int, bbox)), dtokens)
            table_image = crop_table(page.image, bbox)
            # Fallback: if columns detection is weak, try line-based reconstructor
            if not content["columns"]:
                reconstructor = TableReconstructor(table_image)
                table_data = reconstructor.reconstruct()
            else:
                # Compose simple 2D cell strings per row for preview
                table_data = [[tok["text"] for tok in row] for row in content["rows"]]

            page.tables.append({
                "bbox": list(map(int, bbox)),
                "content": content,
                "data": table_data,
            })

    return document
