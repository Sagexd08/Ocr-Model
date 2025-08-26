"""
PDF-native text extraction utilities and CPU-friendly OCR tokenization.
"""

from typing import List
from worker.types import OCRToken, OCRPage
import pdfplumber
from PIL import Image
import logging
import pytesseract
import numpy as np

logger = logging.getLogger(__name__)


def extract_pdf_native_text(file_content: bytes) -> List[OCRPage]:
    """Extract text and bounding boxes from PDF using pdfplumber.
    Returns a list of pages with token-like dicts.
    """
    import io
    pages: List[OCRPage] = []
    try:
        with pdfplumber.open(io.BytesIO(file_content)) as pdf:
            for page_idx, page in enumerate(pdf.pages):
                words = page.extract_words() or []
                tokens: List[OCRToken] = []
                for w in words:
                    tokens.append({
                        "text": w.get("text", ""),
                        "bbox": [int(w.get("x0", 0)), int(w.get("top", 0)), int(w.get("x1", 0)), int(w.get("bottom", 0))],
                        "confidence": 1.0,
                    })
                pages.append({
                    "page_number": page_idx + 1,
                    "tokens": tokens,
                    "page_bbox": [0, 0, int(page.width), int(page.height)]
                })
    except Exception as e:
        logger.warning(f"PDF native extraction failed: {e}")
    return pages


def ocr_tokens_from_image(image: Image.Image | np.ndarray) -> List[OCRToken]:
    """Generate word-level OCR tokens using pytesseract (CPU-friendly).
    Accepts PIL Image or ndarray; returns list of dict tokens with bboxes and confidence.
    """
    if not isinstance(image, Image.Image):
        # assume ndarray
        image = Image.fromarray(image)
    gray = image.convert("L")
    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
    tokens: List[OCRToken] = []
    n = len(data.get("text", []))
    for i in range(n):
        text = (data["text"][i] or "").strip()
        if not text:
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        conf_str = str(data.get("conf", ["-1"]) [i])
        try:
            conf = float(conf_str) / 100.0 if conf_str not in ("-1", "", None) else 0.0
        except Exception:
            conf = 0.0
        tokens.append({
            "text": text,
            "bbox": [int(x), int(y), int(x + w), int(y + h)],
            "confidence": conf,
        })
    return tokens

