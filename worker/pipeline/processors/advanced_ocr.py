from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from dataclasses import asdict

import numpy as np
from PIL import Image

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from ...utils.logging import get_logger

logger = get_logger(__name__)


class AdvancedOCRProcessor:

    def __init__(
        self,
        use_paddle: bool = True,
        use_tesseract: bool = True,
        model_path: Optional[Union[str, Path]] = None,
        lang: str = "en",
        tesseract_lang: str = "eng",
        tesseract_config: str = "--oem 3 --psm 6",
        preprocess: bool = True,
        preprocess_method: str = "auto",
    ):
        from models.ocr_models import OCRModelEnsemble, PaddleOCRAdapter, TesseractOCR

        self.preprocess = preprocess
        self.preprocess_method = preprocess_method
        self.lang = lang
        self.tesseract_lang = tesseract_lang
        self.tesseract_config = tesseract_config

        engines: List[Any] = []
        if use_paddle:
            try:
                engines.append(PaddleOCRAdapter(lang=lang))
            except Exception as e:
                logger.warning(f"PaddleOCR unavailable: {e}")
        if use_tesseract:
            try:
                engines.append(TesseractOCR(lang=tesseract_lang, tesseract_config=tesseract_config))
            except Exception as e:
                logger.warning(f"Tesseract unavailable: {e}")
        if not engines:
            raise RuntimeError("No OCR engines initialized. Install paddleocr and/or pytesseract.")
        self.ensemble = OCRModelEnsemble(engines)

    def _ensure_image(self, image_path: Union[str, Path, np.ndarray, Image.Image]) -> Image.Image:
        if isinstance(image_path, Image.Image):
            return image_path
        if isinstance(image_path, np.ndarray):
            return Image.fromarray(image_path)
        # path-like
        return Image.open(str(image_path)).convert("RGB")

    def _preprocess_image(self, img: Image.Image, method: str = "auto") -> Image.Image:
        from PIL import ImageOps, ImageFilter
        if method in ("auto", "contrast"):
            img = ImageOps.autocontrast(img)
        if method in ("auto", "denoise"):
            img = img.filter(ImageFilter.MedianFilter(size=3))
        if method in ("auto", "sharpen"):
            img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
        return img

    def process_image(self, image_path: Union[str, Path, np.ndarray, Image.Image]) -> Dict[str, Any]:
        try:
            img = self._ensure_image(image_path)
        except Exception as e:
            return {"error": f"Failed to load image: {e}"}

        # Optional pre-processing to improve OCR quality
        if self.preprocess:
            try:
                img = self._preprocess_image(img, method=self.preprocess_method)
            except Exception as e:
                logger.warning(f"Preprocess skipped: {e}")

        res = self.ensemble.extract_text(img)
        # Convert to expected dict format used by PDFProcessor and DocumentProcessor
        out_results: List[Dict[str, Any]] = []
        tokens_for_page: List[Dict[str, Any]] = []
        for tok in res.tokens:
            box = [[tok.bbox[0], tok.bbox[1]], [tok.bbox[2], tok.bbox[1]], [tok.bbox[2], tok.bbox[3]], [tok.bbox[0], tok.bbox[3]]]
            out_results.append({"box": box, "text": tok.text, "confidence": tok.confidence})
            tokens_for_page.append({
                "text": tok.text,
                "bbox": [tok.bbox[0], tok.bbox[1], tok.bbox[2], tok.bbox[3]],
                "confidence": tok.confidence,
            })
        return {
            "image_path": str(image_path) if isinstance(image_path, (str, Path)) else None,
            "image_size": {"height": img.height, "width": img.width},
            "results": out_results,
            "tokens": tokens_for_page,
            "model_name": res.model_name,
        }

    def process(self, document: Dict[str, Any]) -> Dict[str, Any]:
        from ...types import Document as _Doc, Token as _Token, Bbox as _Bbox
        if isinstance(document, _Doc):
            for page in document.pages:
                img = page.image
                if img is not None:
                    try:
                        res = self.ensemble.extract_text(img)
                        for tok in res.tokens:
                            page.tokens.append(_Token(text=tok.text, bbox=_Bbox(x1=tok.bbox[0], y1=tok.bbox[1], x2=tok.bbox[2], y2=tok.bbox[3]), confidence=tok.confidence))
                        logger.info(f"OCR extracted {len(res.tokens)} tokens from page {page.page_num}")
                    except Exception as e:
                        logger.error(f"OCR failed on page {page.page_num}: {e}")
        return document
