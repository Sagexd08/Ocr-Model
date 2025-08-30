from typing import List, Any, Optional
from dataclasses import dataclass

from PIL import Image

# Optional heavy deps
try:  # HuggingFace TrOCR (kept for long-form fallback)
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
except Exception:  # pragma: no cover
    TrOCRProcessor = None
    VisionEncoderDecoderModel = None

try:  # Tesseract OCR
    import pytesseract
except Exception:  # pragma: no cover
    pytesseract = None

try:  # PaddleOCR wrapper
    from paddleocr import PaddleOCR as _PaddleOCR
except Exception:  # pragma: no cover
    _PaddleOCR = None


@dataclass
class OCRTokenResult:
    text: str
    bbox: List[int]  # [x1, y1, x2, y2]
    confidence: float
    token_id: Optional[str] = None


@dataclass
class OCRExtractResult:
    tokens: List[OCRTokenResult]
    page_bbox: Optional[List[int]] = None
    model_name: str = "unknown"


class TesseractOCR:
    """Lightweight OCR using pytesseract image_to_data for word-level tokens."""

    def __init__(self, lang: str = "eng", tesseract_config: str = "--oem 1 --psm 6"):
        self.lang = lang
        self.config = tesseract_config
        if pytesseract is None:
            raise RuntimeError("pytesseract is not available. Please install it to use TesseractOCR.")

    def extract_text(self, image: Any) -> OCRExtractResult:
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            img = image
        else:
            img = Image.fromarray(image).convert("RGB")

        data = pytesseract.image_to_data(img, lang=self.lang, config=self.config, output_type=pytesseract.Output.DICT)
        tokens: List[OCRTokenResult] = []
        n = len(data.get("text", []))
        for i in range(n):
            text = (data["text"][i] or "").strip()
            if not text:
                continue
            try:
                conf = float(data.get("conf", [0])[i]) / 100.0
            except Exception:
                conf = 0.0
            x, y, w, h = data.get("left", [0])[i], data.get("top", [0])[i], data.get("width", [0])[i], data.get("height", [0])[i]
            bbox = [int(x), int(y), int(x) + int(w), int(y) + int(h)]
            tokens.append(OCRTokenResult(text=text, bbox=bbox, confidence=conf))
        return OCRExtractResult(tokens=tokens, page_bbox=[0, 0, img.width, img.height], model_name="tesseract")


class PaddleOCRAdapter:
    """Adapter to normalize PaddleOCR outputs to OCRExtractResult interface."""

    def __init__(self, lang: str = "en", use_angle_cls: bool = True):
        if _PaddleOCR is None:
            raise RuntimeError("PaddleOCR is not available. Please install paddleocr to use this adapter.")
        self._ocr = _PaddleOCR(use_angle_cls=use_angle_cls, lang=lang)

    def extract_text(self, image: Any) -> OCRExtractResult:
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            img = image
        else:
            img = Image.fromarray(image).convert("RGB")
        # PaddleOCR returns list of [ [[x,y],...], (text, conf) ] per line
        result = self._ocr.ocr(img, cls=True)
        tokens: List[OCRTokenResult] = []
        # Flatten across batches (Paddle returns list per image)
        for item in (result[0] if result and isinstance(result, list) else []):
            try:
                box = item[0]
                txt = item[1][0]
                conf = float(item[1][1])
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                bbox = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
                tokens.append(OCRTokenResult(text=txt, bbox=bbox, confidence=conf))
            except Exception:
                continue
        return OCRExtractResult(tokens=tokens, page_bbox=[0, 0, img.width, img.height], model_name="paddleocr")


class OCRModelEnsemble:
    """Simple ensemble that merges tokens from multiple engines, preferring higher confidence."""

    def __init__(self, engines: List[Any]):
        # engines must implement extract_text(image) -> OCRExtractResult
        self.engines = [e for e in engines if e is not None]

    def extract_text(self, image: Any) -> OCRExtractResult:
        if not self.engines:
            raise RuntimeError("No OCR engines configured for ensemble.")
        all_tokens: List[OCRTokenResult] = []
        page_bbox = None
        for engine in self.engines:
            try:
                res: OCRExtractResult = engine.extract_text(image)
                page_bbox = page_bbox or res.page_bbox
                all_tokens.extend(res.tokens)
            except Exception:
                continue
        # De-duplicate overlapping tokens by (text,bbox) with max confidence
        dedup = {}
        for t in all_tokens:
            key = (t.text, tuple(t.bbox))
            if key not in dedup or t.confidence > dedup[key].confidence:
                dedup[key] = t
        merged = list(dedup.values())
        # Sort roughly top-to-bottom, left-to-right
        merged.sort(key=lambda t: (t.bbox[1], t.bbox[0]))
        return OCRExtractResult(tokens=merged, page_bbox=page_bbox, model_name="ocr_ensemble")


# Retain TrOCR wrapper for optional long-form recognition compatibility
class OCRModel:
    def __init__(self, model_name: str = "microsoft/trocr-base-handwritten"):
        if TrOCRProcessor is None or VisionEncoderDecoderModel is None:
            raise RuntimeError("transformers is not available for TrOCR.")
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)

    def run_ocr(self, image: Any):
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            img = Image.fromarray(image).convert("RGB")
        else:
            img = image
        pixel_values = self.processor(images=img, return_tensors="pt").pixel_values
        generated_ids = self.model.generate(pixel_values)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text, []
