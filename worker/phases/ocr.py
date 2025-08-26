from ..types import Document, Token, Bbox
from ...models.renderer_classifier import RendererClassifier
from ...models.ocr_models import OCRModel
from ..pipeline.text_native import ocr_tokens_from_image


def ocr(document: Document) -> Document:
    classifier = RendererClassifier()
    render_type = classifier.classify(document)
    document.render_type = render_type

    if render_type == "digital_pdf":
        # For digitally-rendered PDFs, prefer native text extraction upstream.
        return document
    else:
        # CPU-friendly tokenization using Tesseract for bounding boxes
        for page in document.pages:
            tokens = ocr_tokens_from_image(page.image)
            for t in tokens:
                page.tokens.append(
                    Token(
                        text=t["text"],
                        bbox=Bbox(x1=t["bbox"][0], y1=t["bbox"][1], x2=t["bbox"][2], y2=t["bbox"][3]),
                        confidence=t["confidence"],
                    )
                )
        # Optionally, keep TrOCR for long-form text recognition (off by default on CPU)
        # ocr_model = OCRModel()
        # for page in document.pages:
        #     text, _ = ocr_model.run_ocr(page.image)
        #     if text:
        #         page.tokens.append(Token(text=text, bbox=Bbox(x1=0, y1=0, x2=1, y2=1), confidence=0.5))

    return document
