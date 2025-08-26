"""
Phase helper functions for the document processor.
Split from worker.document_processor to simplify the orchestrator.
"""
from __future__ import annotations
from typing import Dict, Any, List

from PIL import Image

from worker.pipeline.io_conversion import detect_mime_type
from worker.types import OCRPage, OCRToken


def extract_metadata(file_content: bytes, file_path: str) -> Dict[str, Any]:
    """Extract lightweight metadata with best-effort PDF enhancement."""
    from common.logging_utils import get_context_logger

    logger = get_context_logger(__name__)

    metadata: Dict[str, Any] = {
        "file_size": len(file_content),
        "mime_type": detect_mime_type(file_content, file_path),
        "has_embedded_text": False,
        "page_count": 1,
    }

    
    try:
        if metadata["mime_type"] == "application/pdf":
            import io

            try:
                import pdfplumber
            except Exception:
                logger.debug("pdfplumber not available; skipping PDF metadata enhancement")
                pdfplumber = None

            if pdfplumber is not None:
                with pdfplumber.open(io.BytesIO(file_content)) as pdf:
                    metadata["page_count"] = len(pdf.pages)
                    for page in pdf.pages[:3]:
                        try:
                            text = page.extract_text()
                        except Exception:
                            text = None
                        if text and text.strip():
                            metadata["has_embedded_text"] = True
                            break
    except Exception:
        logger.debug("PDF metadata enhancement skipped due to exception", exc_info=True)

    return metadata


def extract_native_text(file_content: bytes) -> List[OCRPage]:
    from worker.pipeline.text_native import extract_pdf_native_text

    return extract_pdf_native_text(file_content)


def extract_table_content(table: Dict[str, Any], tokens: List[OCRToken], image: Image.Image) -> Dict[str, Any]:
    from worker.pipeline.tables import extract_table_content as _extract

    content = _extract(tuple(table.get("bbox", [0, 0, 0, 0])), tokens)
    return {
        "table_id": table.get("table_id", "unknown"),
        "bbox": table.get("bbox", [0, 0, 0, 0]),
        **content,
        "confidence": table.get("confidence", 0.0),
    }


def extract_text_regions(tokens: List[OCRToken], tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    from worker.pipeline.text_regions import group_text_regions

    return group_text_regions(tokens)

