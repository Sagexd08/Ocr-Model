"""
PDF Processor module for extracting text and handling PDF documents.
Provides specialized processing for both native PDFs and scanned PDF documents.
"""

import os
import io
import cv2
import fitz  # PyMuPDF
import numpy as np
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from worker.pipeline.processors.advanced_ocr import AdvancedOCRProcessor
from ...types import Document, Page, Token, Bbox
from ...utils.logging import get_logger

logger = get_logger(__name__)


class PDFProcessor:
    """
    Specialized processor for PDF documents that can:
    - Extract native text from digital PDFs
    - Perform OCR on scanned PDFs
    - Handle hybrid PDFs with both digital and scanned content
    - Extract and maintain layout information
    """

    def __init__(self,
                 ocr_processor: Optional[AdvancedOCRProcessor] = None,
                 dpi: int = 300,
                 ocr_threshold: float = 0.1):
        """
        Initialize the PDF processor.

        Args:
            ocr_processor: Advanced OCR processor for scanned content
            dpi: DPI for rendering PDF pages
            ocr_threshold: Threshold for determining if a page needs OCR
                           (percentage of text coverage)
        """
        self.ocr_processor = ocr_processor
        self.dpi = dpi
        self.ocr_threshold = ocr_threshold

    def _is_page_scanned(self, page) -> bool:
        """
        Determine if a PDF page is scanned or digital.

        Args:
            page: PyMuPDF page object

        Returns:
            True if the page appears to be scanned, False if digital
        """
        # Extract text from the page
        text = page.get_text()

        # Calculate text coverage ratio
        text_length = len(text.strip())
        page_area = page.rect.width * page.rect.height

        # Calculate text density (characters per square point)
        text_density = text_length / page_area if page_area > 0 else 0

        # Check if the page has very little text relative to its size
        return text_density < self.ocr_threshold

    def _extract_native_text(self, page) -> Dict[str, Any]:
        """
        Extract native text and layout information from a digital PDF page.

        Args:
            page: PyMuPDF page object

        Returns:
            Dictionary with extracted text blocks and their positions
        """
        # Extract text with layout information
        blocks = page.get_text("dict")["blocks"]

        result = []

        # Process text blocks: create a token per span for finer granularity
        for block in blocks:
            if "lines" not in block:
                continue

            for line in block["lines"]:
                spans = line.get("spans", [])
                for span in spans:
                    text = span.get("text", "")
                    if not text.strip():
                        continue
                    span_bbox = list(span.get("bbox", [0, 0, 0, 0]))
                    result.append({
                        "text": text,
                        "bbox": span_bbox,
                        "type": "native",
                        "font": span.get("font", ""),
                        "font_size": span.get("size", 0),
                        "is_bold": span.get("flags", 0) & 2 > 0,
                        "is_italic": span.get("flags", 0) & 4 > 0,
                        "confidence": 1.0,
                    })

        return {"blocks": result}

    def _process_scanned_page(self, page) -> Dict[str, Any]:
        """
        Process a scanned PDF page using OCR.

        Args:
            page: PyMuPDF page object

        Returns:
            Dictionary with OCR results
        """
        if not self.ocr_processor:
            return {"error": "OCR processor not provided for scanned page"}

        # Convert the page to an image
        pix = page.get_pixmap(dpi=self.dpi)

        # Convert pixmap to OpenCV format
        img_bytes = pix.tobytes("png")
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Save to temporary file for OCR processing
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp:
            temp_path = temp.name
            cv2.imwrite(temp_path, img)

        try:
            # Process with OCR
            ocr_result = self.ocr_processor.process_image(temp_path)

            # Convert OCR results to match native text format
            result = []
            for item in ocr_result.get("results", []):
                box = item.get("box", [[0, 0], [0, 0], [0, 0], [0, 0]])

                # Convert OCR box to PDF bbox format [x0, y0, x2, y2]
                min_x = min(p[0] for p in box)
                min_y = min(p[1] for p in box)
                max_x = max(p[0] for p in box)
                max_y = max(p[1] for p in box)

                result.append({
                    "text": item.get("text", ""),
                    "bbox": [min_x, min_y, max_x, max_y],
                    "type": "ocr",
                    "confidence": item.get("confidence", 0.0)
                })


        def process(self, document: Document) -> Document:
            """Pipeline-compatible entry: read source path from doc.metadata and populate pages/tokens.
            This does not perform table extraction; only text/native+OCR and tokenization per page.
            """
            src = document.metadata.get("source_path")
            if not src:
                logger.warning("PDFProcessor.process called without source_path in document metadata")
                return document
            if document.metadata.get("doc_type") != "pdf":
                # Not a PDF document; no-op
                return document
            try:
                result = self.process_pdf(src)
                for p in result.get("pages", []):
                    page = Page(
                        page_num=p.get("page_num", len(document.pages) + 1),
                        width=int(p.get("size", {}).get("width", 0) or 0),
                        height=int(p.get("size", {}).get("height", 0) or 0),
                    )
                    # Convert blocks to tokens (roughly per-line bbox as a token)
                    for blk in p.get("blocks", []):
                        bbox = blk.get("bbox")
                        if not bbox:
                            continue
                        page.tokens.append(
                            Token(text=blk.get("text", ""), bbox=Bbox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3]), confidence=float(blk.get("confidence", 1.0)))
                        )
                    document.pages.append(page)
            except Exception as e:
                logger.warning(f"PDFProcessor.process failed: {e}")
            return document

            return {"blocks": result}

        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def process_pdf(self, pdf_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Process a PDF document, extracting text using the appropriate method
        for each page (native extraction or OCR).

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Dictionary with processed text from all pages
        """
        pdf_path = str(pdf_path)

        try:
            doc = fitz.open(pdf_path)

            result = {
                "filename": os.path.basename(pdf_path),
                "page_count": len(doc),
                "pages": []
            }

            # Optional page cap for speed (set by EnhancedDocumentProcessor)
            max_pages = getattr(self, "max_pages", None)
            for page_num, page in enumerate(doc):
                if max_pages is not None and page_num >= int(max_pages):
                    break
                page_result = {
                    "page_num": page_num + 1,
                    "size": {"width": page.rect.width, "height": page.rect.height}
                }

                # Check if page is scanned
                is_scanned = self._is_page_scanned(page)
                page_result["is_scanned"] = is_scanned

                # Process accordingly
                if is_scanned:
                    # Ensure OCR processor exists for scanned pages
                    if self.ocr_processor is None:
                        try:
                            self.ocr_processor = AdvancedOCRProcessor()
                        except Exception:
                            self.ocr_processor = None
                    if self.ocr_processor is not None:
                        page_result.update(self._process_scanned_page(page))
                    else:
                        # Fallback to native extraction if OCR cannot be initialized
                        page_result.update(self._extract_native_text(page))
                else:
                    page_result.update(self._extract_native_text(page))

                result["pages"].append(page_result)

            return result

        except Exception as e:
            return {
                "error": f"Failed to process PDF: {str(e)}",
                "filename": os.path.basename(pdf_path)
            }

    def extract_pdf_metadata(self, pdf_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Extract metadata from a PDF document.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Dictionary with PDF metadata
        """
        pdf_path = str(pdf_path)

        try:
            doc = fitz.open(pdf_path)

            metadata = doc.metadata
            if not metadata:
                metadata = {}

            # Add additional information
            metadata.update({
                "filename": os.path.basename(pdf_path),
                "page_count": len(doc),
                "file_size": os.path.getsize(pdf_path),
                "has_toc": bool(doc.get_toc()),
                "has_links": any(len(page.get_links()) > 0 for page in doc),
                "has_images": any(len(page.get_images()) > 0 for page in doc),
                "form_fields": bool(doc.is_form_pdf)
            })

            return metadata

        except Exception as e:
            return {
                "error": f"Failed to extract metadata: {str(e)}",
                "filename": os.path.basename(pdf_path)
            }

    def extract_images_from_pdf(self, pdf_path: Union[str, Path], output_dir: Union[str, Path]) -> Dict[str, Any]:
        """
        Extract images from a PDF document.

        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save extracted images

        Returns:
            Dictionary with information about extracted images
        """
        pdf_path = str(pdf_path)
        output_dir = Path(output_dir)

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        try:
            doc = fitz.open(pdf_path)
            base_filename = os.path.splitext(os.path.basename(pdf_path))[0]

            result = {
                "filename": os.path.basename(pdf_path),
                "images_extracted": 0,
                "images": []
            }

            image_count = 0

            for page_num, page in enumerate(doc):
                image_list = page.get_images(full=True)

                for img_idx, img_info in enumerate(image_list):
                    xref = img_info[0]

                    # Extract image
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]

                    # Generate image filename
                    img_filename = f"{base_filename}_p{page_num+1}_img{img_idx+1}.{image_ext}"
                    img_path = output_dir / img_filename

                    # Save the image
                    with open(img_path, "wb") as img_file:
                        img_file.write(image_bytes)

                    image_count += 1

                    # Add to results
                    result["images"].append({
                        "filename": img_filename,
                        "page": page_num + 1,
                        "index": img_idx + 1,
                        "path": str(img_path),
                        "size": base_image.get("size", (0, 0)),
                        "extension": image_ext
                    })

            result["images_extracted"] = image_count
            return result

        except Exception as e:
            return {
                "error": f"Failed to extract images: {str(e)}",
                "filename": os.path.basename(pdf_path)
            }
