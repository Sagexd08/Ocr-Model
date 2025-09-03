
import os
import io
import cv2
import fitz
import numpy as np
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from worker.pipeline.processors.advanced_ocr import AdvancedOCRProcessor
from ...types import Document, Page, Token, Bbox
from ...utils.logging import get_logger

logger = get_logger(__name__)


class PDFProcessor:

    def __init__(self,
                 ocr_processor: Optional[AdvancedOCRProcessor] = None,
                 dpi: int = 300,
                 ocr_threshold: float = 0.1):
        self.ocr_processor = ocr_processor
        self.dpi = dpi
        self.ocr_threshold = ocr_threshold

    def _is_page_scanned(self, page) -> bool:
        """
        Determine if a PDF page is scanned or digital.

        Heuristic: if the page has any extractable spans from PyMuPDF,
        we treat it as digital; otherwise we fall back to OCR.
        """
        try:
            text_dict = page.get_text("dict")
            blocks = text_dict.get("blocks", []) if isinstance(text_dict, dict) else []
            for block in blocks:
                if block.get("lines"):
                    for line in block.get("lines", []):
                        if line.get("spans"):
                            # Found at least one span -> digital
                            return False
            # No spans discovered -> likely scanned
            return True
        except Exception:
            # Safe fallback: if we cannot parse, try OCR
            return True

    def _extract_native_text(self, page) -> Dict[str, Any]:
        blocks = page.get_text("dict")["blocks"]

        result = []

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
        if not self.ocr_processor:
            return {"error": "OCR processor not provided for scanned page"}

        pix = page.get_pixmap(dpi=self.dpi)

        img_bytes = pix.tobytes("png")
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp:
                temp_path = temp.name
            cv2.imwrite(temp_path, img)

            ocr_result = self.ocr_processor.process_image(temp_path)

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

            return {"blocks": result}
        finally:
            # Clean up temp file
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

    def process(self, document: Document) -> Document:
        """Pipeline-compatible entry: read source path from doc.metadata and populate pages/tokens."""
        src = document.metadata.get("source_path")
        if not src:
            logger.warning("PDFProcessor.process called without source_path in document metadata")
            return document
        if document.metadata.get("doc_type") != "pdf":
            return document

        try:
            result = self.process_pdf(src)
            for p in result.get("pages", []):
                page = Page(
                    page_num=p.get("page_num", len(document.pages) + 1),
                    width=int(p.get("size", {}).get("width", 0) or 0),
                    height=int(p.get("size", {}).get("height", 0) or 0),
                )

                # For scanned pages, create page image for OCR pipeline
                if p.get("is_scanned", False):
                    try:
                        doc = fitz.open(src)
                        pdf_page = doc[p.get("page_num", 1) - 1]
                        pix = pdf_page.get_pixmap(dpi=self.dpi)
                        img_data = pix.tobytes("png")
                        from PIL import Image
                        import io
                        page.image = Image.open(io.BytesIO(img_data)).convert("RGB")
                        doc.close()
                        logger.info(f"Created page image for scanned page {page.page_num}")
                    except Exception as e:
                        logger.warning(f"Failed to create page image: {e}")

                # Convert blocks to tokens for text-based pages
                for blk in p.get("blocks", []):
                    bbox = blk.get("bbox")
                    if not bbox:
                        continue
                    page.tokens.append(
                        Token(text=blk.get("text", ""), bbox=Bbox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3]), confidence=float(blk.get("confidence", 1.0)))
                    )
                document.pages.append(page)

        except Exception as e:
            logger.error(f"PDFProcessor.process failed: {e}")
        return document

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
                            logger.info("Initialized AdvancedOCRProcessor for scanned page")
                        except Exception as e:
                            logger.error(f"Failed to initialize OCR processor: {e}")
                            self.ocr_processor = None
                    if self.ocr_processor is not None:
                        try:
                            ocr_result = self._process_scanned_page(page)
                            page_result.update(ocr_result)
                            logger.info(f"OCR processed page {page_num + 1}, found {len(ocr_result.get('blocks', []))} blocks")
                        except Exception as e:
                            logger.error(f"OCR processing failed for page {page_num + 1}: {e}")
                            # Fallback to native extraction
                            page_result.update(self._extract_native_text(page))
                    else:
                        # Fallback to native extraction if OCR cannot be initialized
                        page_result.update(self._extract_native_text(page))
                        logger.warning(f"No OCR processor available for scanned page {page_num + 1}")
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
