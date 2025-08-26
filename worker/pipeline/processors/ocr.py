from typing import Dict, Any, List, Optional, Union
import os
import uuid
import cv2
import numpy as np
import time
from pathlib import Path

from ...types import Document, Page, Bbox, Token, Table, Cell
from ...utils.logging import get_logger, log_execution_time
from ...model_manager import ModelManager

logger = get_logger(__name__)

class OCRProcessor:
    """
    Performs Optical Character Recognition (OCR) on document pages.
    Uses multiple OCR engines based on document type and quality.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.default_lang = self.config.get("default_language", "eng")
        self.dpi = self.config.get("dpi", 300)
        self.min_confidence = self.config.get("min_confidence", 0.5)
        self.tesseract_config = self.config.get("tesseract_config", "--oem 1 --psm 6")
        
    @log_execution_time
    def process(self, document: Document, model_manager: ModelManager) -> Document:
        """
        Process a document to extract text using OCR.
        
        Args:
            document: Document object with pages and images
            model_manager: Model manager instance for OCR models
            
        Returns:
            Document with extracted text tokens
        """
        logger.info(f"Processing OCR for document with {len(document.pages)} pages")
        
        for i, page in enumerate(document.pages):
            logger.debug(f"Processing OCR for page {page.page_num}")
            
            if page.image is None:
                logger.warning(f"No image found for page {page.page_num}, skipping OCR")
                continue
            
            try:
                # Get page image
                image = page.image
                if isinstance(image, str):
                    # Load image if it's a file path
                    image = cv2.imread(image)
                
                if image is None:
                    logger.warning(f"Failed to load image for page {page.page_num}")
                    continue
                
                # Extract text using OCR
                ocr_result = model_manager.run_ocr(image)
                
                # Convert OCR tokens to our Token format
                tokens = self._convert_ocr_tokens(ocr_result["tokens"], page.page_num)
                
                # Update the page with extracted tokens
                page.tokens = tokens
                document.pages[i] = page
                
                logger.info(f"Extracted {len(tokens)} tokens from page {page.page_num}")
            except Exception as e:
                logger.error(f"Error performing OCR on page {page.page_num}: {str(e)}")
        
        return document
    
    def _convert_ocr_tokens(self, ocr_tokens: List[Dict[str, Any]], page_num: int) -> List[Token]:
        """Convert OCR tokens to our Token format"""
        tokens = []
        
        for i, ocr_token in enumerate(ocr_tokens):
            # Skip tokens with confidence below threshold
            if ocr_token.get("confidence", 0) < self.min_confidence:
                continue
                
            # Create a Token object
            token = Token(
                id=f"tok_{uuid.uuid4().hex[:8]}",
                text=ocr_token["text"],
                bbox=Bbox(
                    x1=ocr_token["bbox"][0],
                    y1=ocr_token["bbox"][1],
                    x2=ocr_token["bbox"][2],
                    y2=ocr_token["bbox"][3]
                ),
                confidence=ocr_token["confidence"],
                page_num=page_num
            )
            tokens.append(token)
        
        return tokens
