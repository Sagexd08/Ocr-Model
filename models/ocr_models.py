"""
OCR Models for CurioScan

This module implements various OCR models and an ensemble approach for robust text extraction.
Supports Tesseract, EasyOCR, and optional transformer-based models like TrOCR and LayoutLM.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from PIL import Image
import cv2
import pytesseract
import easyocr
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OCRToken:
    """Represents a single OCR token with position and confidence."""
    text: str
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    token_id: int


@dataclass
class OCRResult:
    """Complete OCR result for a document page."""
    tokens: List[OCRToken]
    page_bbox: Tuple[int, int, int, int]
    page_number: int
    processing_time: float
    model_name: str


class BaseOCR(ABC):
    """Abstract base class for OCR models."""
    
    @abstractmethod
    def extract_text(self, image: Image.Image, **kwargs) -> OCRResult:
        """Extract text from an image."""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get the model name."""
        pass


class TesseractOCR(BaseOCR):
    """Tesseract OCR implementation with advanced configuration."""
    
    def __init__(
        self,
        config: str = "--oem 3 --psm 6",
        languages: List[str] = None,
        confidence_threshold: float = 0.0
    ):
        self.config = config
        self.languages = languages or ["eng"]
        self.confidence_threshold = confidence_threshold
        self.lang_string = "+".join(self.languages)
        
        # Verify Tesseract installation
        try:
            pytesseract.get_tesseract_version()
            logger.info(f"Tesseract version: {pytesseract.get_tesseract_version()}")
        except Exception as e:
            logger.error(f"Tesseract not found: {e}")
            raise
    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for better OCR results."""
        # Convert to OpenCV format
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_array
        
        # Apply preprocessing
        # 1. Noise reduction
        denoised = cv2.medianBlur(gray, 3)
        
        # 2. Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # 3. Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        return binary
    
    def extract_text(self, image: Image.Image, **kwargs) -> OCRResult:
        """Extract text using Tesseract."""
        import time
        start_time = time.time()
        
        # Preprocess image
        processed_img = self.preprocess_image(image)
        
        # Get detailed OCR data
        data = pytesseract.image_to_data(
            processed_img,
            config=self.config,
            lang=self.lang_string,
            output_type=pytesseract.Output.DICT
        )
        
        # Parse results
        tokens = []
        token_id = 0
        
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            confidence = float(data['conf'][i])
            
            if text and confidence >= self.confidence_threshold:
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                bbox = (x, y, x + w, y + h)
                
                token = OCRToken(
                    text=text,
                    bbox=bbox,
                    confidence=confidence / 100.0,  # Normalize to 0-1
                    token_id=token_id
                )
                tokens.append(token)
                token_id += 1
        
        processing_time = time.time() - start_time
        
        return OCRResult(
            tokens=tokens,
            page_bbox=(0, 0, image.width, image.height),
            page_number=kwargs.get('page_number', 0),
            processing_time=processing_time,
            model_name=self.get_model_name()
        )
    
    def get_model_name(self) -> str:
        return "tesseract"


class EasyOCR(BaseOCR):
    """EasyOCR implementation for multi-language support."""
    
    def __init__(
        self,
        languages: List[str] = None,
        gpu: bool = True,
        confidence_threshold: float = 0.0
    ):
        self.languages = languages or ["en"]
        self.gpu = gpu
        self.confidence_threshold = confidence_threshold
        
        try:
            self.reader = easyocr.Reader(self.languages, gpu=self.gpu)
            logger.info(f"EasyOCR initialized with languages: {self.languages}")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            raise
    
    def extract_text(self, image: Image.Image, **kwargs) -> OCRResult:
        """Extract text using EasyOCR."""
        import time
        start_time = time.time()
        
        # Convert PIL to numpy array
        img_array = np.array(image)
        
        # Run EasyOCR
        results = self.reader.readtext(img_array)
        
        # Parse results
        tokens = []
        token_id = 0
        
        for result in results:
            bbox_points, text, confidence = result
            
            if confidence >= self.confidence_threshold:
                # Convert bbox points to rectangle
                x_coords = [point[0] for point in bbox_points]
                y_coords = [point[1] for point in bbox_points]
                x1, y1 = int(min(x_coords)), int(min(y_coords))
                x2, y2 = int(max(x_coords)), int(max(y_coords))
                bbox = (x1, y1, x2, y2)
                
                token = OCRToken(
                    text=text,
                    bbox=bbox,
                    confidence=confidence,
                    token_id=token_id
                )
                tokens.append(token)
                token_id += 1
        
        processing_time = time.time() - start_time
        
        return OCRResult(
            tokens=tokens,
            page_bbox=(0, 0, image.width, image.height),
            page_number=kwargs.get('page_number', 0),
            processing_time=processing_time,
            model_name=self.get_model_name()
        )
    
    def get_model_name(self) -> str:
        return "easyocr"


class TrOCROCR(BaseOCR):
    """TrOCR (Transformer-based OCR) implementation."""
    
    def __init__(
        self,
        model_name: str = "microsoft/trocr-base-printed",
        device: str = "auto"
    ):
        self.model_name = model_name
        
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            
            self.processor = TrOCRProcessor.from_pretrained(model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"TrOCR model loaded: {model_name}")
        except ImportError:
            logger.error("TrOCR requires transformers library")
            raise
        except Exception as e:
            logger.error(f"Failed to load TrOCR model: {e}")
            raise
    
    def extract_text(self, image: Image.Image, **kwargs) -> OCRResult:
        """Extract text using TrOCR."""
        import time
        start_time = time.time()
        
        # Process image
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
        
        # Generate text
        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Create single token (TrOCR doesn't provide bbox info)
        token = OCRToken(
            text=generated_text,
            bbox=(0, 0, image.width, image.height),
            confidence=0.9,  # Default confidence
            token_id=0
        )
        
        processing_time = time.time() - start_time
        
        return OCRResult(
            tokens=[token],
            page_bbox=(0, 0, image.width, image.height),
            page_number=kwargs.get('page_number', 0),
            processing_time=processing_time,
            model_name=self.get_model_name()
        )
    
    def get_model_name(self) -> str:
        return f"trocr_{self.model_name.split('/')[-1]}"


class OCRModelEnsemble:
    """Ensemble of multiple OCR models for robust text extraction."""
    
    def __init__(self, models: List[BaseOCR], voting_strategy: str = "confidence_weighted"):
        self.models = models
        self.voting_strategy = voting_strategy
        
        logger.info(f"OCR Ensemble initialized with {len(models)} models")
        for model in models:
            logger.info(f"  - {model.get_model_name()}")
    
    def extract_text(self, image: Image.Image, **kwargs) -> OCRResult:
        """Extract text using ensemble of models."""
        import time
        start_time = time.time()
        
        # Run all models
        results = []
        for model in self.models:
            try:
                result = model.extract_text(image, **kwargs)
                results.append(result)
            except Exception as e:
                logger.warning(f"Model {model.get_model_name()} failed: {e}")
        
        if not results:
            raise RuntimeError("All OCR models failed")
        
        # Combine results based on voting strategy
        if self.voting_strategy == "confidence_weighted":
            combined_result = self._confidence_weighted_voting(results, image, **kwargs)
        elif self.voting_strategy == "majority_vote":
            combined_result = self._majority_voting(results, image, **kwargs)
        else:
            # Default: use best single result
            combined_result = max(results, key=lambda r: len(r.tokens))
        
        combined_result.processing_time = time.time() - start_time
        combined_result.model_name = "ensemble"
        
        return combined_result
    
    def _confidence_weighted_voting(self, results: List[OCRResult], image: Image.Image, **kwargs) -> OCRResult:
        """Combine results using confidence-weighted voting."""
        all_tokens = []
        token_id = 0
        
        # Collect all tokens from all models
        for result in results:
            for token in result.tokens:
                # Create new token with updated ID
                new_token = OCRToken(
                    text=token.text,
                    bbox=token.bbox,
                    confidence=token.confidence,
                    token_id=token_id
                )
                all_tokens.append(new_token)
                token_id += 1
        
        # Sort by confidence and remove duplicates
        all_tokens.sort(key=lambda t: t.confidence, reverse=True)
        
        # Simple deduplication based on bbox overlap
        final_tokens = []
        for token in all_tokens:
            is_duplicate = False
            for existing_token in final_tokens:
                if self._bbox_overlap(token.bbox, existing_token.bbox) > 0.7:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                final_tokens.append(token)
        
        return OCRResult(
            tokens=final_tokens,
            page_bbox=(0, 0, image.width, image.height),
            page_number=kwargs.get('page_number', 0),
            processing_time=0.0,  # Will be set by caller
            model_name="ensemble"
        )
    
    def _majority_voting(self, results: List[OCRResult], image: Image.Image, **kwargs) -> OCRResult:
        """Combine results using majority voting."""
        # For simplicity, return the result with most tokens
        best_result = max(results, key=lambda r: len(r.tokens))
        return best_result
    
    def _bbox_overlap(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate overlap ratio between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0


def create_ocr_ensemble(config: Dict) -> OCRModelEnsemble:
    """
    Factory function to create OCR ensemble from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured OCRModelEnsemble
    """
    ocr_config = config.get("model", {}).get("ocr_models", {})
    models = []
    
    # Add Tesseract if enabled
    if ocr_config.get("tesseract", {}).get("enabled", True):
        tesseract_config = ocr_config["tesseract"]
        models.append(TesseractOCR(
            config=tesseract_config.get("config", "--oem 3 --psm 6"),
            languages=tesseract_config.get("languages", ["eng"])
        ))
    
    # Add EasyOCR if enabled
    if ocr_config.get("easyocr", {}).get("enabled", True):
        easyocr_config = ocr_config["easyocr"]
        models.append(EasyOCR(
            languages=easyocr_config.get("languages", ["en"]),
            gpu=easyocr_config.get("gpu", True)
        ))
    
    # Add TrOCR if enabled
    if ocr_config.get("trocr", {}).get("enabled", False):
        trocr_config = ocr_config["trocr"]
        models.append(TrOCROCR(
            model_name=trocr_config.get("model_name", "microsoft/trocr-base-printed")
        ))
    
    if not models:
        raise ValueError("No OCR models enabled in configuration")
    
    return OCRModelEnsemble(models)
