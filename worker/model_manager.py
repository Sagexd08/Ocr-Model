"""
Model manager for CurioScan workers.

Handles loading, caching, and managing ML models for document processing.
"""

import os
import threading
import time
from typing import Dict, Any, Optional
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import cv2
try:
    import pytesseract
except ImportError:
    pytesseract = None

# Optional PaddleOCR integration
try:
    from paddleocr import PaddleOCR as _PaddleOCR
except Exception:  # broad: handle missing pkg or runtime issues
    _PaddleOCR = None

from .utils.logging import get_logger

logger = get_logger(__name__)

# Set default feature flags
USE_GPU = True
USE_ADVANCED_LAYOUT = True
USE_TABLE_DETECTION = True
LOAD_EASYOCR = False
LOAD_TABLE_DETECTOR = False
LOAD_LAYOUT_ANALYZER = False

# Try to import feature flags if available
try:
    from common.feature_flags import (
        USE_GPU, 
        USE_ADVANCED_LAYOUT, 
        USE_TABLE_DETECTION,
        LOAD_EASYOCR,
        LOAD_TABLE_DETECTOR,
        LOAD_LAYOUT_ANALYZER
    )
except ImportError:
    logger.info("Common feature flags not found, using defaults")


class ModelManager:
    """Manages ML models for document processing."""
    
    # Default paths for models
    DEFAULT_MODEL_PATHS = {
        # OCR models
        "ocr_detection": "models/ocr/detection.onnx",
        "ocr_recognition": "models/ocr/recognition.onnx",
        "ocr": "models/ocr",  # Base OCR model directory
        
        # Table detection models
        "table_detection": "models/table_detector/model.onnx",
        "table_structure": "models/table_detector/structure.onnx",
        
        # Document classification models
        "document_classification": "models/document_classifier/model.onnx",
        "feature_extraction": "models/document_classifier/extractor.onnx",
        
        # Layout analysis models
        "layout_analysis": "models/layout/model.onnx",
        
        # Form field detection models
        "form_detection": "models/form_detector/model.onnx"
    }
    
    def __init__(self, models_path: str = "models"):
        self.models_path = Path(models_path)
        self.models: Dict[str, Any] = {}
        self.device = self._get_device()
        self._lock = threading.Lock()
        
        logger.info(f"ModelManager initialized with device: {self.device}")
    
    def _get_device(self) -> str:
        """Determine the best device for model inference."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def get_model_path(self, model_id: str) -> str:
        """
        Get the path to a specific model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Absolute path to the model file or directory
        """
        if model_id in self.DEFAULT_MODEL_PATHS:
            return str(self.models_path / self.DEFAULT_MODEL_PATHS[model_id])
        else:
            logger.warning(f"Unknown model ID: {model_id}, using generic path")
            return str(self.models_path / model_id)
    
    def initialize_models(self):
        """Initialize all models."""
        logger.info("Initializing models...")
        
        try:
            self._load_renderer_classifier()
            self._load_ocr_models()
            self._load_table_detector()
            self._load_layout_analyzer()
            
            logger.info("All models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {str(e)}")
            
    
    def _load_renderer_classifier(self):
        """Load the renderer classifier model."""
        try:
            from models.renderer_classifier import RendererClassifier
            
            model_path = self.models_path / "renderer_classifier.pth"
            
            model = RendererClassifier()
            
            if model_path.exists():
                state_dict = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state_dict)
                logger.info("Loaded renderer classifier from checkpoint")
            else:
                logger.warning("Renderer classifier checkpoint not found, using pretrained weights")
            
            model.to(self.device)
            model.eval()
            
            self.models["renderer_classifier"] = model
            logger.info("Renderer classifier loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load renderer classifier: {str(e)}")
    
    def _load_ocr_models(self):
        """Load OCR models."""
        try:
            from models.ocr_models import TesseractOCR, PaddleOCRAdapter, OCRModelEnsemble

            # Tesseract
            try:
                tesseract_ocr = TesseractOCR()
                self.models["tesseract_ocr"] = tesseract_ocr
                logger.info("Tesseract OCR loaded")
            except Exception as e:
                logger.warning(f"Tesseract init failed: {e}")

            # PaddleOCR (optional)
            if _PaddleOCR is not None:
                try:
                    self.models["paddleocr_adapter"] = PaddleOCRAdapter(lang="en")
                    logger.info("PaddleOCR adapter loaded")
                except Exception as e:
                    logger.warning(f"Failed to init PaddleOCR adapter: {e}")
            else:
                logger.info("PaddleOCR not available (package not installed)")

            # Build ensemble in preferred order
            ocr_models = [m for m in [self.models.get("paddleocr_adapter"), self.models.get("tesseract_ocr")] if m]
            if not ocr_models:
                raise RuntimeError("No OCR engines available")
            ensemble = OCRModelEnsemble(ocr_models)
            self.models["ocr_ensemble"] = ensemble
            logger.info("OCR ensemble created (PaddleOCR,Tesseract)")

        except Exception as e:
            logger.error(f"Failed to load OCR models: {str(e)}")

    def _load_table_detector(self):
        """Load table detection model."""
        try:
            from models.table_detector import TableDetector
            
            model_path = self.models_path / "table_detector.pth"
            
            model = TableDetector()
            
            if model_path.exists():
                
                pass
            
            if LOAD_TABLE_DETECTOR:
                self.models["table_detector"] = model
                logger.info("Table detector loaded")
            else:
                logger.info("Skipping table detector per env flag")
            
        except Exception as e:
            logger.error(f"Failed to load table detector: {str(e)}")
    
    def _load_layout_analyzer(self):
        """Load layout analysis model."""
        try:
            from models.layout_analyzer import LayoutAnalyzer
            
            model_path = self.models_path / "layout_analyzer.pth"
            
            model = LayoutAnalyzer()
            
            if model_path.exists():
                
                pass
            
            if LOAD_LAYOUT_ANALYZER:
                self.models["layout_analyzer"] = model
                logger.info("Layout analyzer loaded")
            else:
                logger.info("Skipping layout analyzer per env flag")
            
        except Exception as e:
            logger.error(f"Failed to load layout analyzer: {str(e)}")
    
    def get_model(self, model_name: str) -> Optional[Any]:
        """Get a model by name."""
        with self._lock:
            return self.models.get(model_name)
    
    def is_model_loaded(self, model_name: str) -> bool:
        """Check if a model is loaded."""
        return model_name in self.models
    
    def classify_document(self, image: Image.Image, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Classify document render type."""
        model = self.get_model("renderer_classifier")
        
        if model is None:
            
            return self._fallback_classification(metadata)
        
        try:
            render_type, confidence = model.predict(image, metadata)
            
            return {
                "render_type": render_type,
                "confidence": confidence,
                "method": "model"
            }
            
        except Exception as e:
            logger.error(f"Model classification failed: {str(e)}")
            return self._fallback_classification(metadata)
    
    def _fallback_classification(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback classification based on metadata."""
        mime_type = metadata.get("mime_type", "")
        has_text = metadata.get("has_embedded_text", False)
        
        if mime_type == "application/pdf":
            if has_text:
                render_type = "digital_pdf"
            else:
                render_type = "scanned_image"
        elif mime_type.startswith("image/"):
            render_type = "photograph"
        elif "word" in mime_type:
            render_type = "docx"
        else:
            render_type = "scanned_image"
        
        return {
            "render_type": render_type,
            "confidence": 0.5,
            "method": "fallback"
        }

    def run_ocr(self, image: Image.Image) -> Dict[str, Any]:
        """Compatibility method used by OCRProcessor to get token results."""
        return self.extract_text_ocr(image, render_type="auto")

    def extract_text_ocr(self, image: Image.Image, render_type: str) -> Dict[str, Any]:
        """Extract text using OCR models with rotation and multi-column handling."""

        # Prefer the ensemble which may include PaddleOCR if available
        model = self.get_model("ocr_ensemble") or self.get_model("tesseract_ocr")
        if model is None:
            raise RuntimeError("No OCR models available")

        # Basic orientation correction: try 0, 90, 180, 270 and pick best average confidence
        rotations = [0, 90, 180, 270]
        best = None
        for deg in rotations:
            try:
                if deg != 0:
                    img_rot = image.rotate(deg, expand=True) if isinstance(image, Image.Image) else image
                else:
                    img_rot = image
                start = time.time()
                result = model.extract_text(img_rot)
                dur = time.time() - start
                # Compute score: prefer more tokens and higher confidence
                toks = getattr(result, "tokens", [])
                if toks:
                    avg_conf = sum(getattr(t, "confidence", 0.0) for t in toks) / max(1, len(toks))
                else:
                    avg_conf = 0.0
                score = (avg_conf, len(toks))
                cand = (score, deg, result, dur)
                if best is None or cand[0] > best[0]:
                    best = cand
            except Exception as e:
                logger.debug(f"Rotation {deg} OCR failed: {e}")
                continue

        if best is None:
            raise RuntimeError("OCR failed for all rotations")

        _, best_deg, result, dur = best

        # Optional multi-column heuristic: group tokens by y bands and reorder left-to-right
        toks = getattr(result, "tokens", [])
        toks_sorted = sorted(toks, key=lambda t: (t.bbox[1], t.bbox[0])) if toks else []

        return {
            "tokens": [
                {
                    "text": t.text,
                    "bbox": t.bbox,
                    "confidence": t.confidence,
                    "token_id": getattr(t, "token_id", None),
                }
                for t in toks_sorted
            ],
            "page_bbox": getattr(result, "page_bbox", None),
            "processing_time": dur,
            "model_name": getattr(result, "model_name", "ocr_ensemble"),
            "rotation": best_deg,
        }

    def detect_tables(self, image: Image.Image) -> Dict[str, Any]:
        """Detect tables in the image."""
        model = self.get_model("table_detector")
        
        if model is None:
            logger.warning("Table detector not available, skipping table detection")
            return {"tables": [], "method": "none"}
        
        try:
            tables = model.detect_tables(image)
            return {
                "tables": tables,
                "method": "model"
            }
            
        except Exception as e:
            logger.error(f"Table detection failed: {str(e)}")
            return {"tables": [], "method": "failed"}
    
    def analyze_layout(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze document layout."""
        model = self.get_model("layout_analyzer")
        
        if model is None:
            logger.warning("Layout analyzer not available, using basic layout")
            return self._basic_layout_analysis(image)
        
        try:
            layout = model.analyze_layout(image)
            return layout
            
        except Exception as e:
            logger.error(f"Layout analysis failed: {str(e)}")
            return self._basic_layout_analysis(image)
    
    def _basic_layout_analysis(self, image: Image.Image) -> Dict[str, Any]:
        """Basic layout analysis fallback."""
        width, height = image.size
        
        return {
            "regions": [
                {
                    "type": "text",
                    "bbox": [0, 0, width, height],
                    "confidence": 0.5
                }
            ],
            "method": "basic"
        }
    
    def get_device(self) -> str:
        """Get the device being used."""
        return self.device
    
    def get_loaded_models(self) -> Dict[str, str]:
        """Get list of loaded models."""
        return {name: type(model).__name__ for name, model in self.models.items()}
