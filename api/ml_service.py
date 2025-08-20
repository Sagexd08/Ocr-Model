"""
ML service initialization for CurioScan API.

This module handles loading and initializing ML models for the API.
"""

import logging
import os
from typing import Optional, Dict, Any

import torch
from .config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Global model instances
_models: Dict[str, Any] = {}


async def init_models():
    """Initialize ML models for the API."""
    try:
        logger.info("Initializing ML models...")
        
        # Determine device
        device = _get_device()
        logger.info(f"Using device: {device}")
        
        # Initialize renderer classifier
        await _init_renderer_classifier(device)
        
        # Initialize OCR models
        await _init_ocr_models(device)
        
        # Initialize table detector
        await _init_table_detector(device)
        
        # Initialize layout analyzer
        await _init_layout_analyzer(device)
        
        logger.info("ML models initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize ML models: {str(e)}")
        # Don't raise exception - API should still work without models
        # Models will be loaded on-demand by workers


async def _init_renderer_classifier(device: str):
    """Initialize the renderer classifier model."""
    try:
        from models.renderer_classifier import RendererClassifier
        
        model_path = os.path.join(settings.models_path, "renderer_classifier.pth")
        
        if os.path.exists(model_path):
            model = RendererClassifier()
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            
            _models["renderer_classifier"] = model
            logger.info("Renderer classifier loaded successfully")
        else:
            logger.warning(f"Renderer classifier model not found at {model_path}")
            
    except Exception as e:
        logger.error(f"Failed to load renderer classifier: {str(e)}")


async def _init_ocr_models(device: str):
    """Initialize OCR models."""
    try:
        from models.ocr_models import TesseractOCR, EasyOCR
        
        # Initialize Tesseract
        tesseract_ocr = TesseractOCR()
        _models["tesseract_ocr"] = tesseract_ocr
        logger.info("Tesseract OCR initialized")
        
        # Initialize EasyOCR if available
        try:
            easyocr_model = EasyOCR(["en"], gpu=(device != "cpu"))
            _models["easyocr"] = easyocr_model
            logger.info("EasyOCR initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize EasyOCR: {str(e)}")
            
    except Exception as e:
        logger.error(f"Failed to initialize OCR models: {str(e)}")


async def _init_table_detector(device: str):
    """Initialize table detection model."""
    try:
        from models.table_detector import TableDetector
        
        model_path = os.path.join(settings.models_path, "table_detector.pth")
        
        if os.path.exists(model_path):
            model = TableDetector()
            # Load model weights if available
            _models["table_detector"] = model
            logger.info("Table detector initialized")
        else:
            logger.warning(f"Table detector model not found at {model_path}")
            
    except Exception as e:
        logger.error(f"Failed to initialize table detector: {str(e)}")


async def _init_layout_analyzer(device: str):
    """Initialize layout analysis model."""
    try:
        from models.layout_analyzer import LayoutAnalyzer
        
        model_path = os.path.join(settings.models_path, "layout_analyzer.pth")
        
        if os.path.exists(model_path):
            model = LayoutAnalyzer()
            # Load model weights if available
            _models["layout_analyzer"] = model
            logger.info("Layout analyzer initialized")
        else:
            logger.warning(f"Layout analyzer model not found at {model_path}")
            
    except Exception as e:
        logger.error(f"Failed to initialize layout analyzer: {str(e)}")


def _get_device() -> str:
    """Determine the best device for model inference."""
    if settings.device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    else:
        return settings.device


def get_model(model_name: str) -> Optional[Any]:
    """Get a loaded model by name."""
    return _models.get(model_name)


def is_model_loaded(model_name: str) -> bool:
    """Check if a model is loaded."""
    return model_name in _models


def get_loaded_models() -> Dict[str, Any]:
    """Get all loaded models."""
    return _models.copy()
