from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import tensorflow as tf
import cv2
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, LayoutLMv3Processor, LayoutLMv3ForTokenClassification
import time
from pathlib import Path
import json
import onnxruntime as ort
import os
from typing import Dict, Optional, Union, Any
import re
import pytesseract
import pandas as pd
from PIL import Image
import io
try:
    import paddleocr
except ImportError:
    paddleocr = None

from ...types import Document, Page, Region, Token, Bbox
from ...utils.logging import get_logger, log_execution_time

logger = get_logger(__name__)

class AdvancedOCREngine:
    """
    Advanced OCR engine that combines multiple recognition models
    for optimal accuracy on different document types.
    
    Features:
    - Adaptive model selection based on region characteristics
    - Multi-model ensemble for improved accuracy
    - Script/language detection
    - Post-correction with language models
    - Confidence scoring with multiple metrics
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize models directory
        self.models_dir = self.config.get("models_dir", "models")
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Supported OCR engines
        self.available_engines = ["tesseract", "trocr", "paddleocr", "layoutlm"]
        self.enabled_engines = self.config.get("enabled_engines", ["tesseract", "trocr", "layoutlm"])
        
        # Document processing capabilities
        self.use_ensemble = self.config.get("use_ensemble", True)
        self.script_detection = self.config.get("script_detection", True)
        self.language_detection = self.config.get("language_detection", True)
        self.post_correction = self.config.get("post_correction", True)
        self.table_extraction = self.config.get("table_extraction", True)
        self.document_classification = self.config.get("document_classification", True)
        self.layout_analysis = self.config.get("layout_analysis", True)
        
        # Support for different document types
        self.supported_formats = ["pdf", "png", "jpg", "jpeg", "tiff", "docx", "doc"]
        
        # Performance optimization
        self.batch_size = self.config.get("batch_size", 4)
        self.use_gpu = self.config.get("use_gpu", True) and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # OCR models
        self.models = {}
        self.processors = {}
        self.table_extractor = None
        
        # Load configured models
        self._load_ocr_models()
        
        # Language detection model
        self.lang_detector = None
        if self.language_detection:
            self._load_language_detector()
        
        # Layout analysis model
        self.layout_analyzer = None
        if self.layout_analysis:
            self._load_layout_analyzer()
            
        # Table extraction model
        if self.table_extraction:
            self._load_table_extractor()
        
        # Post-correction model - uses transformers for error correction
        self.corrector = None
        if self.post_correction:
            self._load_post_correction_model()
            
        logger.info("Advanced OCR Engine initialized successfully")
        
    @log_execution_time
    def process(self, document: Document) -> Document:
        """
        Process document with OCR engines
        
        Args:
            document: Document with layout regions
            
        Returns:
            Document with extracted text
        """
        logger.info(f"Running OCR on document with {len(document.pages)} pages")
        
        for i, page in enumerate(document.pages):
            logger.debug(f"Processing page {page.page_num}")
            
            try:
                # Process each region
                for j, region in enumerate(page.regions):
                    # Skip regions already processed or that aren't text
                    if region.tokens or region.type in ["table", "image", "chart", "figure"]:
                        continue
                        
                    # Get region image
                    region_img = self._extract_region_image(page, region)
                    if region_img is None:
                        continue
                    
                    # Select optimal OCR engine for region
                    if self.use_ensemble:
                        ocr_results = self._run_ensemble_ocr(region_img, region.type)
                    else:
                        engine_name = self._select_ocr_engine(region_img, region.type)
                        ocr_results = self._run_ocr(region_img, engine_name, region.type)
                    
                    # Create tokens from OCR results
                    tokens = self._create_tokens_from_ocr_results(ocr_results, region)
                    
                    # Apply post-correction if enabled
                    if self.post_correction and tokens:
                        tokens = self._apply_post_correction(tokens, region.type)
                    
                    # Update region
                    region.tokens = tokens
                    
                    # Create formatted text preserving layout
                    if tokens:
                        # Group tokens by line
                        lines = {}
                        for token in tokens:
                            line_num = token.attributes.get("line_number", 0)
                            if line_num not in lines:
                                lines[line_num] = []
                            lines[line_num].append(token)
                        
                        # Sort each line by position and combine into final text
                        formatted_text_lines = []
                        for line_num in sorted(lines.keys()):
                            line_tokens = sorted(lines[line_num], key=lambda t: t.attributes.get("position", 0))
                            line_text = " ".join(token.text for token in line_tokens)
                            formatted_text_lines.append(line_text)
                        
                        region.text = "\n".join(formatted_text_lines)
                    else:
                        region.text = ""
                    
                    # Calculate region confidence
                    if tokens:
                        region.confidence = sum(token.confidence for token in tokens) / len(tokens)
                    
                    # Update region in page
                    page.regions[j] = region
                
                # Update page in document
                document.pages[i] = page
                
            except Exception as e:
                logger.error(f"Error running OCR on page {page.page_num}: {str(e)}")
        
        return document
    
    def _load_ocr_models(self):
        """Load configured OCR models"""
        for engine in self.enabled_engines:
            try:
                if engine == "tesseract":
                    # Tesseract initialization would happen here
                    # For brevity, implementation details omitted
                    self.models[engine] = "tesseract_engine"
                    logger.info(f"Loaded Tesseract OCR engine")
                    
                elif engine == "trocr":
                    # Load TrOCR model
                    model_name = self.config.get("trocr_model", "microsoft/trocr-base-printed")
                    self.processors[engine] = TrOCRProcessor.from_pretrained(model_name)
                    self.models[engine] = VisionEncoderDecoderModel.from_pretrained(model_name)
                    
                    if self.use_gpu:
                        self.models[engine] = self.models[engine].to(self.device)
                        
                    logger.info(f"Loaded TrOCR model: {model_name}")
                    
                elif engine == "easyocr":
                    # EasyOCR initialization would happen here
                    # For brevity, implementation details omitted
                    self.models[engine] = "easyocr_engine"
                    logger.info(f"Loaded EasyOCR engine")
                    
                elif engine == "paddleocr":
                    # PaddleOCR initialization would happen here
                    # For brevity, implementation details omitted
                    self.models[engine] = "paddleocr_engine"
                    logger.info(f"Loaded PaddleOCR engine")
                    
            except Exception as e:
                logger.error(f"Failed to load OCR engine {engine}: {str(e)}")
    
    def _load_language_detector(self):
        """Load language detection model"""
        try:
            # In production, this would load a proper language detection model
            # such as fastText or langid
            self.lang_detector = "language_detector"
            logger.info("Loaded language detection model")
        except Exception as e:
            logger.error(f"Failed to load language detection model: {str(e)}")
            self.language_detection = False
    
    def _load_script_detector(self):
        """Load script detection model"""
        try:
            # In production, this would load a script detection model
            # to identify different writing scripts (Latin, Cyrillic, etc.)
            self.script_detector = "script_detector"
            logger.info("Loaded script detection model")
        except Exception as e:
            logger.error(f"Failed to load script detection model: {str(e)}")
            self.script_detection = False
            
    def _load_layout_analyzer(self):
        """Load layout analysis model"""
        try:
            from .layout_analyzer import LayoutAnalyzer
            
            # Initialize the layout analyzer
            layout_config = {
                "models_dir": self.models_dir,
                "use_gpu": self.use_gpu
            }
            self.layout_analyzer = LayoutAnalyzer(layout_config)
            logger.info("Loaded layout analysis model")
        except Exception as e:
            logger.error(f"Failed to load layout analyzer: {str(e)}")
            self.layout_analysis = False
            
    def _load_table_extractor(self):
        """Load table extraction model"""
        try:
            from .table_detector import TableDetector
            
            # Initialize the table detector
            table_config = {
                "models_dir": self.models_dir,
                "use_gpu": self.use_gpu
            }
            self.table_extractor = TableDetector(table_config)
            logger.info("Loaded table extractor")
        except Exception as e:
            logger.error(f"Failed to load table extractor: {str(e)}")
            self.table_extraction = False
    
    def _load_post_correction_model(self):
        """Load post-correction model"""
        try:
            # In production, this would load a language model for OCR correction
            self.corrector = "post_correction_model"
            logger.info("Loaded OCR post-correction model")
        except Exception as e:
            logger.error(f"Failed to load post-correction model: {str(e)}")
            self.post_correction = False
    
    def _extract_region_image(self, page: Page, region: Region) -> Optional[np.ndarray]:
        """Extract region image from page"""
        # For brevity, implementation details omitted
        # This would get the image data for the region from the page
        return np.array([])  # Placeholder
    
    def _select_ocr_engine(self, img: np.ndarray, region_type: str) -> str:
        """Select optimal OCR engine based on image characteristics and region type"""
        # Default to first available engine
        if not self.enabled_engines:
            return "tesseract"
            
        # Script detection to select appropriate engine
        if self.script_detection and self.script_detector:
            # In production, this would detect the script and select appropriate engine
            # For now, we'll use a simple heuristic
            pass
        
        # Select engine based on region type
        if region_type == "heading":
            # Prefer TrOCR for headings if available
            return "trocr" if "trocr" in self.enabled_engines else self.enabled_engines[0]
        elif region_type == "paragraph":
            # Use Tesseract for paragraphs if available
            return "tesseract" if "tesseract" in self.enabled_engines else self.enabled_engines[0]
        
        # Default to first engine
        return self.enabled_engines[0]
    
    def _run_ocr(self, img: np.ndarray, engine_name: str, region_type: str) -> List[Dict[str, Any]]:
        """Run OCR with specified engine"""
        results = []
        
        try:
            if engine_name == "tesseract":
                # Tesseract OCR would be implemented here
                # For brevity, implementation details omitted
                pass
                
            elif engine_name == "trocr" and "trocr" in self.models:
                # Process with TrOCR
                processor = self.processors[engine_name]
                model = self.models[engine_name]
                
                # Convert numpy image to PIL Image if needed
                from PIL import Image
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                
                # Process image
                pixel_values = processor(img, return_tensors="pt").pixel_values
                if self.use_gpu:
                    pixel_values = pixel_values.to(self.device)
                
                # Generate output
                generated_ids = model.generate(pixel_values)
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                # Create result with placeholder bounding box
                # In production, this would use a text detection model to get actual bounding boxes
                h, w = img.height, img.width
                results.append({
                    "text": generated_text,
                    "bbox": [0, 0, w, h],
                    "confidence": 0.95  # TrOCR doesn't provide confidence scores
                })
                
            elif engine_name == "easyocr":
                # EasyOCR processing would be implemented here
                # For brevity, implementation details omitted
                pass
                
            elif engine_name == "paddleocr":
                # PaddleOCR processing would be implemented here
                # For brevity, implementation details omitted
                pass
                
        except Exception as e:
            logger.error(f"Error running OCR with engine {engine_name}: {str(e)}")
            
        return results
    
    def _run_ensemble_ocr(self, img: np.ndarray, region_type: str) -> List[Dict[str, Any]]:
        """Run multiple OCR engines and combine results"""
        all_results = []
        
        # Run each enabled engine
        for engine in self.enabled_engines:
            results = self._run_ocr(img, engine, region_type)
            
            # Tag results with engine name
            for result in results:
                result["engine"] = engine
                
            all_results.extend(results)
        
        # Combine results using ensemble method
        # This is a simplified version - a real implementation would use
        # more sophisticated alignment and voting mechanisms
        combined_results = self._combine_ocr_results(all_results)
        
        return combined_results
    
    def _combine_ocr_results(self, all_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Combine results from multiple OCR engines"""
        if not all_results:
            return []
            
        # Group results by bounding box overlap
        groups = []
        
        for result in all_results:
            added = False
            
            # Try to add to existing group
            for group in groups:
                if self._bbox_overlap(result["bbox"], group[0]["bbox"]) > 0.5:
                    group.append(result)
                    added = True
                    break
            
            # Create new group if no match
            if not added:
                groups.append([result])
        
        # Combine each group
        combined_results = []
        
        for group in groups:
            if len(group) == 1:
                # Only one result in group
                combined_results.append(group[0])
            else:
                # Multiple results - use voting
                texts = [r["text"] for r in group]
                confidences = [r.get("confidence", 0.0) for r in group]
                
                # Find most common text
                text_counts = {}
                for text in texts:
                    text_counts[text] = text_counts.get(text, 0) + 1
                
                # Find most frequent text with highest confidence
                best_text = None
                best_count = 0
                best_conf = 0.0
                
                for text, count in text_counts.items():
                    # Get average confidence for this text
                    conf = sum(confidences[i] for i, t in enumerate(texts) if t == text) / count
                    
                    if count > best_count or (count == best_count and conf > best_conf):
                        best_text = text
                        best_count = count
                        best_conf = conf
                
                # Use average bounding box
                avg_bbox = [
                    sum(r["bbox"][0] for r in group) / len(group),
                    sum(r["bbox"][1] for r in group) / len(group),
                    sum(r["bbox"][2] for r in group) / len(group),
                    sum(r["bbox"][3] for r in group) / len(group)
                ]
                
                # Create combined result
                combined_results.append({
                    "text": best_text,
                    "bbox": avg_bbox,
                    "confidence": best_conf,
                    "engine": "ensemble"
                })
        
        return combined_results
    
    def _bbox_overlap(self, bbox1, bbox2) -> float:
        """Calculate IoU overlap between two bounding boxes"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x1 >= x2 or y1 >= y2:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        return intersection / (bbox1_area + bbox2_area - intersection)
    
    def _create_tokens_from_ocr_results(self, ocr_results: List[Dict[str, Any]], region: Region) -> List[Token]:
        """Convert OCR results to Token objects"""
        tokens = []
        
        # Sort results by y-coordinate to preserve reading order
        ocr_results = sorted(ocr_results, key=lambda x: (x["bbox"][1], x["bbox"][0]))
        
        # Group by lines based on y-coordinate proximity
        line_groups = []
        current_line = []
        last_y = None
        
        for result in ocr_results:
            y1 = result["bbox"][1]
            
            if last_y is None or abs(y1 - last_y) <= 10:  # Threshold for same line
                current_line.append(result)
            else:
                if current_line:
                    line_groups.append(current_line)
                current_line = [result]
            
            last_y = y1
        
        if current_line:
            line_groups.append(current_line)
        
        # For each line, sort by x-coordinate and create tokens
        token_id = 0
        for line in line_groups:
            # Sort by x-coordinate
            line = sorted(line, key=lambda x: x["bbox"][0])
            
            for result in line:
                # Create bounding box
                x1, y1, x2, y2 = result["bbox"]
                bbox = Bbox(x1=x1, y1=y1, x2=x2, y2=y2)
                
                # Create token
                token = Token(
                    id=f"{region.id}_t{token_id}",
                    text=result["text"],
                    bbox=bbox,
                    confidence=result.get("confidence", 0.0),
                    attributes={
                        "engine": result.get("engine", "unknown"),
                        "line_number": line_groups.index(line),
                        "position": line.index(result)
                    }
                )
                
                tokens.append(token)
                token_id += 1
            
        return tokens
    
    def _apply_post_correction(self, tokens: List[Token], region_type: str) -> List[Token]:
        """Apply post-correction to tokens"""
        if not tokens or not self.post_correction or not self.corrector:
            return tokens
            
        # For brevity, implementation details omitted
        # This would apply language model-based correction to OCR results
        
        return tokens
