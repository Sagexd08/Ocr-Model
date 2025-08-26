"""
Layout Analysis Module for Document Processing Pipeline
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import cv2
import torch
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from PIL import Image
import os
from pathlib import Path
import json

from ...types import Document, Page, Region, Bbox
from ...utils.logging import get_logger, log_execution_time

logger = get_logger(__name__)

class LayoutAnalyzer:
    """
    LayoutAnalyzer uses LayoutLMv3 to analyze document layout and classify regions
    into types such as header, paragraph, table, figure, etc.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize models directory
        self.models_dir = self.config.get("models_dir", "models")
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Model configuration
        self.model_name = self.config.get("layout_model", "microsoft/layoutlmv3-base-finetuned-publaynet")
        self.use_gpu = self.config.get("use_gpu", True) and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
        
        # Load the model
        self.processor = None
        self.model = None
        self._load_model()
        
        # Label mapping (standard PublayNet classes)
        self.id2label = {
            0: "text",
            1: "title",
            2: "list",
            3: "table",
            4: "figure"
        }
        
        # Enhanced label mapping with additional classes
        self.enhanced_labels = {
            "text": "paragraph",
            "title": "heading",
            "list": "list",
            "table": "table",
            "figure": "figure",
            "header": "header",
            "footer": "footer",
            "page_number": "page_number",
            "caption": "caption",
            "footnote": "footnote",
            "signature": "signature",
            "stamp": "stamp",
            "form": "form"
        }
        
    def _load_model(self):
        """Load the LayoutLMv3 model for layout analysis"""
        try:
            logger.info(f"Loading LayoutLMv3 model from {self.model_name}")
            self.processor = LayoutLMv3Processor.from_pretrained(self.model_name)
            self.model = LayoutLMv3ForTokenClassification.from_pretrained(self.model_name)
            
            if self.use_gpu:
                self.model = self.model.to(self.device)
                
            logger.info(f"Successfully loaded LayoutLMv3 model")
            
        except Exception as e:
            logger.error(f"Failed to load LayoutLMv3 model: {e}")
            raise
    
    @log_execution_time
    def analyze(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Analyze document layout from image
        
        Args:
            image: np.ndarray - Document image
            
        Returns:
            List of detected regions with classification and bounding boxes
        """
        if self.model is None or self.processor is None:
            logger.error("Model not loaded")
            return []
        
        try:
            # Convert numpy array to PIL Image if needed
            if isinstance(image, np.ndarray):
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Prepare input features
            encoding = self.processor(image, return_tensors="pt")
            
            # Move to GPU if available
            if self.use_gpu:
                for k, v in encoding.items():
                    if isinstance(v, torch.Tensor):
                        encoding[k] = v.to(self.device)
            
            # Perform inference
            with torch.no_grad():
                outputs = self.model(**encoding)
                
            # Process predictions
            predictions = outputs.logits.argmax(-1).squeeze().cpu().numpy()
            token_boxes = encoding["bbox"][0].cpu().numpy()
            
            # Group by regions
            regions = self._group_by_regions(predictions, token_boxes, image.width, image.height)
            
            return regions
            
        except Exception as e:
            logger.error(f"Error during layout analysis: {e}")
            return []
    
    def _group_by_regions(self, predictions: np.ndarray, boxes: np.ndarray, 
                         img_width: int, img_height: int) -> List[Dict[str, Any]]:
        """
        Group token predictions into coherent regions
        
        Args:
            predictions: Class predictions for each token
            boxes: Bounding boxes for each token
            img_width: Original image width
            img_height: Original image height
            
        Returns:
            List of regions with type and bounding box
        """
        regions = []
        current_class = None
        current_tokens = []
        
        # Group tokens by class
        for pred, box in zip(predictions, boxes):
            if pred == 0 or box.sum() == 0:  # Skip padding tokens
                continue
                
            label = self.id2label.get(pred, "text")
            
            if current_class != label:
                # Save the current group if it exists
                if current_tokens:
                    region = self._create_region_from_tokens(current_class, current_tokens, img_width, img_height)
                    regions.append(region)
                    current_tokens = []
                
                current_class = label
            
            current_tokens.append(box)
        
        # Add the last group
        if current_tokens:
            region = self._create_region_from_tokens(current_class, current_tokens, img_width, img_height)
            regions.append(region)
        
        # Post-process regions to merge nearby regions of the same type
        regions = self._merge_nearby_regions(regions)
        
        return regions
    
    def _create_region_from_tokens(self, class_label: str, token_boxes: List[np.ndarray], 
                                 img_width: int, img_height: int) -> Dict[str, Any]:
        """
        Create a region from a group of tokens with the same class
        
        Args:
            class_label: The class of the region
            token_boxes: List of token bounding boxes
            img_width: Original image width
            img_height: Original image height
            
        Returns:
            Region dictionary with type and bounding box
        """
        # Calculate the enclosing bounding box
        x_min = min([box[0] for box in token_boxes])
        y_min = min([box[1] for box in token_boxes])
        x_max = max([box[2] for box in token_boxes])
        y_max = max([box[3] for box in token_boxes])
        
        # Convert to absolute coordinates
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(img_width, x_max)
        y_max = min(img_height, y_max)
        
        # Map to enhanced labels
        region_type = self.enhanced_labels.get(class_label, class_label)
        
        return {
            "type": region_type,
            "bbox": [x_min, y_min, x_max, y_max],
            "confidence": 0.95  # This is just a placeholder, real models would provide confidence
        }
    
    def _merge_nearby_regions(self, regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge nearby regions of the same type
        
        Args:
            regions: List of regions
            
        Returns:
            List of merged regions
        """
        if len(regions) <= 1:
            return regions
            
        merged_regions = []
        i = 0
        
        while i < len(regions):
            current = regions[i]
            j = i + 1
            
            # Check if there are adjacent regions of the same type
            while j < len(regions) and regions[j]["type"] == current["type"]:
                # Merge the bounding boxes
                current["bbox"] = [
                    min(current["bbox"][0], regions[j]["bbox"][0]),
                    min(current["bbox"][1], regions[j]["bbox"][1]),
                    max(current["bbox"][2], regions[j]["bbox"][2]),
                    max(current["bbox"][3], regions[j]["bbox"][3])
                ]
                j += 1
                
            merged_regions.append(current)
            i = j
            
        return merged_regions
