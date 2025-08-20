"""
Layout Analysis for CurioScan

This module implements document layout analysis to identify different regions
like text blocks, tables, images, headers, etc.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from PIL import Image, ImageDraw
import cv2
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class RegionType(Enum):
    """Types of document regions."""
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    HEADER = "header"
    FOOTER = "footer"
    TITLE = "title"
    LIST = "list"
    FIGURE = "figure"
    CAPTION = "caption"
    UNKNOWN = "unknown"


@dataclass
class LayoutRegion:
    """Represents a layout region in a document."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    region_type: RegionType
    confidence: float
    region_id: str
    text_content: Optional[str] = None
    reading_order: Optional[int] = None


@dataclass
class LayoutAnalysisResult:
    """Result of layout analysis on a page."""
    regions: List[LayoutRegion]
    page_bbox: Tuple[int, int, int, int]
    page_number: int
    processing_time: float
    model_name: str


class LayoutAnalyzer:
    """Document layout analyzer using computer vision techniques."""
    
    def __init__(
        self,
        min_text_area: int = 100,
        min_table_area: int = 1000,
        confidence_threshold: float = 0.7
    ):
        self.min_text_area = min_text_area
        self.min_table_area = min_table_area
        self.confidence_threshold = confidence_threshold
    
    def analyze_layout(self, image: Image.Image, **kwargs) -> LayoutAnalysisResult:
        """Analyze document layout."""
        import time
        start_time = time.time()
        
        # Convert to OpenCV format
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Detect different types of regions
        regions = []
        
        # 1. Detect text regions
        text_regions = self._detect_text_regions(gray, image.size)
        regions.extend(text_regions)
        
        # 2. Detect table regions
        table_regions = self._detect_table_regions(gray, image.size)
        regions.extend(table_regions)
        
        # 3. Detect image regions
        image_regions = self._detect_image_regions(gray, image.size)
        regions.extend(image_regions)
        
        # 4. Detect headers and footers
        header_footer_regions = self._detect_headers_footers(gray, image.size)
        regions.extend(header_footer_regions)
        
        # 5. Assign reading order
        regions = self._assign_reading_order(regions)
        
        processing_time = time.time() - start_time
        
        return LayoutAnalysisResult(
            regions=regions,
            page_bbox=(0, 0, image.width, image.height),
            page_number=kwargs.get('page_number', 0),
            processing_time=processing_time,
            model_name="cv_layout_analyzer"
        )
    
    def _detect_text_regions(self, gray: np.ndarray, image_size: Tuple[int, int]) -> List[LayoutRegion]:
        """Detect text regions using morphological operations."""
        # Apply threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Create kernel for text detection
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        
        # Apply morphological operations
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        region_id = 0
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            # Filter by area and aspect ratio
            if area > self.min_text_area and 0.1 < h/w < 10:
                bbox = (x, y, x + w, y + h)
                
                region = LayoutRegion(
                    bbox=bbox,
                    region_type=RegionType.TEXT,
                    confidence=0.8,
                    region_id=f"text_{region_id}"
                )
                regions.append(region)
                region_id += 1
        
        return regions
    
    def _detect_table_regions(self, gray: np.ndarray, image_size: Tuple[int, int]) -> List[LayoutRegion]:
        """Detect table regions using line detection."""
        # Apply threshold
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
        
        # Detect horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
        
        # Combine lines
        table_mask = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
        
        # Find table regions
        contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        region_id = 0
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            # Filter by area
            if area > self.min_table_area:
                bbox = (x, y, x + w, y + h)
                
                region = LayoutRegion(
                    bbox=bbox,
                    region_type=RegionType.TABLE,
                    confidence=0.7,
                    region_id=f"table_{region_id}"
                )
                regions.append(region)
                region_id += 1
        
        return regions
    
    def _detect_image_regions(self, gray: np.ndarray, image_size: Tuple[int, int]) -> List[LayoutRegion]:
        """Detect image/figure regions."""
        # Use edge detection to find image boundaries
        edges = cv2.Canny(gray, 50, 150)
        
        # Apply morphological operations to connect edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        region_id = 0
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            # Filter by area and aspect ratio (images tend to be more square)
            if area > 5000 and 0.3 < h/w < 3:
                bbox = (x, y, x + w, y + h)
                
                region = LayoutRegion(
                    bbox=bbox,
                    region_type=RegionType.IMAGE,
                    confidence=0.6,
                    region_id=f"image_{region_id}"
                )
                regions.append(region)
                region_id += 1
        
        return regions
    
    def _detect_headers_footers(self, gray: np.ndarray, image_size: Tuple[int, int]) -> List[LayoutRegion]:
        """Detect header and footer regions."""
        width, height = image_size
        regions = []
        
        # Define header and footer zones (top and bottom 10% of page)
        header_zone = height * 0.1
        footer_zone = height * 0.9
        
        # Apply threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        header_id = 0
        footer_id = 0
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            center_y = y + h // 2
            
            # Check if in header zone
            if center_y < header_zone and w > width * 0.3:  # Wide enough to be header
                bbox = (x, y, x + w, y + h)
                
                region = LayoutRegion(
                    bbox=bbox,
                    region_type=RegionType.HEADER,
                    confidence=0.7,
                    region_id=f"header_{header_id}"
                )
                regions.append(region)
                header_id += 1
            
            # Check if in footer zone
            elif center_y > footer_zone and w > width * 0.3:  # Wide enough to be footer
                bbox = (x, y, x + w, y + h)
                
                region = LayoutRegion(
                    bbox=bbox,
                    region_type=RegionType.FOOTER,
                    confidence=0.7,
                    region_id=f"footer_{footer_id}"
                )
                regions.append(region)
                footer_id += 1
        
        return regions
    
    def _assign_reading_order(self, regions: List[LayoutRegion]) -> List[LayoutRegion]:
        """Assign reading order to regions based on position."""
        # Sort regions by vertical position (top to bottom), then horizontal (left to right)
        def sort_key(region):
            x1, y1, x2, y2 = region.bbox
            center_y = (y1 + y2) // 2
            center_x = (x1 + x2) // 2
            return (center_y // 50, center_x)  # Group by rows of ~50 pixels
        
        sorted_regions = sorted(regions, key=sort_key)
        
        # Assign reading order
        for i, region in enumerate(sorted_regions):
            region.reading_order = i
        
        return sorted_regions
    
    def visualize_layout(self, image: Image.Image, result: LayoutAnalysisResult) -> Image.Image:
        """Visualize detected layout regions."""
        vis_image = image.copy()
        draw = ImageDraw.Draw(vis_image)
        
        # Color mapping for different region types
        colors = {
            RegionType.TEXT: "blue",
            RegionType.TABLE: "red",
            RegionType.IMAGE: "green",
            RegionType.HEADER: "purple",
            RegionType.FOOTER: "orange",
            RegionType.TITLE: "yellow",
            RegionType.LIST: "cyan",
            RegionType.FIGURE: "magenta",
            RegionType.CAPTION: "brown",
            RegionType.UNKNOWN: "gray"
        }
        
        for region in result.regions:
            color = colors.get(region.region_type, "gray")
            
            # Draw bounding box
            draw.rectangle(region.bbox, outline=color, width=2)
            
            # Add label
            label = f"{region.region_type.value}_{region.reading_order}"
            text_x = region.bbox[0] + 5
            text_y = region.bbox[1] + 5
            draw.text((text_x, text_y), label, fill=color)
        
        return vis_image
    
    def get_reading_order_regions(self, result: LayoutAnalysisResult) -> List[LayoutRegion]:
        """Get regions sorted by reading order."""
        return sorted(result.regions, key=lambda r: r.reading_order or 0)
    
    def filter_regions_by_type(self, result: LayoutAnalysisResult, region_type: RegionType) -> List[LayoutRegion]:
        """Filter regions by type."""
        return [region for region in result.regions if region.region_type == region_type]


def create_layout_analyzer(config: Dict) -> LayoutAnalyzer:
    """
    Factory function to create layout analyzer from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured LayoutAnalyzer
    """
    processing_config = config.get("processing", {})
    
    return LayoutAnalyzer(
        min_text_area=processing_config.get("min_text_area", 100),
        min_table_area=processing_config.get("min_table_area", 1000),
        confidence_threshold=processing_config.get("confidence_thresholds", {}).get("region_detection", 0.7)
    )
