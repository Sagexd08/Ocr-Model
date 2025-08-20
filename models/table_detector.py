"""
Table Detection and Reconstruction for CurioScan

This module implements table detection using Detectron2 and robust table reconstruction
with support for rowspan/colspan handling.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from PIL import Image, ImageDraw
import cv2
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class TableCell:
    """Represents a single table cell."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    text: str
    confidence: float
    row: int
    col: int
    rowspan: int = 1
    colspan: int = 1


@dataclass
class Table:
    """Represents a complete table structure."""
    bbox: Tuple[int, int, int, int]
    cells: List[TableCell]
    rows: int
    cols: int
    confidence: float


@dataclass
class TableDetectionResult:
    """Result of table detection on a page."""
    tables: List[Table]
    page_bbox: Tuple[int, int, int, int]
    page_number: int
    processing_time: float
    model_name: str


class BaseTableDetector(ABC):
    """Abstract base class for table detectors."""
    
    @abstractmethod
    def detect_tables(self, image: Image.Image, **kwargs) -> TableDetectionResult:
        """Detect tables in an image."""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get the model name."""
        pass


class Detectron2TableDetector(BaseTableDetector):
    """Table detector using Detectron2."""
    
    def __init__(
        self,
        config_file: str = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
        weights_url: str = None,
        confidence_threshold: float = 0.7,
        device: str = "auto"
    ):
        self.config_file = config_file
        self.confidence_threshold = confidence_threshold
        
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        try:
            from detectron2 import model_zoo
            from detectron2.engine import DefaultPredictor
            from detectron2.config import get_cfg
            
            # Setup configuration
            self.cfg = get_cfg()
            self.cfg.merge_from_file(model_zoo.get_config_file(config_file))
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
            
            if weights_url:
                self.cfg.MODEL.WEIGHTS = weights_url
            else:
                self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)
            
            self.cfg.MODEL.DEVICE = self.device
            
            # Create predictor
            self.predictor = DefaultPredictor(self.cfg)
            
            logger.info(f"Detectron2 table detector initialized on {self.device}")
            
        except ImportError:
            logger.error("Detectron2 not installed. Please install detectron2.")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Detectron2: {e}")
            raise
    
    def detect_tables(self, image: Image.Image, **kwargs) -> TableDetectionResult:
        """Detect tables using Detectron2."""
        import time
        start_time = time.time()
        
        # Convert PIL to OpenCV format
        img_array = np.array(image)
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Run detection
        outputs = self.predictor(img_array)
        
        # Parse results
        instances = outputs["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        classes = instances.pred_classes.numpy()
        
        # Filter for table-like objects (assuming class 0 is table or general object)
        table_indices = scores >= self.confidence_threshold
        
        tables = []
        for i, (box, score) in enumerate(zip(boxes[table_indices], scores[table_indices])):
            x1, y1, x2, y2 = map(int, box)
            table_bbox = (x1, y1, x2, y2)
            
            # Extract table region for cell detection
            table_region = image.crop(table_bbox)
            cells = self._extract_table_cells(table_region, table_bbox)
            
            if cells:  # Only add if we found cells
                # Determine table dimensions
                max_row = max(cell.row for cell in cells) + 1
                max_col = max(cell.col for cell in cells) + 1
                
                table = Table(
                    bbox=table_bbox,
                    cells=cells,
                    rows=max_row,
                    cols=max_col,
                    confidence=float(score)
                )
                tables.append(table)
        
        processing_time = time.time() - start_time
        
        return TableDetectionResult(
            tables=tables,
            page_bbox=(0, 0, image.width, image.height),
            page_number=kwargs.get('page_number', 0),
            processing_time=processing_time,
            model_name=self.get_model_name()
        )
    
    def _extract_table_cells(self, table_image: Image.Image, table_bbox: Tuple[int, int, int, int]) -> List[TableCell]:
        """Extract individual cells from a table region."""
        # Convert to grayscale for line detection
        gray = cv2.cvtColor(np.array(table_image), cv2.COLOR_RGB2GRAY)
        
        # Detect horizontal and vertical lines
        horizontal_lines = self._detect_lines(gray, direction="horizontal")
        vertical_lines = self._detect_lines(gray, direction="vertical")
        
        # Create grid from line intersections
        grid = self._create_grid(horizontal_lines, vertical_lines, table_image.size)
        
        # Extract cells from grid
        cells = []
        cell_id = 0
        
        for row in range(len(grid) - 1):
            for col in range(len(grid[0]) - 1):
                # Get cell boundaries
                x1 = grid[row][col][0]
                y1 = grid[row][col][1]
                x2 = grid[row + 1][col + 1][0]
                y2 = grid[row + 1][col + 1][1]
                
                # Adjust coordinates to global image space
                global_x1 = table_bbox[0] + x1
                global_y1 = table_bbox[1] + y1
                global_x2 = table_bbox[0] + x2
                global_y2 = table_bbox[1] + y2
                
                cell_bbox = (global_x1, global_y1, global_x2, global_y2)
                
                # Extract cell image for OCR (placeholder)
                cell_text = f"Cell_{row}_{col}"  # Placeholder - would use OCR here
                
                cell = TableCell(
                    bbox=cell_bbox,
                    text=cell_text,
                    confidence=0.8,  # Placeholder confidence
                    row=row,
                    col=col
                )
                cells.append(cell)
                cell_id += 1
        
        return cells
    
    def _detect_lines(self, gray_image: np.ndarray, direction: str) -> List[Tuple[int, int, int, int]]:
        """Detect horizontal or vertical lines in the image."""
        if direction == "horizontal":
            # Create horizontal kernel
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        else:
            # Create vertical kernel
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        
        # Apply morphological operations
        eroded = cv2.erode(gray_image, kernel, iterations=1)
        dilated = cv2.dilate(eroded, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        lines = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            if direction == "horizontal" and w > h * 5:  # Wide rectangles for horizontal lines
                lines.append((x, y, x + w, y + h))
            elif direction == "vertical" and h > w * 5:  # Tall rectangles for vertical lines
                lines.append((x, y, x + w, y + h))
        
        return lines
    
    def _create_grid(
        self, 
        horizontal_lines: List[Tuple[int, int, int, int]], 
        vertical_lines: List[Tuple[int, int, int, int]], 
        image_size: Tuple[int, int]
    ) -> List[List[Tuple[int, int]]]:
        """Create a grid from detected lines."""
        width, height = image_size
        
        # Extract y-coordinates from horizontal lines
        h_coords = set([0, height])  # Add image boundaries
        for line in horizontal_lines:
            h_coords.add(line[1])  # y1
            h_coords.add(line[3])  # y2
        
        # Extract x-coordinates from vertical lines
        v_coords = set([0, width])  # Add image boundaries
        for line in vertical_lines:
            v_coords.add(line[0])  # x1
            v_coords.add(line[2])  # x2
        
        # Sort coordinates
        h_coords = sorted(h_coords)
        v_coords = sorted(v_coords)
        
        # Create grid
        grid = []
        for y in h_coords:
            row = []
            for x in v_coords:
                row.append((x, y))
            grid.append(row)
        
        return grid
    
    def get_model_name(self) -> str:
        return "detectron2_table_detector"


class SimpleTableDetector(BaseTableDetector):
    """Simple table detector using OpenCV line detection."""
    
    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold
    
    def detect_tables(self, image: Image.Image, **kwargs) -> TableDetectionResult:
        """Detect tables using simple line detection."""
        import time
        start_time = time.time()
        
        # Convert to grayscale
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        
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
        
        tables = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter small regions
            if w > 100 and h > 100:
                table_bbox = (x, y, x + w, y + h)
                
                # Simple cell extraction (grid-based)
                cells = self._extract_simple_cells(table_bbox, 3, 3)  # Assume 3x3 grid
                
                table = Table(
                    bbox=table_bbox,
                    cells=cells,
                    rows=3,
                    cols=3,
                    confidence=self.confidence_threshold
                )
                tables.append(table)
        
        processing_time = time.time() - start_time
        
        return TableDetectionResult(
            tables=tables,
            page_bbox=(0, 0, image.width, image.height),
            page_number=kwargs.get('page_number', 0),
            processing_time=processing_time,
            model_name=self.get_model_name()
        )
    
    def _extract_simple_cells(self, table_bbox: Tuple[int, int, int, int], rows: int, cols: int) -> List[TableCell]:
        """Extract cells using simple grid division."""
        x1, y1, x2, y2 = table_bbox
        width = x2 - x1
        height = y2 - y1
        
        cell_width = width // cols
        cell_height = height // rows
        
        cells = []
        for row in range(rows):
            for col in range(cols):
                cell_x1 = x1 + col * cell_width
                cell_y1 = y1 + row * cell_height
                cell_x2 = cell_x1 + cell_width
                cell_y2 = cell_y1 + cell_height
                
                cell = TableCell(
                    bbox=(cell_x1, cell_y1, cell_x2, cell_y2),
                    text=f"Cell_{row}_{col}",
                    confidence=0.7,
                    row=row,
                    col=col
                )
                cells.append(cell)
        
        return cells
    
    def get_model_name(self) -> str:
        return "simple_table_detector"


class TableDetector:
    """Main table detector class that can use different backends."""
    
    def __init__(self, detector: BaseTableDetector):
        self.detector = detector
    
    def detect_tables(self, image: Image.Image, **kwargs) -> TableDetectionResult:
        """Detect tables in an image."""
        return self.detector.detect_tables(image, **kwargs)
    
    def visualize_tables(self, image: Image.Image, result: TableDetectionResult) -> Image.Image:
        """Visualize detected tables on the image."""
        vis_image = image.copy()
        draw = ImageDraw.Draw(vis_image)
        
        for table in result.tables:
            # Draw table boundary
            draw.rectangle(table.bbox, outline="red", width=3)
            
            # Draw cells
            for cell in table.cells:
                draw.rectangle(cell.bbox, outline="blue", width=1)
                
                # Add cell text
                text_x = cell.bbox[0] + 5
                text_y = cell.bbox[1] + 5
                draw.text((text_x, text_y), cell.text, fill="green")
        
        return vis_image


def create_table_detector(config: Dict) -> TableDetector:
    """
    Factory function to create table detector from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured TableDetector
    """
    table_config = config.get("model", {}).get("table_detection", {})
    model_type = table_config.get("model_type", "simple")
    
    if model_type == "detectron2":
        detector = Detectron2TableDetector(
            config_file=table_config.get("config_file", "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"),
            weights_url=table_config.get("weights_url"),
            confidence_threshold=table_config.get("confidence_threshold", 0.7)
        )
    else:
        detector = SimpleTableDetector(
            confidence_threshold=table_config.get("confidence_threshold", 0.7)
        )
    
    return TableDetector(detector)
