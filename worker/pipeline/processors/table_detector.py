"""
Table Extraction Module for Document Processing Pipeline
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import cv2
import pandas as pd
import torch
import onnxruntime as ort
import json
from pathlib import Path
import os
from PIL import Image
import io

# For PDF table extraction
try:
    import camelot
except ImportError:
    camelot = None

try:
    import tabula
except ImportError:
    tabula = None

from ...types import Document, Page, Region, Bbox, Table
from ...utils.logging import get_logger, log_execution_time

logger = get_logger(__name__)

class TableDetector:
    """
    Table detection and extraction using multiple approaches:
    1. Deep learning model for detection
    2. Camelot/Tabula for extraction from PDFs
    3. Cell structure analysis for proper table reconstruction
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize models directory
        self.models_dir = self.config.get("models_dir", "models")
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Table detection model
        self.detection_model_type = self.config.get("table_detection_model", "deepsrt")
        self.detection_model = None
        self.use_gpu = self.config.get("use_gpu", True) and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
        
        # Table extraction settings
        self.use_camelot = self.config.get("use_camelot", True) and camelot is not None
        self.use_tabula = self.config.get("use_tabula", True) and tabula is not None
        
        # Structure analysis settings
        self.min_cell_height = self.config.get("min_cell_height", 10)
        self.min_cell_width = self.config.get("min_cell_width", 20)
        self.line_detection_threshold = self.config.get("line_detection_threshold", 0.5)
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the table detection model"""
        try:
            model_path = os.path.join(self.models_dir, "table_detector.onnx")
            
            # Check if model exists, if not, download it (dummy step for this example)
            if not os.path.exists(model_path):
                logger.info(f"Model not found at {model_path}, using default detection")
                return
                
            # Load ONNX model
            logger.info(f"Loading table detection model from {model_path}")
            
            # Create ONNX Runtime session
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Use CUDA if available
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.use_gpu else ['CPUExecutionProvider']
            self.detection_model = ort.InferenceSession(model_path, session_options, providers=providers)
            
            logger.info(f"Successfully loaded table detection model")
            
        except Exception as e:
            logger.error(f"Failed to load table detection model: {e}")
    
    @log_execution_time
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect tables in an image
        
        Args:
            image: Document image
            
        Returns:
            List of detected table regions with bounding boxes
        """
        tables = []
        
        try:
            # Resize image for model input if needed
            height, width = image.shape[:2]
            
            # If we have a detection model, use it
            if self.detection_model is not None:
                # Preprocess image for model
                input_shape = (800, 800)  # Example input shape
                resized = cv2.resize(image, input_shape)
                input_data = np.expand_dims(resized.transpose(2, 0, 1), axis=0).astype(np.float32) / 255.0
                
                # Run inference
                outputs = self.detection_model.run(None, {"input": input_data})
                
                # Process outputs (format depends on model)
                # This is a simplified example
                for detection in outputs[0]:
                    confidence = detection[4]
                    if confidence > self.confidence_threshold:
                        # Denormalize coordinates
                        x1, y1, x2, y2 = detection[:4]
                        x1 *= width
                        y1 *= height
                        x2 *= width
                        y2 *= height
                        
                        tables.append({
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "confidence": float(confidence)
                        })
            else:
                # Fallback: use OpenCV to detect potential table regions
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
                
                # Detect horizontal and vertical lines
                horizontal = thresh.copy()
                vertical = thresh.copy()
                
                # Kernel size for line detection
                kernel_length_h = width // 30
                kernel_length_v = height // 30
                
                # Horizontal line detection
                kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length_h, 1))
                horizontal = cv2.erode(horizontal, kernel_h)
                horizontal = cv2.dilate(horizontal, kernel_h)
                
                # Vertical line detection
                kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length_v))
                vertical = cv2.erode(vertical, kernel_v)
                vertical = cv2.dilate(vertical, kernel_v)
                
                # Combine horizontal and vertical lines
                table_mask = cv2.add(horizontal, vertical)
                contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Filter contours by area
                min_table_area = (width * height) * 0.01  # 1% of image area
                
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    if w * h > min_table_area:
                        tables.append({
                            "bbox": [x, y, x + w, y + h],
                            "confidence": 0.8  # Default confidence for heuristic detection
                        })
            
            logger.info(f"Detected {len(tables)} tables in image")
            
        except Exception as e:
            logger.error(f"Error detecting tables: {e}")
        
        return tables
    
    @log_execution_time
    def extract(self, image: np.ndarray, table_region: Dict[str, Any], pdf_path: Optional[str] = None, page_num: int = 0) -> Optional[Table]:
        """
        Extract table data from an image region or PDF
        
        Args:
            image: Document image
            table_region: Table region with bounding box
            pdf_path: Path to PDF file (if available)
            page_num: Page number in PDF
            
        Returns:
            Table object with extracted data
        """
        try:
            # Extract the table region from the image
            x1, y1, x2, y2 = table_region["bbox"]
            table_img = image[y1:y2, x1:x2]
            
            # PDF-based extraction (more accurate if PDF is available)
            if pdf_path and os.path.exists(pdf_path) and (self.use_camelot or self.use_tabula):
                table_data = self._extract_from_pdf(pdf_path, page_num, table_region["bbox"])
            else:
                # Image-based extraction
                table_data = self._extract_from_image(table_img)
            
            if table_data is not None and len(table_data) > 0:
                # Create a table object
                table = Table(
                    id=f"table_{page_num}_{x1}_{y1}",
                    bbox=Bbox(x1=x1, y1=y1, x2=x2, y2=y2),
                    data=table_data,
                    rows=len(table_data),
                    cols=len(table_data[0]) if table_data else 0
                )
                
                # Save table image for visualization
                table_img_pil = Image.fromarray(cv2.cvtColor(table_img, cv2.COLOR_BGR2RGB))
                img_buffer = io.BytesIO()
                table_img_pil.save(img_buffer, format="PNG")
                table.image_data = img_buffer.getvalue()
                
                return table
            
        except Exception as e:
            logger.error(f"Error extracting table: {e}")
        
        return None
    
    def _extract_from_pdf(self, pdf_path: str, page_num: int, bbox: List[int]) -> List[List[str]]:
        """
        Extract table data from PDF using Camelot or Tabula
        
        Args:
            pdf_path: Path to PDF file
            page_num: Page number
            bbox: Bounding box of the table [x1, y1, x2, y2]
            
        Returns:
            2D list of cell data
        """
        try:
            if self.use_camelot:
                # Convert bbox to camelot format (0-100 range)
                x1, y1, x2, y2 = bbox
                page_area = f"{y1/100},{x1/100},{y2/100},{x2/100}"
                
                # Extract tables using Camelot
                tables = camelot.read_pdf(
                    pdf_path,
                    pages=str(page_num + 1),
                    flavor='lattice',
                    table_areas=[page_area]
                )
                
                if len(tables) > 0:
                    # Convert to our format
                    return tables[0].data.tolist()
            
            if self.use_tabula:
                # Convert bbox to tabula format
                x1, y1, x2, y2 = bbox
                area = [y1, x1, y2, x2]
                
                # Extract tables using Tabula
                tables = tabula.read_pdf(
                    pdf_path,
                    pages=page_num + 1,
                    area=area,
                    multiple_tables=False
                )
                
                if len(tables) > 0:
                    # Convert to our format
                    return tables[0].values.tolist()
        
        except Exception as e:
            logger.error(f"Error extracting table from PDF: {e}")
        
        return []
    
    def _extract_from_image(self, table_img: np.ndarray) -> List[List[str]]:
        """
        Extract table data from an image using image processing techniques
        
        Args:
            table_img: Table image region
            
        Returns:
            2D list of cell data
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(table_img, cv2.COLOR_BGR2GRAY)
            
            # Apply binary thresholding
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
            
            # Detect horizontal and vertical lines
            height, width = thresh.shape
            
            # Kernel size for line detection
            kernel_length_h = width // 30
            kernel_length_v = height // 30
            
            # Horizontal line detection
            kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length_h, 1))
            horizontal = cv2.erode(thresh, kernel_h)
            horizontal = cv2.dilate(horizontal, kernel_h)
            
            # Vertical line detection
            kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length_v))
            vertical = cv2.erode(thresh, kernel_v)
            vertical = cv2.dilate(vertical, kernel_v)
            
            # Combine horizontal and vertical lines
            table_mask = cv2.add(horizontal, vertical)
            
            # Find contours of cells
            contours, _ = cv2.findContours(table_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Get cell bounding boxes
            cells = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w > self.min_cell_width and h > self.min_cell_height:
                    cells.append([x, y, x + w, y + h])
            
            # Sort cells by position
            cells.sort(key=lambda cell: (cell[1], cell[0]))
            
            # Identify rows based on y-coordinate
            rows = []
            current_row = []
            last_y = -1
            
            for cell in cells:
                _, y, _, _ = cell
                
                if last_y == -1 or abs(y - last_y) <= self.min_cell_height:
                    current_row.append(cell)
                else:
                    if current_row:
                        # Sort cells in row by x-coordinate
                        current_row.sort(key=lambda c: c[0])
                        rows.append(current_row)
                    
                    current_row = [cell]
                
                last_y = y
            
            # Add the last row
            if current_row:
                current_row.sort(key=lambda c: c[0])
                rows.append(current_row)
            
            # Placeholder data - in a real implementation, we'd extract text from each cell
            # using OCR on the cell regions
            table_data = []
            for row in rows:
                table_row = []
                for cell in row:
                    x1, y1, x2, y2 = cell
                    cell_img = gray[y1:y2, x1:x2]
                    # In a real implementation, we'd run OCR on the cell image
                    # text = run_ocr(cell_img)
                    # For now, just add a placeholder
                    table_row.append("Cell data")
                table_data.append(table_row)
            
            return table_data
            
        except Exception as e:
            logger.error(f"Error extracting table from image: {e}")
            return []
