from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from scipy.ndimage import binary_dilation, binary_erosion, binary_fill_holes
import torch
import math
from dataclasses import dataclass

from ...types import Document, Page
from ...utils.logging import get_logger, log_execution_time

logger = get_logger(__name__)

@dataclass
class BinarizationParams:
    """Parameters for document binarization"""
    adaptive_method: int = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    block_size: int = 11
    constant: int = 2
    invert: bool = True
    

class DocumentBinarizer:
    """
    Advanced document binarization that applies optimal thresholding techniques
    to separate foreground from background in document images.
    
    This is a critical preprocessing step that significantly affects OCR accuracy.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Configure methods
        self.method = self.config.get("method", "adaptive")  # 'adaptive', 'otsu', 'sauvola', 'wolf', 'niblack'
        self.block_size = self.config.get("block_size", 15)
        self.k_value = self.config.get("k_value", 0.2)  # For Sauvola and Wolf methods
        self.r_value = self.config.get("r_value", 128)  # For Sauvola method
        
        # Noise removal settings
        self.remove_noise = self.config.get("remove_noise", True)
        self.min_contour_area = self.config.get("min_contour_area", 30)
        
        # Morphological operations
        self.apply_morphology = self.config.get("apply_morphology", True)
        self.morph_kernel_size = self.config.get("morph_kernel_size", 3)
        
        # Edge enhancement
        self.enhance_edges = self.config.get("enhance_edges", True)
        self.edge_sigma = self.config.get("edge_sigma", 1.0)
    
    @log_execution_time
    def process(self, document: Document) -> Document:
        """
        Binarize document images for optimal OCR processing
        
        Args:
            document: Document with page images
            
        Returns:
            Document with binarized images
        """
        logger.info(f"Binarizing images for document with {len(document.pages)} pages")
        
        for i, page in enumerate(document.pages):
            if not page.image:
                logger.warning(f"Page {page.page_num} has no image data, skipping binarization")
                continue
                
            logger.debug(f"Binarizing image for page {page.page_num}")
            
            try:
                # Get image as numpy array
                img = self._get_image_from_page(page)
                if img is None:
                    continue
                
                # Convert to grayscale if needed
                if len(img.shape) == 3:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    gray = img.copy()
                
                # Enhance edges if configured
                if self.enhance_edges:
                    gray = self._enhance_image_edges(gray)
                
                # Apply binarization based on method
                binary = self._apply_binarization(gray)
                
                # Apply morphological operations if configured
                if self.apply_morphology:
                    binary = self._apply_morphology(binary)
                
                # Remove noise if configured
                if self.remove_noise:
                    binary = self._remove_noise(binary)
                
                # Store binarized image in page metadata for downstream processors
                # Keep original image in page.image
                if not page.metadata:
                    page.metadata = {}
                    
                page.metadata["binarized_image"] = self._binary_to_base64(binary)
                document.pages[i] = page
                
            except Exception as e:
                logger.error(f"Error binarizing image for page {page.page_num}: {str(e)}")
        
        return document
    
    def _get_image_from_page(self, page: Page) -> Optional[np.ndarray]:
        """Convert page image to numpy array"""
        # Implementation would be similar to ImageEnhancer._get_image_from_page
        # This would handle base64 encoded images and numpy arrays
        # For brevity, assuming implementation exists
        return np.array([])  # Placeholder
    
    def _binary_to_base64(self, binary_img: np.ndarray) -> str:
        """Convert binary image to base64 string"""
        # Implementation would be similar to ImageEnhancer._image_to_base64
        # For brevity, assuming implementation exists
        return ""  # Placeholder
    
    def _apply_binarization(self, gray_img: np.ndarray) -> np.ndarray:
        """Apply selected binarization method"""
        h, w = gray_img.shape
        
        if self.method == "adaptive":
            # Adaptive thresholding - good for varying illumination
            binary = cv2.adaptiveThreshold(
                gray_img, 
                255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV if self.config.get("invert", True) else cv2.THRESH_BINARY,
                self.block_size if self.block_size % 2 == 1 else self.block_size + 1,  # Must be odd
                self.config.get("constant", 2)
            )
            
        elif self.method == "otsu":
            # Otsu's method - automatically determines optimal threshold
            _, binary = cv2.threshold(
                gray_img, 
                0, 
                255, 
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU if self.config.get("invert", True) else cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            
        elif self.method == "sauvola":
            # Sauvola thresholding - better for document images with background variations
            binary = np.zeros((h, w), dtype=np.uint8)
            
            # Pad image to handle border regions
            pad = self.block_size // 2
            padded = np.pad(gray_img, ((pad, pad), (pad, pad)), mode='symmetric')
            
            # Calculate local mean and standard deviation
            integral = cv2.integral(padded)
            integral_sqr = cv2.integral(np.square(padded.astype(np.float32)))
            
            for i in range(h):
                for j in range(w):
                    # Get local window coordinates
                    y1, x1 = i, j
                    y2, x2 = i + self.block_size, j + self.block_size
                    
                    # Calculate local mean and standard deviation
                    count = (y2 - y1) * (x2 - x1)
                    sum_val = integral[y2, x2] - integral[y2, x1] - integral[y1, x2] + integral[y1, x1]
                    sum_sqr = integral_sqr[y2, x2] - integral_sqr[y2, x1] - integral_sqr[y1, x2] + integral_sqr[y1, x1]
                    
                    mean = sum_val / count
                    variance = (sum_sqr / count) - (mean ** 2)
                    std = math.sqrt(variance) if variance > 0 else 0
                    
                    # Sauvola threshold formula
                    threshold = mean * (1 + self.k_value * ((std / self.r_value) - 1))
                    
                    # Apply threshold
                    if self.config.get("invert", True):
                        binary[i, j] = 255 if gray_img[i, j] < threshold else 0
                    else:
                        binary[i, j] = 0 if gray_img[i, j] < threshold else 255
            
        else:
            # Default to Otsu's method if unsupported method specified
            logger.warning(f"Unsupported binarization method: {self.method}, falling back to Otsu")
            _, binary = cv2.threshold(
                gray_img, 
                0, 
                255, 
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU if self.config.get("invert", True) else cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        
        return binary
    
    def _enhance_image_edges(self, gray_img: np.ndarray) -> np.ndarray:
        """Enhance edges in grayscale image"""
        # Apply unsharp masking for edge enhancement
        blurred = cv2.GaussianBlur(gray_img, (0, 0), self.edge_sigma)
        enhanced = cv2.addWeighted(gray_img, 1.5, blurred, -0.5, 0)
        return enhanced
    
    def _apply_morphology(self, binary_img: np.ndarray) -> np.ndarray:
        """Apply morphological operations to improve binary image"""
        # Create kernel
        kernel = np.ones((self.morph_kernel_size, self.morph_kernel_size), np.uint8)
        
        # Apply closing operation (dilation followed by erosion)
        # This helps to close small holes in the foreground
        binary = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
        
        # Remove small isolated pixels in background
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        return binary
    
    def _remove_noise(self, binary_img: np.ndarray) -> np.ndarray:
        """Remove noise from binary image"""
        # Find all contours
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create mask with only contours above threshold size
        mask = np.zeros_like(binary_img)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_contour_area:
                cv2.drawContours(mask, [contour], -1, 255, -1)
        
        # Apply mask to original binary image
        cleaned = cv2.bitwise_and(binary_img, mask)
        
        return cleaned
