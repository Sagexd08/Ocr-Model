from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import cv2
from PIL import Image
import math
from skimage.filters import threshold_otsu, threshold_local
from skimage.transform import hough_line, hough_line_peaks
import os
import uuid

from ...types import Document, Page
from ...utils.logging import get_logger, log_execution_time

logger = get_logger(__name__)

class PreprocessingProcessor:
    """
    Performs document image preprocessing to enhance OCR quality.
    Includes operations like deskewing, denoising, contrast enhancement, etc.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.max_image_dim = self.config.get("max_image_dim", 3000)
        self.contrast_adjustment = self.config.get("contrast_adjustment", 1.3)
        self.denoise_strength = self.config.get("denoise_strength", 10)
        self.auto_deskew = self.config.get("auto_deskew", True)
        self.adaptive_threshold = self.config.get("adaptive_threshold", True)
        self.target_dpi = self.config.get("target_dpi", 300)
        
    @log_execution_time
    def process(self, document: Document, file_data: bytes) -> Document:
        """
        Preprocess document images to enhance OCR quality.
        
        Args:
            document: Document object with pages and images
            file_data: Raw file data for document creation
            
        Returns:
            Document with preprocessed images
        """
        logger.info(f"Preprocessing document with {len(document.pages)} pages")
        
        # Process each page
        for i, page in enumerate(document.pages):
            logger.debug(f"Preprocessing page {page.page_num}")
            
            try:
                # Get page image
                image = page.image
                if image is None:
                    logger.warning(f"No image found for page {page.page_num}, skipping preprocessing")
                    continue
                
                # Load image if it's a file path
                if isinstance(image, str):
                    image = cv2.imread(image)
                
                if image is None:
                    logger.warning(f"Failed to load image for page {page.page_num}")
                    continue
                
                # Store original dimensions
                orig_h, orig_w = image.shape[:2]
                
                # Apply preprocessing pipeline
                preprocessed = self._preprocess_image(image)
                
                # Update the page with preprocessed image
                page.image = preprocessed
                document.pages[i] = page
                
                logger.info(f"Preprocessed page {page.page_num}")
                
            except Exception as e:
                logger.error(f"Error preprocessing page {page.page_num}: {str(e)}")
        
        return document
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Apply image preprocessing pipeline to enhance OCR quality"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Resize if needed
        h, w = gray.shape[:2]
        if max(h, w) > self.max_image_dim:
            scale = self.max_image_dim / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Apply denoising
        if self.denoise_strength > 0:
            denoised = cv2.fastNlMeansDenoising(gray, None, h=self.denoise_strength)
        else:
            denoised = gray
        
        # Deskew if needed
        if self.auto_deskew:
            deskewed = self._deskew_image(denoised)
        else:
            deskewed = denoised
        
        # Apply contrast enhancement
        if self.contrast_adjustment != 1.0:
            enhanced = self._enhance_contrast(deskewed, self.contrast_adjustment)
        else:
            enhanced = deskewed
        
        # Apply adaptive thresholding if configured
        if self.adaptive_threshold:
            # Use adaptive thresholding for better handling of uneven lighting
            binary = cv2.adaptiveThreshold(
                enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Apply morphological operations to clean up the image
            kernel = np.ones((2, 2), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # Return the binary image for OCR
            return binary
        
        # Return the enhanced grayscale image
        return enhanced
    
    def _deskew_image(self, image: np.ndarray, max_angle: float = 10.0) -> np.ndarray:
        """Detect and correct skew in document image"""
        try:
            # Create binary image for line detection
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Use Hough transform to detect lines
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            
            # Detect lines with Hough transform
            tested_angles = np.deg2rad(np.linspace(-max_angle, max_angle, 361))
            hough_space = hough_line(binary, theta=tested_angles)
            _, angles, _ = hough_line_peaks(hough_space, tested_angles, min_distance=20, min_angle=1,
                                         threshold=0.5 * np.max(hough_space))
            
            if len(angles) == 0:
                return image  # No clear lines detected
            
            # Get dominant angle (convert to degrees)
            skew_angle = np.rad2deg(angles[0] - np.pi/2)
            
            # Only correct if skew is significant
            if abs(skew_angle) < 0.5:
                return image
                
            # Limit correction angle
            skew_angle = max(-max_angle, min(max_angle, skew_angle))
            
            # Create rotation matrix
            M = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
            
            # Apply rotation
            rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, 
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=255)
            
            logger.debug(f"Deskewed image by {skew_angle:.2f} degrees")
            return rotated
            
        except Exception as e:
            logger.warning(f"Error during deskew: {str(e)}")
            return image
    
    def _enhance_contrast(self, image: np.ndarray, factor: float = 1.5) -> np.ndarray:
        """Enhance contrast in document image"""
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        
        return enhanced
