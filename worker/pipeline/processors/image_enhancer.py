from typing import Dict, Any, Optional, List, Union, Tuple
import numpy as np
import cv2
from scipy import stats
from scipy.ndimage import binary_fill_holes, gaussian_filter
from sklearn.cluster import DBSCAN
from skimage.transform import radon, rotate
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.util import img_as_ubyte, img_as_float
import torch
from PIL import Image
from io import BytesIO
import base64

from ...types import Document, Page
from ...utils.logging import get_logger, log_execution_time

logger = get_logger(__name__)

class ImageEnhancer:
    """
    Advanced image enhancement processor for improving document quality before OCR.
    Includes methods for denoising, deskewing, contrast adjustment, and resolution upscaling.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.deskew_enabled = self.config.get("deskew_enabled", True)
        self.denoise_enabled = self.config.get("denoise_enabled", True)
        self.enhance_contrast = self.config.get("enhance_contrast", True)
        self.upscale_resolution = self.config.get("upscale_resolution", False)
        self.upscale_factor = self.config.get("upscale_factor", 2.0)
        self.target_dpi = self.config.get("target_dpi", 300)
        
        # Load super-resolution model if needed
        self.sr_model = None
        if self.upscale_resolution:
            self._load_sr_model()
    
    @log_execution_time
    def process(self, document: Document) -> Document:
        """
        Enhance document images for better OCR quality.
        
        Args:
            document: Document with page images
            
        Returns:
            Document with enhanced page images
        """
        logger.info(f"Enhancing images for document with {len(document.pages)} pages")
        
        for i, page in enumerate(document.pages):
            if not page.image:
                logger.warning(f"Page {page.page_num} has no image data, skipping enhancement")
                continue
            
            logger.debug(f"Enhancing image for page {page.page_num}")
            
            try:
                # Convert base64 to image if needed
                img = self._get_image_from_page(page)
                if img is None:
                    continue
                
                # Apply enhancements in sequence
                enhanced_img = img.copy()
                
                if self.deskew_enabled:
                    enhanced_img = self._deskew_image(enhanced_img)
                
                if self.denoise_enabled:
                    enhanced_img = self._denoise_image(enhanced_img)
                
                if self.enhance_contrast:
                    enhanced_img = self._enhance_contrast(enhanced_img)
                    
                if self.upscale_resolution:
                    enhanced_img = self._upscale_image(enhanced_img)
                
                # Update the page with enhanced image
                page.image = self._image_to_base64(enhanced_img)
                document.pages[i] = page
                
            except Exception as e:
                logger.error(f"Error enhancing image for page {page.page_num}: {str(e)}")
                
        return document
    
    def _get_image_from_page(self, page: Page) -> Optional[np.ndarray]:
        """Convert page image data to numpy array"""
        try:
            if isinstance(page.image, str) and page.image.startswith('data:image/'):
                # Extract base64 part
                base64_data = page.image.split(',')[1]
                img_data = base64.b64decode(base64_data)
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            elif isinstance(page.image, str):
                # Assume direct base64
                img_data = base64.b64decode(page.image)
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            elif isinstance(page.image, np.ndarray):
                # Already numpy array
                img = page.image
            else:
                logger.error(f"Unsupported image format for page {page.page_num}")
                return None
            
            return img
            
        except Exception as e:
            logger.error(f"Error converting page image to numpy array: {str(e)}")
            return None
    
    def _image_to_base64(self, img: np.ndarray) -> str:
        """Convert numpy array image to base64 string"""
        _, buffer = cv2.imencode('.png', img)
        img_str = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/png;base64,{img_str}"
    
    def _deskew_image(self, img: np.ndarray) -> np.ndarray:
        """Deskew image using multi-method approach for highest accuracy"""
        try:
            # Convert to grayscale if needed
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
            
            # Enhance edges to improve deskew performance
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Try Hough Transform method first
            hough_angle = self._get_skew_angle_hough(edges)
            
            # If Hough method failed (returned None), use Radon transform
            if hough_angle is None:
                # Apply threshold to get binary image
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                
                # Apply morphology to enhance text lines
                kernel = np.ones((5, 1), np.uint8)  # Horizontal kernel
                binary = cv2.dilate(binary, kernel, iterations=1)
                
                # Calculate skew angle using Radon transform
                theta = np.linspace(-10., 10., 100, endpoint=False)  # Narrower search range for precision
                sinogram = radon(binary, theta=theta)
                r = np.array([np.sum(sinogram[:, i]**2) for i in range(len(theta))])
                rotation = theta[np.argmax(r)]
                
                # If the angle is near 0, it's likely not skewed
                if abs(rotation) < 0.5:
                    return img
                
                skew_angle = -rotation  # Negative because we need to rotate in the opposite direction
            else:
                skew_angle = hough_angle
            
            # Only correct if the skew is significant but not too extreme
            if abs(skew_angle) > 0.5 and abs(skew_angle) < 20:
                # Use OpenCV for better rotation performance
                height, width = img.shape[:2]
                center = (width // 2, height // 2)
                
                # Get rotation matrix
                rotation_matrix = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
                
                # Calculate new image dimensions
                abs_cos = abs(rotation_matrix[0, 0])
                abs_sin = abs(rotation_matrix[0, 1])
                new_w = int(height * abs_sin + width * abs_cos)
                new_h = int(height * abs_cos + width * abs_sin)
                
                # Adjust rotation matrix for translation
                rotation_matrix[0, 2] += (new_w / 2) - center[0]
                rotation_matrix[1, 2] += (new_h / 2) - center[1]
                
                # Apply rotation with border replication to avoid black borders
                rotated = cv2.warpAffine(
                    img, 
                    rotation_matrix, 
                    (new_w, new_h), 
                    borderMode=cv2.BORDER_REPLICATE
                )
                
                return rotated
            else:
                return img
                
        except Exception as e:
            logger.warning(f"Error deskewing image: {str(e)}")
            return img
            
    def _get_skew_angle_hough(self, edges: np.ndarray) -> Optional[float]:
        """Get skew angle using Hough line transform"""
        try:
            # Apply Hough transform to detect lines
            lines = cv2.HoughLinesP(
                edges, 
                1, 
                np.pi/180, 
                threshold=100, 
                minLineLength=100, 
                maxLineGap=10
            )
            
            if lines is None or len(lines) < 5:
                return None
            
            # Calculate angles of detected lines
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Filter out near-vertical lines
                if abs(x2 - x1) > abs(y2 - y1):  # Horizontal-ish line
                    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                    # Normalize angle
                    while angle < -45:
                        angle += 90
                    while angle > 45:
                        angle -= 90
                    angles.append(angle)
            
            if not angles:
                return None
            
            # Use median for robustness against outliers
            skew_angle = np.median(angles)
            
            return skew_angle
            
        except Exception as e:
            logger.warning(f"Error in Hough skew detection: {str(e)}")
            return None
    
    def _denoise_image(self, img: np.ndarray) -> np.ndarray:
        """Denoise image using advanced filtering techniques"""
        try:
            # Analyze image characteristics to select optimal denoising method
            image_variance = np.var(img)
            
            # Choose denoising method based on image characteristics
            if image_variance < 100:  # Low variance indicates low noise or flat areas
                # For low-noise images, use gentle denoising
                if len(img.shape) == 3:
                    # Color image - use fastNlMeans with light settings
                    denoised = cv2.fastNlMeansDenoisingColored(
                        img, 
                        None, 
                        h=5,  # Lower filter strength
                        hColor=5, 
                        templateWindowSize=7, 
                        searchWindowSize=21
                    )
                else:
                    # Grayscale image - use fastNlMeans with light settings
                    denoised = cv2.fastNlMeansDenoising(
                        img, 
                        None, 
                        h=7, 
                        templateWindowSize=7, 
                        searchWindowSize=21
                    )
            else:
                # For noisy images, try more aggressive approach with scikit-image's non-local means
                if len(img.shape) == 3:
                    # Convert to float for scikit-image
                    img_float = img_as_float(img)
                    
                    # Process each channel
                    denoised_channels = []
                    for channel in range(3):
                        # Estimate noise standard deviation
                        sigma_est = np.mean(estimate_sigma(img_float[:, :, channel]))
                        
                        # Apply non-local means denoising
                        denoised_channel = denoise_nl_means(
                            img_float[:, :, channel],
                            h=0.8 * sigma_est,
                            fast_mode=True,
                            patch_size=5,
                            patch_distance=7
                        )
                        denoised_channels.append(denoised_channel)
                        
                    # Combine channels and convert back to uint8
                    denoised_float = np.stack(denoised_channels, axis=2)
                    denoised = img_as_ubyte(denoised_float)
                    
                    # Further reduce remaining noise with bilateral filter
                    denoised = cv2.bilateralFilter(denoised, 5, 50, 50)
                else:
                    # Grayscale - use combination of non-local means and bilateral
                    # Estimate noise
                    sigma_est = np.mean(estimate_sigma(img))
                    
                    # Apply bilateral filter to preserve edges while removing noise
                    denoised = cv2.bilateralFilter(img, 5, 75, 75)
                    
                    # Apply additional non-local means denoising if needed
                    if sigma_est > 0.05:  # Higher noise level
                        denoised = cv2.fastNlMeansDenoising(
                            denoised, 
                            None, 
                            h=10, 
                            templateWindowSize=7, 
                            searchWindowSize=21
                        )
            
            # Apply subtle Gaussian smoothing to remove any remaining noise
            denoised = cv2.GaussianBlur(denoised, (3, 3), 0.5)
                
            return denoised
            
        except Exception as e:
            logger.warning(f"Error denoising image: {str(e)}")
            return img
    
    def _enhance_contrast(self, img: np.ndarray) -> np.ndarray:
        """Enhance image contrast using adaptive methods with document-specific optimization"""
        try:
            # Analyze image histogram to determine contrast enhancement strategy
            if len(img.shape) == 3:
                # Calculate histogram for grayscale equivalent to assess contrast
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                hist_normalized = hist / (img.shape[0] * img.shape[1])
            else:
                # Already grayscale
                gray = img
                hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                hist_normalized = hist / (img.shape[0] * img.shape[1])
            
            # Calculate histogram spread (standard deviation)
            hist_std = np.std(hist_normalized)
            
            # Calculate percentage of dark and light pixels
            dark_pixel_ratio = np.sum(hist_normalized[:50]) / np.sum(hist_normalized)
            light_pixel_ratio = np.sum(hist_normalized[200:]) / np.sum(hist_normalized)
            
            # Choose enhancement strategy based on image characteristics
            if hist_std < 0.02:  # Low contrast image
                # Apply stronger contrast enhancement
                if len(img.shape) == 3:
                    # Convert to LAB color space
                    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    
                    # Apply CLAHE with stronger settings
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                    l = clahe.apply(l)
                    
                    # Merge and convert back
                    enhanced_lab = cv2.merge((l, a, b))
                    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
                    
                    # Additional contrast stretch if very low contrast
                    if hist_std < 0.01:
                        enhanced_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
                        min_val, max_val = np.percentile(enhanced_gray, [2, 98])
                        for i in range(3):
                            enhanced[:,:,i] = np.clip(255 * (enhanced[:,:,i] - min_val) / (max_val - min_val), 0, 255).astype(np.uint8)
                else:
                    # Strong CLAHE for grayscale
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                    enhanced = clahe.apply(img)
                    
                    # Additional contrast stretch
                    min_val, max_val = np.percentile(enhanced, [2, 98])
                    enhanced = np.clip(255 * (enhanced - min_val) / (max_val - min_val), 0, 255).astype(np.uint8)
                    
            elif dark_pixel_ratio > 0.6:  # Dark document
                # Brighten dark areas while preserving light areas
                if len(img.shape) == 3:
                    # Use gamma correction to brighten dark areas
                    gamma = 1.5
                    lookUpTable = np.empty((1,256), np.uint8)
                    for i in range(256):
                        lookUpTable[0,i] = np.clip(pow(i / 255.0, 1.0 / gamma) * 255.0, 0, 255)
                    
                    # Apply gamma correction
                    brightened = cv2.LUT(img, lookUpTable)
                    
                    # Then apply gentle CLAHE in LAB space
                    lab = cv2.cvtColor(brightened, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    l = clahe.apply(l)
                    enhanced_lab = cv2.merge((l, a, b))
                    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
                else:
                    # Gamma correction for grayscale
                    gamma = 1.5
                    lookUpTable = np.empty((1,256), np.uint8)
                    for i in range(256):
                        lookUpTable[0,i] = np.clip(pow(i / 255.0, 1.0 / gamma) * 255.0, 0, 255)
                    brightened = cv2.LUT(img, lookUpTable)
                    
                    # Apply gentle CLAHE
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    enhanced = clahe.apply(brightened)
                    
            elif light_pixel_ratio > 0.6:  # Light document with faint text
                # Enhance light text while preserving dark areas
                if len(img.shape) == 3:
                    # Convert to LAB
                    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    
                    # Apply adaptive thresholding to L channel
                    l_enhanced = cv2.adaptiveThreshold(
                        l, 
                        255, 
                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY,
                        11,
                        2
                    )
                    
                    # Blend original and enhanced L channel
                    l = cv2.addWeighted(l, 0.6, l_enhanced, 0.4, 0)
                    
                    # Apply CLAHE for additional enhancement
                    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
                    l = clahe.apply(l)
                    
                    # Merge and convert back
                    enhanced_lab = cv2.merge((l, a, b))
                    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
                else:
                    # Apply adaptive thresholding
                    enhanced_binary = cv2.adaptiveThreshold(
                        img, 
                        255, 
                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY,
                        11,
                        2
                    )
                    
                    # Blend original and thresholded image
                    enhanced = cv2.addWeighted(img, 0.6, enhanced_binary, 0.4, 0)
                    
                    # Apply CLAHE for additional enhancement
                    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
                    enhanced = clahe.apply(enhanced)
            else:
                # Standard document - use regular CLAHE
                if len(img.shape) == 3:
                    # Convert to LAB
                    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    
                    # Apply CLAHE to L channel
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    l = clahe.apply(l)
                    
                    # Merge and convert back
                    enhanced_lab = cv2.merge((l, a, b))
                    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
                else:
                    # For grayscale images
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    enhanced = clahe.apply(img)
            
            return enhanced
            
        except Exception as e:
            logger.warning(f"Error enhancing contrast: {str(e)}")
            return img
    
    def _upscale_image(self, img: np.ndarray) -> np.ndarray:
        """Upscale image resolution"""
        try:
            if self.sr_model is not None:
                # Use super-resolution model
                return self._apply_sr_model(img)
            else:
                # Use bicubic interpolation as fallback
                h, w = img.shape[:2]
                new_h, new_w = int(h * self.upscale_factor), int(w * self.upscale_factor)
                upscaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                return upscaled
                
        except Exception as e:
            logger.warning(f"Error upscaling image: {str(e)}")
            return img
    
    def _load_sr_model(self):
        """Load super-resolution model"""
        try:
            # In a production environment, this would load a proper super-resolution model
            # For demonstration, we're not actually loading a model
            # but in production you might use something like:
            # self.sr_model = torch.hub.load('xinntao/ESRGAN', 'RRDBNet_PSNR')
            logger.info("Super-resolution model would be loaded here in production")
        except Exception as e:
            logger.error(f"Error loading super-resolution model: {str(e)}")
    
    def _apply_sr_model(self, img: np.ndarray) -> np.ndarray:
        """Apply super-resolution model to image"""
        # In production, this would use the actual model
        # For demonstration, we just use bicubic upscaling
        h, w = img.shape[:2]
        new_h, new_w = int(h * self.upscale_factor), int(w * self.upscale_factor)
        upscaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        return upscaled
