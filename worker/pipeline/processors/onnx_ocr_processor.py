"""
ONNX-accelerated OCR Processor for optimized performance.
This module provides OCR functionality using ONNX runtime for improved 
inference speed across platforms.
"""

import os
import cv2
import numpy as np
import onnxruntime as ort
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

class ONNXOCRProcessor:
    """
    OCR processor that uses ONNX runtime for optimized performance.
    Supports text detection, recognition and post-processing.
    """
    
    def __init__(self, 
                 detection_model_path: str,
                 recognition_model_path: str,
                 providers: Optional[List[str]] = None):
        """
        Initialize the ONNX OCR Processor with detection and recognition models.
        
        Args:
            detection_model_path: Path to the ONNX text detection model
            recognition_model_path: Path to the ONNX text recognition model
            providers: ONNX runtime execution providers (CPU/GPU/etc)
        """
        self.detection_model_path = detection_model_path
        self.recognition_model_path = recognition_model_path
        
        # Default to CPU if no providers specified
        self.providers = providers or ['CPUExecutionProvider']
        
        # Initialize ONNX Runtime sessions
        self.detection_session = ort.InferenceSession(
            detection_model_path, 
            providers=self.providers
        )
        
        self.recognition_session = ort.InferenceSession(
            recognition_model_path,
            providers=self.providers
        )
        
        # Get model metadata
        self.detection_inputs = [input.name for input in self.detection_session.get_inputs()]
        self.detection_outputs = [output.name for output in self.detection_session.get_outputs()]
        
        self.recognition_inputs = [input.name for input in self.recognition_session.get_inputs()]
        self.recognition_outputs = [output.name for output in self.recognition_session.get_outputs()]
        
        # Load character dictionary for recognition
        self.character_dict = self._load_character_dict()
    
    def _load_character_dict(self) -> Dict[int, str]:
        """
        Load character dictionary for recognition model.
        Returns:
            Dictionary mapping indices to characters
        """
        # Default dictionary path relative to models
        dict_path = os.path.join(
            os.path.dirname(self.recognition_model_path),
            'dict.txt'
        )
        
        if not os.path.exists(dict_path):
            # Fallback to common ASCII and extended characters
            chars = ''.join([chr(i) for i in range(32, 127)])
            return {i: c for i, c in enumerate(chars)}
        
        # Load from file
        with open(dict_path, 'r', encoding='utf-8') as f:
            characters = [line.strip() for line in f]
            return {i: c for i, c in enumerate(characters)}
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for OCR detection.
        
        Args:
            image: Input image in BGR format (OpenCV)
            
        Returns:
            Preprocessed image ready for detection model
        """
        # Resize to detection model input size
        resized = cv2.resize(image, (640, 640))
        
        # Normalize to 0-1 range
        normalized = resized.astype(np.float32) / 255.0
        
        # Channel ordering (HWC -> CHW)
        transposed = np.transpose(normalized, (2, 0, 1))
        
        # Add batch dimension
        return np.expand_dims(transposed, axis=0)
    
    def detect_text_regions(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Detect text regions in the input image.
        
        Args:
            image: Input image in BGR format (OpenCV)
            
        Returns:
            List of text region bounding boxes
        """
        # Preprocess image
        input_tensor = self.preprocess_image(image)
        
        # Run detection
        outputs = self.detection_session.run(
            self.detection_outputs,
            {self.detection_inputs[0]: input_tensor}
        )
        
        # Post-process detection results
        # This will depend on the specific detection model output format
        boxes = self._post_process_detection(outputs, image.shape)
        
        return boxes
    
    def _post_process_detection(self, 
                               outputs: List[np.ndarray], 
                               original_shape: Tuple[int, int, int]) -> List[np.ndarray]:
        """
        Post-process detection model outputs to get bounding boxes.
        
        Args:
            outputs: Detection model outputs
            original_shape: Original image shape (H, W, C)
            
        Returns:
            List of detected bounding boxes
        """
        # Implementation depends on the specific detection model
        # This is a simplified example
        h, w = original_shape[:2]
        scale_h, scale_w = h / 640, w / 640
        
        # Assuming outputs[0] contains the bounding boxes
        # Format: [batch_id, x0, y0, x1, y1, x2, y2, x3, y3, confidence]
        boxes = []
        
        # Threshold for confidence
        confidence_threshold = 0.5
        
        detections = outputs[0]
        for detection in detections:
            confidence = detection[-1]
            if confidence < confidence_threshold:
                continue
                
            # Extract coordinates and rescale to original image
            coords = detection[1:-1].reshape(-1, 2)
            coords[:, 0] *= scale_w
            coords[:, 1] *= scale_h
            
            boxes.append(coords.astype(np.int32))
        
        return boxes
    
    def recognize_text(self, image: np.ndarray, boxes: List[np.ndarray]) -> List[Dict[str, Any]]:
        """
        Recognize text in detected regions.
        
        Args:
            image: Original input image
            boxes: List of bounding boxes for text regions
            
        Returns:
            List of dictionaries with recognized text and coordinates
        """
        results = []
        
        for box in boxes:
            # Extract region of interest (ROI)
            roi = self._extract_roi(image, box)
            
            if roi is None:
                continue
                
            # Preprocess ROI for recognition
            roi_tensor = self._preprocess_roi(roi)
            
            # Run recognition
            recognition_outputs = self.recognition_session.run(
                self.recognition_outputs,
                {self.recognition_inputs[0]: roi_tensor}
            )
            
            # Decode recognition results
            text, confidence = self._decode_recognition_result(recognition_outputs)
            
            # Add to results
            results.append({
                'text': text,
                'confidence': confidence,
                'box': box.tolist()
            })
            
        return results
    
    def _extract_roi(self, image: np.ndarray, box: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract region of interest based on box coordinates.
        
        Args:
            image: Original input image
            box: Bounding box coordinates
            
        Returns:
            Extracted region or None if extraction failed
        """
        try:
            # Create a rectangular ROI from possibly quad coordinates
            rect = cv2.boundingRect(box)
            x, y, w, h = rect
            
            # Add small padding
            x = max(0, x - 5)
            y = max(0, y - 5)
            w = min(image.shape[1] - x, w + 10)
            h = min(image.shape[0] - y, h + 10)
            
            return image[y:y+h, x:x+w]
        except Exception:
            return None
    
    def _preprocess_roi(self, roi: np.ndarray) -> np.ndarray:
        """
        Preprocess region of interest for recognition model.
        
        Args:
            roi: Extracted region of interest
            
        Returns:
            Preprocessed ROI tensor
        """
        # Resize to recognition model input size (typically height=32)
        height = 32
        ratio = height / roi.shape[0]
        width = int(roi.shape[1] * ratio)
        resized = cv2.resize(roi, (width, height))
        
        # Convert to grayscale if needed
        if len(resized.shape) == 3:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = resized
            
        # Normalize and expand dimensions
        normalized = gray.astype(np.float32) / 255.0
        
        # Add batch and channel dimensions [1, 1, H, W]
        return np.expand_dims(np.expand_dims(normalized, axis=0), axis=0)
    
    def _decode_recognition_result(self, 
                                  outputs: List[np.ndarray]) -> Tuple[str, float]:
        """
        Decode recognition model output to text.
        
        Args:
            outputs: Recognition model outputs
            
        Returns:
            Tuple of (recognized text, confidence)
        """
        # Implementation depends on the specific recognition model
        # This is a simplified CTC-like decoding example
        
        # Assuming outputs[0] shape is [1, sequence_length, num_characters]
        probs = outputs[0][0]
        
        # Get the most likely character at each position
        indices = np.argmax(probs, axis=1)
        
        # Get confidence scores
        confidences = np.max(probs, axis=1)
        
        # Decode indices to characters
        chars = []
        last_idx = -1
        
        for i, idx in enumerate(indices):
            # Skip repeated characters (CTC decoding)
            if idx != last_idx and idx != 0:  # Assuming 0 is blank/padding
                if idx < len(self.character_dict):
                    chars.append(self.character_dict[idx])
            last_idx = idx
        
        text = ''.join(chars)
        confidence = float(np.mean(confidences))
        
        return text, confidence
    
    def process_image(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Process a single image for text detection and recognition.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with detected text regions and recognized text
        """
        # Load image
        image_path = str(image_path)
        image = cv2.imread(image_path)
        if image is None:
            return {'error': f'Failed to load image: {image_path}'}
            
        # Detect text regions
        boxes = self.detect_text_regions(image)
        
        # Recognize text in each region
        results = self.recognize_text(image, boxes)
        
        # Sort results by vertical position
        results.sort(key=lambda x: x['box'][0][1])  # Sort by y-coordinate
        
        return {
            'image_path': image_path,
            'image_size': {'height': image.shape[0], 'width': image.shape[1]},
            'results': results
        }

    @classmethod
    def available_providers(cls) -> List[str]:
        """Get available ONNX Runtime providers on this system."""
        return ort.get_available_providers()

    @classmethod
    def get_recommended_provider(cls) -> str:
        """Get the recommended provider for the current system."""
        providers = ort.get_available_providers()
        
        # Prefer GPU acceleration if available
        if 'CUDAExecutionProvider' in providers:
            return 'CUDAExecutionProvider'
        elif 'DirectMLExecutionProvider' in providers:
            return 'DirectMLExecutionProvider'  # For Windows with DirectX
        elif 'TensorrtExecutionProvider' in providers:
            return 'TensorrtExecutionProvider'  # For systems with TensorRT
        else:
            return 'CPUExecutionProvider'
