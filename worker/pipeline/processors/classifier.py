from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import cv2
from sklearn.ensemble import IsolationForest
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import time
import uuid

from ...types import Document, Page, Region
from ...utils.logging import get_logger, log_execution_time

logger = get_logger(__name__)

class DocumentClassifier:
    """
    Classifies documents by type (invoice, receipt, form, etc.) and renderer source
    (digital, scanned, image-based, etc.) to optimize downstream processing.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.model_path = self.config.get("model_path")
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
        self._model = None
        self._class_names = [
            "invoice", "receipt", "form", "contract", "letter", 
            "report", "resume", "scientific", "presentation", "other"
        ]
        self._renderer_types = ["digital", "scanned", "photo", "mixed", "unknown"]
        
        # Initialize the model if configured
        if self.model_path:
            self._initialize_model()
        
    @log_execution_time
    def process(self, document: Document) -> Document:
        """
        Process a document to classify its type and renderer.
        
        Args:
            document: Document object with pages and images
            
        Returns:
            Document with classification metadata
        """
        logger.info(f"Classifying document with {len(document.pages)} pages")
        
        try:
            # Get first page for classification
            first_page = document.pages[0] if document.pages else None
            if first_page is None or first_page.image is None:
                logger.warning(f"No image available for classification")
                document.metadata["document_type"] = "unknown"
                document.metadata["render_type"] = "unknown"
                document.metadata["classification_confidence"] = 0.0
                return document
            
            # Extract features and classify
            doc_type, render_type, confidence = self._classify_document(first_page.image)
            
            # Update document metadata
            document.metadata["document_type"] = doc_type
            document.metadata["render_type"] = render_type
            document.metadata["classification_confidence"] = confidence
            
            logger.info(f"Classified document as {doc_type} ({render_type}) with confidence {confidence:.2f}")
        except Exception as e:
            logger.error(f"Error classifying document: {str(e)}")
            document.metadata["document_type"] = "unknown"
            document.metadata["render_type"] = "unknown"
            document.metadata["classification_confidence"] = 0.0
        
        return document
    
    def _initialize_model(self):
        """Initialize the document classification model"""
        try:
            # Use TensorFlow for document classification
            if tf.config.list_physical_devices('GPU'):
                logger.info("GPU available for document classification")
            else:
                logger.info("Using CPU for document classification")
                
            # For now, create a simple feature extractor based on MobileNetV2
            # In a production system, this would be replaced with a fine-tuned model
            base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            x = GlobalAveragePooling2D()(base_model.output)
            output = Dense(len(self._class_names) + len(self._renderer_types), activation='softmax')(x)
            self._model = Model(inputs=base_model.input, outputs=output)
            
            logger.info("Document classification model initialized")
        except Exception as e:
            logger.error(f"Error initializing classification model: {str(e)}")
            self._model = None
    
    def _classify_document(self, image: Any) -> Tuple[str, str, float]:
        """Classify document type and renderer from image"""
        # Use a rule-based approach for now
        # In production, this would use the trained model
        
        # Convert image to numpy array if it's a file path
        if isinstance(image, str):
            try:
                image = cv2.imread(image)
            except Exception as e:
                logger.error(f"Error reading image: {str(e)}")
                return "unknown", "unknown", 0.0
        
        if image is None:
            return "unknown", "unknown", 0.0
            
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Extract basic features for classification
        height, width = gray.shape[:2]
        aspect_ratio = width / height
        
        # Calculate basic image statistics
        mean = np.mean(gray)
        std = np.std(gray)
        
        # Detect if document is digital or scanned based on background uniformity
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / (height * width)
        
        # Get histogram for background analysis
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()
        background_peaks = len([i for i in range(1, 255) if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > 0.01])
        
        # Classify renderer type
        if std < 50 and background_peaks < 3:
            render_type = "digital"
            render_confidence = 0.85
        elif edge_density > 0.1:
            render_type = "photo"
            render_confidence = 0.7
        else:
            render_type = "scanned"
            render_confidence = 0.75
            
        # Use model for document type if available
        if self._model is not None:
            try:
                # Preprocess image for model
                img_resized = cv2.resize(image, (224, 224))
                img_array = preprocess_input(img_resized)
                img_batch = np.expand_dims(img_array, axis=0)
                
                # Get predictions
                predictions = self._model.predict(img_batch)[0]
                
                # Split predictions for document type and renderer
                doc_type_pred = predictions[:len(self._class_names)]
                renderer_pred = predictions[len(self._class_names):]
                
                # Get top class and confidence
                doc_type_idx = np.argmax(doc_type_pred)
                doc_type = self._class_names[doc_type_idx]
                doc_confidence = float(doc_type_pred[doc_type_idx])
                
                # Also check renderer predictions from model
                render_type_idx = np.argmax(renderer_pred)
                model_render_type = self._renderer_types[render_type_idx]
                model_render_confidence = float(renderer_pred[render_type_idx])
                
                # Use model renderer type if confidence is higher
                if model_render_confidence > render_confidence:
                    render_type = model_render_type
                    render_confidence = model_render_confidence
                    
                return doc_type, render_type, (doc_confidence + render_confidence) / 2
                
            except Exception as e:
                logger.error(f"Error during model prediction: {str(e)}")
        
        # Fallback to rule-based type classification
        if aspect_ratio < 1.1:
            doc_type = "form"
        elif aspect_ratio > 2.5:
            doc_type = "receipt"
        elif edge_density < 0.02:
            doc_type = "letter"
        else:
            doc_type = "invoice"
            
        return doc_type, render_type, 0.65
        
    def _detect_anomalies(self, image: np.ndarray) -> bool:
        """Detect anomalies in document images using Isolation Forest"""
        try:
            # Extract features from image
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
                
            # Resize for consistent feature extraction
            img_small = cv2.resize(gray, (32, 32))
            
            # Extract simple features
            features = []
            features.append(np.mean(img_small))
            features.append(np.std(img_small))
            
            # Add histogram features
            hist = cv2.calcHist([img_small], [0], None, [8], [0, 256])
            hist = hist.flatten() / hist.sum()
            features.extend(hist)
            
            # Create isolation forest
            clf = IsolationForest(contamination=0.1, random_state=42)
            anomaly = clf.fit_predict([features])[0]
            
            return anomaly == -1  # -1 indicates anomaly
        except Exception as e:
            logger.error(f"Error in anomaly detection: {str(e)}")
            return False
