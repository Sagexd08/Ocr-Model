"""
Document Classification processor for identifying document types and categories.
Uses a combination of rule-based features and ML-based classification.
"""

import os
import cv2
import numpy as np
import json
import pickle
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import onnxruntime as ort

class DocumentClassifier:
    """
    Document classifier that identifies document types using visual and textual features.
    Supports both rule-based classification and ML-based approaches.
    """
    
    # Common document types and their identifying keywords
    DOCUMENT_TYPES = {
        "invoice": ["invoice", "bill", "payment", "amount due", "total due", "invoice number", "invoice date"],
        "receipt": ["receipt", "payment received", "amount paid", "thank you", "store", "purchase", "merchant"],
        "contract": ["agreement", "contract", "terms", "conditions", "parties", "signed", "hereby"],
        "resume": ["resume", "cv", "curriculum vitae", "experience", "education", "skills", "references"],
        "letter": ["dear", "sincerely", "regards", "respectfully", "attention", "re:", "subject:"],
        "form": ["form", "please fill", "complete", "signature", "date", "print name"],
        "report": ["report", "analysis", "findings", "conclusion", "summary", "prepared by"],
        "academic": ["abstract", "methodology", "literature review", "references", "et al", "journal", "university"],
        "presentation": ["presentation", "slide", "agenda", "introduction", "overview", "conclusion"],
        "legal": ["legal", "law", "court", "plaintiff", "defendant", "hearing", "attorney", "judge"]
    }
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 feature_extractor_path: Optional[str] = None,
                 use_onnx: bool = False):
        """
        Initialize document classifier.
        
        Args:
            model_path: Path to trained classification model
            feature_extractor_path: Path to feature extractor model
            use_onnx: Whether to use ONNX runtime for ML-based classification
        """
        self.model_path = model_path
        self.feature_extractor_path = feature_extractor_path
        self.use_onnx = use_onnx
        
        self.model = None
        self.vectorizer = None
        self.onnx_session = None
        
        # Load models if paths are provided
        if model_path and os.path.exists(model_path):
            if use_onnx:
                self._load_onnx_model(model_path)
            else:
                self._load_model(model_path)
                
        if feature_extractor_path and os.path.exists(feature_extractor_path):
            self._load_vectorizer(feature_extractor_path)
    
    def _load_model(self, model_path: str) -> None:
        """
        Load the trained classification model.
        
        Args:
            model_path: Path to the model file
        """
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
        except Exception as e:
            print(f"Error loading model: {str(e)}")
    
    def _load_vectorizer(self, vectorizer_path: str) -> None:
        """
        Load the feature extractor/vectorizer.
        
        Args:
            vectorizer_path: Path to the vectorizer file
        """
        try:
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
        except Exception as e:
            print(f"Error loading vectorizer: {str(e)}")
    
    def _load_onnx_model(self, model_path: str) -> None:
        """
        Load an ONNX model for classification.
        
        Args:
            model_path: Path to the ONNX model file
        """
        try:
            self.onnx_session = ort.InferenceSession(
                model_path, 
                providers=['CPUExecutionProvider']
            )
        except Exception as e:
            print(f"Error loading ONNX model: {str(e)}")
    
    def extract_features(self, text: str, image: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Extract features from text and image for classification.
        
        Args:
            text: Document text content
            image: Document image (optional)
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Text length
        features["text_length"] = len(text)
        
        # Word count
        words = text.split()
        features["word_count"] = len(words)
        
        # Presence of specific document type keywords
        for doc_type, keywords in self.DOCUMENT_TYPES.items():
            text_lower = text.lower()
            keyword_count = sum(1 for keyword in keywords if keyword in text_lower)
            features[f"{doc_type}_keyword_count"] = keyword_count
            features[f"{doc_type}_keyword_ratio"] = keyword_count / len(keywords) if keywords else 0
        
        # Check for special patterns
        features["has_date"] = self._contains_date(text)
        features["has_currency"] = self._contains_currency(text)
        features["has_email"] = self._contains_email(text)
        features["has_phone"] = self._contains_phone(text)
        features["has_url"] = self._contains_url(text)
        
        # Image features if available
        if image is not None:
            image_features = self._extract_image_features(image)
            features.update(image_features)
        
        return features
    
    def _contains_date(self, text: str) -> bool:
        """Check if text contains date patterns."""
        import re
        # Simple date patterns (MM/DD/YYYY, DD-MM-YYYY, etc.)
        date_patterns = [
            r'\d{1,2}/\d{1,2}/\d{2,4}',  # MM/DD/YY or MM/DD/YYYY
            r'\d{1,2}-\d{1,2}-\d{2,4}',  # MM-DD-YY or MM-DD-YYYY
            r'\d{1,2}\.\d{1,2}\.\d{2,4}'  # MM.DD.YY or MM.DD.YYYY
        ]
        return any(re.search(pattern, text) for pattern in date_patterns)
    
    def _contains_currency(self, text: str) -> bool:
        """Check if text contains currency patterns."""
        import re
        # Currency patterns ($, €, £, etc. followed by numbers)
        currency_patterns = [
            r'[$€£¥]\s*\d+(?:,\d{3})*(?:\.\d{2})?',  # $100, $1,000.00, etc.
            r'\d+(?:,\d{3})*(?:\.\d{2})?\s*[$€£¥]'   # 100$, 1,000.00$, etc.
        ]
        return any(re.search(pattern, text) for pattern in currency_patterns)
    
    def _contains_email(self, text: str) -> bool:
        """Check if text contains email patterns."""
        import re
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        return bool(re.search(email_pattern, text))
    
    def _contains_phone(self, text: str) -> bool:
        """Check if text contains phone number patterns."""
        import re
        phone_patterns = [
            r'\(\d{3}\)\s*\d{3}-\d{4}',  # (123) 456-7890
            r'\d{3}-\d{3}-\d{4}',        # 123-456-7890
            r'\d{3}\.\d{3}\.\d{4}'       # 123.456.7890
        ]
        return any(re.search(pattern, text) for pattern in phone_patterns)
    
    def _contains_url(self, text: str) -> bool:
        """Check if text contains URL patterns."""
        import re
        url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
        return bool(re.search(url_pattern, text))
    
    def _extract_image_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract features from document image.
        
        Args:
            image: Document image as numpy array
            
        Returns:
            Dictionary of image features
        """
        features = {}
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Image dimensions
        height, width = gray.shape[:2]
        features["aspect_ratio"] = width / height if height > 0 else 0
        
        # Calculate histogram features
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / (height * width)  # Normalize
        
        # Basic histogram statistics
        features["hist_mean"] = np.mean(hist)
        features["hist_std"] = np.std(hist)
        features["hist_median"] = np.median(hist)
        
        # Detect edges and calculate edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.count_nonzero(edges) / (height * width)
        features["edge_density"] = edge_density
        
        # Line detection (useful for forms, tables, etc.)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                              minLineLength=100, maxLineGap=10)
        features["line_count"] = len(lines) if lines is not None else 0
        
        # Check if the document has a colored header or footer
        if len(image.shape) == 3:
            # Check top 10% of the image (potential header)
            header_region = image[:int(height * 0.1), :]
            header_std = np.std(header_region)
            features["header_color_std"] = float(header_std)
            
            # Check bottom 10% of the image (potential footer)
            footer_region = image[int(height * 0.9):, :]
            footer_std = np.std(footer_region)
            features["footer_color_std"] = float(footer_std)
        
        return features
    
    def rule_based_classify(self, text: str) -> Dict[str, float]:
        """
        Use rule-based approach to classify document type.
        
        Args:
            text: Document text content
            
        Returns:
            Dictionary of document types and confidence scores
        """
        text_lower = text.lower()
        scores = {}
        
        # Calculate keyword matching score for each document type
        for doc_type, keywords in self.DOCUMENT_TYPES.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            score = matches / len(keywords) if keywords else 0
            scores[doc_type] = score
        
        # Normalize scores
        total_score = sum(scores.values())
        if total_score > 0:
            scores = {k: v / total_score for k, v in scores.items()}
            
        return scores
    
    def ml_classify(self, text: str, image: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Use machine learning to classify document type.
        
        Args:
            text: Document text content
            image: Document image (optional)
            
        Returns:
            Dictionary of document types and confidence scores
        """
        if self.use_onnx and self.onnx_session:
            return self._onnx_classify(text, image)
            
        if not self.model or not self.vectorizer:
            return {}
            
        try:
            # Transform text using vectorizer
            text_features = self.vectorizer.transform([text])
            
            # Get prediction probabilities
            probs = self.model.predict_proba(text_features)[0]
            
            # Map to class names
            scores = {self.model.classes_[i]: float(p) for i, p in enumerate(probs)}
            
            return scores
        except Exception as e:
            print(f"ML classification error: {str(e)}")
            return {}
    
    def _onnx_classify(self, text: str, image: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Use ONNX model for classification.
        
        Args:
            text: Document text content
            image: Document image (optional)
            
        Returns:
            Dictionary of document types and confidence scores
        """
        if not self.onnx_session or not self.vectorizer:
            return {}
            
        try:
            # Transform text using vectorizer
            text_features = self.vectorizer.transform([text]).astype(np.float32)
            
            # Get model input name
            input_name = self.onnx_session.get_inputs()[0].name
            
            # Run inference
            probs = self.onnx_session.run(None, {input_name: text_features})[0][0]
            
            # Map to class names (need to know class mapping)
            class_names = list(self.DOCUMENT_TYPES.keys())  # Simplified example
            scores = {class_names[i]: float(p) for i, p in enumerate(probs)}
            
            return scores
        except Exception as e:
            print(f"ONNX classification error: {str(e)}")
            return {}
    
    def classify(self, 
                text: str, 
                image: Optional[np.ndarray] = None, 
                method: str = "hybrid") -> Dict[str, Any]:
        """
        Classify a document based on its text content and image.
        
        Args:
            text: Document text content
            image: Document image (optional)
            method: Classification method (rule, ml, or hybrid)
            
        Returns:
            Classification result with document types and confidence scores
        """
        if method == "rule" or (method == "hybrid" and (not self.model or not self.vectorizer)):
            scores = self.rule_based_classify(text)
        elif method == "ml":
            scores = self.ml_classify(text, image)
        else:  # hybrid
            rule_scores = self.rule_based_classify(text)
            ml_scores = self.ml_classify(text, image)
            
            # Combine scores (weighted average)
            scores = {}
            for doc_type in set(rule_scores.keys()) | set(ml_scores.keys()):
                rule_score = rule_scores.get(doc_type, 0)
                ml_score = ml_scores.get(doc_type, 0)
                # Give ML model more weight if available
                scores[doc_type] = 0.3 * rule_score + 0.7 * ml_score if ml_scores else rule_score
        
        # Get top class
        top_class = max(scores.items(), key=lambda x: x[1]) if scores else ("unknown", 0)
        
        # Extract features for additional analysis
        features = self.extract_features(text, image)
        
        return {
            "document_type": top_class[0],
            "confidence": top_class[1],
            "scores": scores,
            "features": features,
            "method": method
        }
    
    def extract_document_metadata(self, text: str) -> Dict[str, Any]:
        """
        Extract structured metadata from document text.
        
        Args:
            text: Document text content
            
        Returns:
            Dictionary with extracted metadata fields
        """
        import re
        
        metadata = {}
        
        # Extract dates
        date_patterns = [
            (r'(?:Date|DATE):\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', "document_date"),
            (r'(?:Date|DATE)\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', "document_date"),
            (r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', "date_found")
        ]
        
        for pattern, field in date_patterns:
            match = re.search(pattern, text)
            if match and field not in metadata:
                metadata[field] = match.group(1)
        
        # Extract invoice/document numbers
        id_patterns = [
            (r'(?:Invoice|INVOICE)(?:\s+(?:no|NO|number|NUMBER|#))?\s*[:#]?\s*([A-Za-z0-9-]+)', "invoice_number"),
            (r'(?:Document|DOC)(?:\s+(?:no|NO|number|NUMBER|#))?\s*[:#]?\s*([A-Za-z0-9-]+)', "document_number"),
            (r'(?:Reference|REF)(?:\s+(?:no|NO|number|NUMBER|#))?\s*[:#]?\s*([A-Za-z0-9-]+)', "reference_number")
        ]
        
        for pattern, field in id_patterns:
            match = re.search(pattern, text)
            if match:
                metadata[field] = match.group(1)
        
        # Extract amounts/totals
        amount_patterns = [
            (r'(?:Total|TOTAL|Amount Due|AMOUNT DUE)[\s:]*[$€£¥]?\s*([\d,]+\.\d{2})', "total_amount"),
            (r'(?:Subtotal|SUBTOTAL)[\s:]*[$€£¥]?\s*([\d,]+\.\d{2})', "subtotal"),
            (r'(?:Tax|TAX|VAT|GST)[\s:]*[$€£¥]?\s*([\d,]+\.\d{2})', "tax_amount")
        ]
        
        for pattern, field in amount_patterns:
            match = re.search(pattern, text)
            if match:
                # Remove commas and convert to float
                value = match.group(1).replace(',', '')
                try:
                    metadata[field] = float(value)
                except ValueError:
                    metadata[field] = match.group(1)
        
        # Extract sender/receiver information
        # This is more complex, but we'll use a simplified approach
        lines = text.split('\n')
        in_address_block = False
        address_lines = []
        
        for line in lines[:20]:  # Check first 20 lines for address
            line = line.strip()
            
            # Look for email pattern
            email_match = re.search(r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', line)
            if email_match:
                metadata["email"] = email_match.group(1)
            
            # Look for phone pattern
            phone_match = re.search(r'(?:Phone|TEL|T)[\s:]*(\+?[\d\s\(\)-]{10,})', line)
            if phone_match:
                metadata["phone"] = phone_match.group(1)
            
            # Try to detect address blocks
            if re.search(r'\b(?:street|ave|avenue|road|rd|boulevard|blvd|lane|ln|drive|dr)\b', 
                        line, re.IGNORECASE):
                in_address_block = True
                address_lines.append(line)
            elif in_address_block and line and len(line) > 5:
                address_lines.append(line)
            elif in_address_block:
                in_address_block = False
        
        if address_lines:
            metadata["address"] = "\n".join(address_lines)
        
        return metadata
    
    def train(self, 
             texts: List[str], 
             labels: List[str], 
             vectorizer_params: Optional[Dict[str, Any]] = None) -> None:
        """
        Train the document classifier on labeled examples.
        
        Args:
            texts: List of document texts
            labels: Corresponding document type labels
            vectorizer_params: Parameters for TF-IDF vectorizer
        """
        if not texts or not labels or len(texts) != len(labels):
            raise ValueError("Invalid training data")
        
        # Initialize vectorizer
        vectorizer_params = vectorizer_params or {
            "max_features": 10000,
            "ngram_range": (1, 2),
            "min_df": 2,
            "stop_words": "english"
        }
        
        self.vectorizer = TfidfVectorizer(**vectorizer_params)
        
        # Transform text to feature vectors
        X = self.vectorizer.fit_transform(texts)
        y = labels
        
        # Train classifier
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X, y)
    
    def save_model(self, model_path: str) -> None:
        """
        Save the trained model to a file.
        
        Args:
            model_path: Path to save the model
        """
        if not self.model:
            raise ValueError("No trained model to save")
            
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        self.model_path = model_path
    
    def save_vectorizer(self, vectorizer_path: str) -> None:
        """
        Save the feature vectorizer to a file.
        
        Args:
            vectorizer_path: Path to save the vectorizer
        """
        if not self.vectorizer:
            raise ValueError("No vectorizer to save")
            
        os.makedirs(os.path.dirname(vectorizer_path), exist_ok=True)
        
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        self.feature_extractor_path = vectorizer_path
    
    def convert_to_onnx(self, output_path: str) -> None:
        """
        Convert the trained scikit-learn model to ONNX format.
        
        Args:
            output_path: Path to save the ONNX model
        """
        if not self.model:
            raise ValueError("No trained model to convert")
            
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
            
            # Define input type
            initial_type = [('input', FloatTensorType([None, self.vectorizer.get_feature_names_out().shape[0]]))]
            
            # Convert model
            onnx_model = convert_sklearn(self.model, initial_types=initial_type)
            
            # Save the model
            with open(output_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
                
            print(f"Model converted to ONNX format: {output_path}")
            
        except ImportError:
            print("skl2onnx package is required for ONNX conversion.")
        except Exception as e:
            print(f"Error converting model to ONNX: {str(e)}")
