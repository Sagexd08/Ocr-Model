"""
Renderer Classifier for CurioScan

This module implements a lightweight CNN/ViT classifier that determines the render type
of input documents using both visual features and metadata.

Supported render types:
- digital_pdf: PDF with embedded text
- scanned_image: Scanned document as image
- photograph: Photographed document
- docx: Microsoft Word document
- form: Structured form document
- table_heavy: Document with many tables
- invoice: Invoice or receipt
- handwritten: Handwritten document
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import timm
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class RendererClassifier(nn.Module):
    """
    Lightweight classifier for document render type detection.
    
    Uses EfficientNet backbone with additional metadata features.
    """
    
    RENDER_TYPES = [
        "digital_pdf",
        "scanned_image", 
        "photograph",
        "docx",
        "form",
        "table_heavy",
        "invoice",
        "handwritten"
    ]
    
    def __init__(
        self,
        model_name: str = "efficientnet_b0",
        num_classes: int = 8,
        input_size: Tuple[int, int] = (224, 224),
        pretrained: bool = True,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.input_size = input_size
        
        # Vision backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier head
            global_pool="avg"
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, *input_size)
            backbone_features = self.backbone(dummy_input).shape[1]
        
        # Metadata features (MIME type, page count, text presence, etc.)
        self.metadata_dim = 16
        self.metadata_embedding = nn.Sequential(
            nn.Linear(self.metadata_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32)
        )
        
        # Combined classifier
        combined_dim = backbone_features + 32
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
    def extract_metadata_features(self, metadata: Dict) -> torch.Tensor:
        """
        Extract numerical features from document metadata.
        
        Args:
            metadata: Dictionary containing document metadata
            
        Returns:
            Tensor of metadata features
        """
        features = []
        
        # MIME type encoding (one-hot)
        mime_types = ["application/pdf", "image/jpeg", "image/png", "image/tiff", 
                     "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]
        mime_encoding = [1.0 if metadata.get("mime_type") == mt else 0.0 for mt in mime_types]
        features.extend(mime_encoding)
        
        # Page count (normalized)
        page_count = min(metadata.get("page_count", 1), 100) / 100.0
        features.append(page_count)
        
        # File size (normalized, MB)
        file_size = min(metadata.get("file_size", 0), 50 * 1024 * 1024) / (50 * 1024 * 1024)
        features.append(file_size)
        
        # Text presence indicators
        features.append(1.0 if metadata.get("has_embedded_text", False) else 0.0)
        features.append(metadata.get("text_density", 0.0))  # Ratio of text to image area
        
        # Image characteristics
        features.append(metadata.get("image_width", 0) / 2000.0)  # Normalized width
        features.append(metadata.get("image_height", 0) / 2000.0)  # Normalized height
        features.append(metadata.get("color_channels", 3) / 3.0)  # RGB=3, Grayscale=1
        
        # Quality indicators
        features.append(metadata.get("blur_score", 0.0))  # 0-1, higher = more blurry
        features.append(metadata.get("noise_score", 0.0))  # 0-1, higher = more noisy
        features.append(metadata.get("skew_angle", 0.0) / 45.0)  # Normalized skew angle
        
        # Pad to fixed size
        while len(features) < self.metadata_dim:
            features.append(0.0)
        
        return torch.tensor(features[:self.metadata_dim], dtype=torch.float32)
    
    def forward(self, images: torch.Tensor, metadata: List[Dict]) -> torch.Tensor:
        """
        Forward pass of the classifier.
        
        Args:
            images: Batch of preprocessed images [B, 3, H, W]
            metadata: List of metadata dictionaries for each image
            
        Returns:
            Logits for each render type [B, num_classes]
        """
        batch_size = images.shape[0]
        
        # Extract visual features
        visual_features = self.backbone(images)  # [B, feature_dim]
        
        # Extract metadata features
        metadata_features = []
        for meta in metadata:
            meta_feat = self.extract_metadata_features(meta)
            metadata_features.append(meta_feat)
        
        metadata_features = torch.stack(metadata_features).to(images.device)  # [B, metadata_dim]
        metadata_embedded = self.metadata_embedding(metadata_features)  # [B, 32]
        
        # Combine features
        combined_features = torch.cat([visual_features, metadata_embedded], dim=1)
        
        # Classify
        logits = self.classifier(combined_features)
        
        return logits
    
    def predict(
        self, 
        image: Image.Image, 
        metadata: Dict,
        return_confidence: bool = True
    ) -> Tuple[str, float]:
        """
        Predict render type for a single image.
        
        Args:
            image: PIL Image
            metadata: Document metadata
            return_confidence: Whether to return confidence score
            
        Returns:
            Tuple of (predicted_type, confidence)
        """
        self.eval()
        
        with torch.no_grad():
            # Preprocess image
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            image_tensor = self.transform(image).unsqueeze(0)  # [1, 3, H, W]
            
            # Forward pass
            logits = self.forward(image_tensor, [metadata])
            probabilities = F.softmax(logits, dim=1)
            
            # Get prediction
            pred_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, pred_idx].item()
            
            predicted_type = self.RENDER_TYPES[pred_idx]
            
            if return_confidence:
                return predicted_type, confidence
            else:
                return predicted_type
    
    def predict_batch(
        self,
        images: List[Image.Image],
        metadata_list: List[Dict]
    ) -> List[Tuple[str, float]]:
        """
        Predict render types for a batch of images.
        
        Args:
            images: List of PIL Images
            metadata_list: List of metadata dictionaries
            
        Returns:
            List of (predicted_type, confidence) tuples
        """
        self.eval()
        
        with torch.no_grad():
            # Preprocess images
            image_tensors = []
            for image in images:
                if image.mode != "RGB":
                    image = image.convert("RGB")
                image_tensor = self.transform(image)
                image_tensors.append(image_tensor)
            
            batch_tensor = torch.stack(image_tensors)  # [B, 3, H, W]
            
            # Forward pass
            logits = self.forward(batch_tensor, metadata_list)
            probabilities = F.softmax(logits, dim=1)
            
            # Get predictions
            results = []
            for i in range(len(images)):
                pred_idx = torch.argmax(probabilities[i]).item()
                confidence = probabilities[i, pred_idx].item()
                predicted_type = self.RENDER_TYPES[pred_idx]
                results.append((predicted_type, confidence))
            
            return results


def create_renderer_classifier(config: Dict) -> RendererClassifier:
    """
    Factory function to create a renderer classifier from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured RendererClassifier instance
    """
    model_config = config.get("model", {}).get("renderer_classifier", {})
    
    return RendererClassifier(
        model_name=model_config.get("name", "efficientnet_b0"),
        num_classes=model_config.get("num_classes", 8),
        input_size=tuple(model_config.get("input_size", [224, 224])),
        pretrained=model_config.get("pretrained", True)
    )
