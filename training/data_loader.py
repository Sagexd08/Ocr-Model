"""
Data loaders for CurioScan model training.

Supports various data formats and augmentation strategies.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import json
import random

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms

logger = logging.getLogger(__name__)


class CurioScanDataset(Dataset):
    """Base dataset class for CurioScan training."""
    
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        target_size: Tuple[int, int] = (224, 224)
    ):
        self.data_path = Path(data_path)
        self.split = split
        self.transform = transform
        self.target_size = target_size
        
        # Load data
        self.samples = self._load_samples()
        
        logger.info(f"Loaded {len(self.samples)} samples for {split} split")
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load samples from data directory."""
        samples = []
        
        # Look for annotation files
        annotation_file = self.data_path / f"{self.split}_annotations.json"
        
        if annotation_file.exists():
            with open(annotation_file, 'r') as f:
                annotations = json.load(f)
            
            for ann in annotations:
                sample = {
                    "image_path": self.data_path / ann["image_path"],
                    "label": ann.get("label", 0),
                    "metadata": ann.get("metadata", {})
                }
                samples.append(sample)
        else:
            # Fallback: scan directory structure
            logger.warning(f"Annotation file not found: {annotation_file}")
            samples = self._scan_directory()
        
        return samples
    
    def _scan_directory(self) -> List[Dict[str, Any]]:
        """Scan directory for images (fallback method)."""
        samples = []
        
        image_extensions = {".jpg", ".jpeg", ".png", ".tiff", ".bmp"}
        
        for image_path in self.data_path.rglob("*"):
            if image_path.suffix.lower() in image_extensions:
                # Try to infer label from directory structure
                label = 0  # Default label
                if "digital_pdf" in str(image_path):
                    label = 0
                elif "scanned_image" in str(image_path):
                    label = 1
                elif "photograph" in str(image_path):
                    label = 2
                # Add more label mappings as needed
                
                sample = {
                    "image_path": image_path,
                    "label": label,
                    "metadata": {}
                }
                samples.append(sample)
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        # Load image
        image = self._load_image(sample["image_path"])
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return {
            "image": image,
            "targets": torch.tensor(sample["label"], dtype=torch.long),
            "metadata": sample["metadata"]
        }
    
    def _load_image(self, image_path: Path) -> Image.Image:
        """Load and preprocess image."""
        try:
            image = Image.open(image_path).convert("RGB")
            
            # Resize to target size
            image = image.resize(self.target_size, Image.Resampling.LANCZOS)
            
            return image
            
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            # Return a blank image as fallback
            return Image.new("RGB", self.target_size, color=(255, 255, 255))


class RendererClassifierDataset(CurioScanDataset):
    """Dataset for renderer classifier training."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Renderer type mapping
        self.label_mapping = {
            "digital_pdf": 0,
            "scanned_image": 1,
            "photograph": 2,
            "docx": 3,
            "form": 4,
            "table_heavy": 5,
            "invoice": 6,
            "handwritten": 7
        }
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        # Load image
        image = self._load_image(sample["image_path"])
        
        # Extract metadata features
        metadata = sample["metadata"]
        metadata_features = self._extract_metadata_features(metadata)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return {
            "image": image,
            "metadata": torch.tensor(metadata_features, dtype=torch.float32),
            "targets": torch.tensor(sample["label"], dtype=torch.long)
        }
    
    def _extract_metadata_features(self, metadata: Dict[str, Any]) -> List[float]:
        """Extract numerical features from metadata."""
        features = []
        
        # File size (normalized)
        file_size = metadata.get("file_size", 0)
        features.append(min(file_size / 1e6, 100.0))  # MB, capped at 100MB
        
        # Page count
        page_count = metadata.get("page_count", 1)
        features.append(min(page_count, 100.0))  # Capped at 100 pages
        
        # Has embedded text (binary)
        has_text = metadata.get("has_embedded_text", False)
        features.append(1.0 if has_text else 0.0)
        
        # Image dimensions (normalized)
        width = metadata.get("width", 800)
        height = metadata.get("height", 600)
        features.append(width / 1000.0)  # Normalized to ~1.0 for typical documents
        features.append(height / 1000.0)
        
        # Aspect ratio
        aspect_ratio = width / max(height, 1)
        features.append(aspect_ratio)
        
        return features


class OCRDataset(CurioScanDataset):
    """Dataset for OCR model training."""
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        # Load image
        image = self._load_image(sample["image_path"])
        
        # Load OCR annotations
        ocr_annotations = sample["metadata"].get("ocr_annotations", [])
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Process OCR targets
        targets = self._process_ocr_targets(ocr_annotations)
        
        return {
            "image": image,
            "targets": targets
        }
    
    def _process_ocr_targets(self, annotations: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Process OCR annotations into training targets."""
        # This would depend on the specific OCR model architecture
        # For now, return a placeholder
        return {
            "text_tokens": torch.zeros(100, dtype=torch.long),  # Placeholder
            "bboxes": torch.zeros(100, 4, dtype=torch.float32),  # Placeholder
            "attention_mask": torch.zeros(100, dtype=torch.bool)  # Placeholder
        }


def create_transforms(config: Dict[str, Any], split: str) -> transforms.Compose:
    """Create image transforms for training/validation."""
    
    transform_config = config.get("transforms", {})
    target_size = transform_config.get("target_size", [224, 224])
    
    if split == "train":
        # Training transforms with augmentation
        transform_list = [
            transforms.Resize(target_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
    else:
        # Validation/test transforms (no augmentation)
        transform_list = [
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
    
    return transforms.Compose(transform_list)


def create_data_loaders(
    config: Dict[str, Any],
    world_size: int = 1,
    rank: int = 0
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Create data loaders for training, validation, and testing."""
    
    data_config = config.get("data", {})
    model_type = config.get("model", {}).get("type", "renderer_classifier")
    
    # Dataset paths
    train_path = data_config.get("train_path", "data/train")
    val_path = data_config.get("val_path", "data/val")
    test_path = data_config.get("test_path", "data/test")
    
    # Batch sizes
    batch_size = data_config.get("batch_size", 32)
    num_workers = data_config.get("num_workers", 4)
    
    # Create transforms
    train_transform = create_transforms(config, "train")
    val_transform = create_transforms(config, "val")
    
    # Create datasets
    if model_type == "renderer_classifier":
        train_dataset = RendererClassifierDataset(
            train_path, split="train", transform=train_transform
        )
        val_dataset = RendererClassifierDataset(
            val_path, split="val", transform=val_transform
        )
        test_dataset = RendererClassifierDataset(
            test_path, split="test", transform=val_transform
        ) if os.path.exists(test_path) else None
        
    elif model_type == "ocr":
        train_dataset = OCRDataset(
            train_path, split="train", transform=train_transform
        )
        val_dataset = OCRDataset(
            val_path, split="val", transform=val_transform
        )
        test_dataset = OCRDataset(
            test_path, split="test", transform=val_transform
        ) if os.path.exists(test_path) else None
        
    else:
        # Default to base dataset
        train_dataset = CurioScanDataset(
            train_path, split="train", transform=train_transform
        )
        val_dataset = CurioScanDataset(
            val_path, split="val", transform=val_transform
        )
        test_dataset = CurioScanDataset(
            test_path, split="test", transform=val_transform
        ) if os.path.exists(test_path) else None
    
    # Create samplers for distributed training
    train_sampler = None
    val_sampler = None
    test_sampler = None
    
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
        if test_dataset:
            test_sampler = DistributedSampler(
                test_dataset, num_replicas=world_size, rank=rank, shuffle=False
            )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=test_sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )
    
    logger.info(f"Created data loaders - Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader) if test_loader else 0}")
    
    return train_loader, val_loader, test_loader
