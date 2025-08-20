"""
Training Package for CurioScan

This package contains training scripts, data loaders, and utilities for training
OCR models, renderer classifiers, and other components.
"""

from .train import train_model
from .data_loader import CurioScanDataset, create_data_loaders
from .trainer import CurioScanTrainer
from .utils import setup_training, save_checkpoint, load_checkpoint

__all__ = [
    "train_model",
    "CurioScanDataset",
    "create_data_loaders", 
    "CurioScanTrainer",
    "setup_training",
    "save_checkpoint",
    "load_checkpoint"
]
