#!/usr/bin/env python3
"""
Main training script for CurioScan models.

This script supports training various models including:
- Renderer classifier
- OCR models
- Table detection models
- Layout analysis models

Usage:
    python training/train.py --config configs/demo.yaml
    python training/train.py --config configs/demo.yaml --model renderer_classifier
    python training/train.py --config configs/demo.yaml --resume checkpoints/latest.pth
"""

import argparse
import logging
import os
import sys
import yaml
from pathlib import Path
from typing import Dict, Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from training.trainer import CurioScanTrainer
from training.data_loader import create_data_loaders
from training.utils import setup_training, setup_logging
from models import create_renderer_classifier

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train CurioScan models")
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="renderer_classifier",
        choices=["renderer_classifier", "ocr", "table_detector", "layout_analyzer"],
        help="Model type to train"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for checkpoints and logs"
    )
    
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_distributed_training(local_rank: int) -> bool:
    """Setup distributed training if available."""
    if local_rank == -1:
        return False
    
    # Initialize distributed training
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    
    return True


def create_model(model_type: str, config: Dict[str, Any]) -> torch.nn.Module:
    """Create model based on type and configuration."""
    if model_type == "renderer_classifier":
        return create_renderer_classifier(config)
    elif model_type == "ocr":
        # Placeholder for OCR model creation
        raise NotImplementedError("OCR model training not yet implemented")
    elif model_type == "table_detector":
        # Placeholder for table detector training
        raise NotImplementedError("Table detector training not yet implemented")
    elif model_type == "layout_analyzer":
        # Placeholder for layout analyzer training
        raise NotImplementedError("Layout analyzer training not yet implemented")
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup logging
    setup_logging(debug=args.debug)
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    
    # Setup distributed training
    is_distributed = setup_distributed_training(args.local_rank)
    
    # Setup training environment
    device, world_size, rank = setup_training(config, args.local_rank)
    logger.info(f"Training on device: {device}, world_size: {world_size}, rank: {rank}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model
    model = create_model(args.model, config)
    model = model.to(device)
    
    # Wrap model for distributed training
    if is_distributed:
        model = DDP(model, device_ids=[args.local_rank])
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(config, world_size, rank)
    
    # Create trainer
    trainer = CurioScanTrainer(
        model=model,
        config=config,
        device=device,
        output_dir=output_dir,
        is_distributed=is_distributed,
        rank=rank
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
        logger.info(f"Resumed training from epoch {start_epoch}")
    
    # Train model
    try:
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            start_epoch=start_epoch
        )
        
        # Evaluate on test set
        if test_loader is not None:
            test_metrics = trainer.evaluate(test_loader)
            logger.info(f"Test metrics: {test_metrics}")
        
        logger.info("Training completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        trainer.save_checkpoint(epoch=-1, is_best=False, filename="interrupted.pth")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    
    finally:
        # Cleanup distributed training
        if is_distributed:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
