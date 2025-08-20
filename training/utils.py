"""
Training utilities for CurioScan.

Includes helper functions for training setup, logging, and model management.
"""

import os
import logging
import random
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import torch
import torch.distributed as dist
import numpy as np


def setup_logging(debug: bool = False):
    """Setup logging configuration."""
    
    log_level = logging.DEBUG if debug else logging.INFO
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )


def setup_training(config: Dict[str, Any], local_rank: int = -1) -> Tuple[torch.device, int, int]:
    """Setup training environment."""
    
    # Set random seeds for reproducibility
    seed = config.get("training", {}).get("seed", 42)
    set_random_seeds(seed)
    
    # Setup device
    if local_rank >= 0:
        # Distributed training
        device = torch.device(f"cuda:{local_rank}")
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        # Single GPU or CPU training
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        world_size = 1
        rank = 0
    
    return device, world_size, rank


def set_random_seeds(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif self._is_better(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_better(self, score: float) -> bool:
        if self.mode == "max":
            return score > self.best_score + self.min_delta
        else:
            return score < self.best_score - self.min_delta


def save_checkpoint(state: Dict[str, Any], filepath: str, is_best: bool = False):
    """Save model checkpoint."""
    
    torch.save(state, filepath)
    
    if is_best:
        best_filepath = filepath.replace('.pth', '_best.pth')
        torch.save(state, best_filepath)


def load_checkpoint(filepath: str, model: torch.nn.Module, 
                   optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, Any]:
    """Load model checkpoint."""
    
    checkpoint = torch.load(filepath, map_location='cpu')
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    """Get current learning rate from optimizer."""
    return optimizer.param_groups[0]['lr']


def warmup_learning_rate(optimizer: torch.optim.Optimizer, epoch: int, 
                        warmup_epochs: int, base_lr: float):
    """Apply learning rate warmup."""
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


class GradientClipping:
    """Gradient clipping utility."""
    
    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0):
        self.max_norm = max_norm
        self.norm_type = norm_type
    
    def __call__(self, model: torch.nn.Module) -> float:
        """Clip gradients and return the gradient norm."""
        return torch.nn.utils.clip_grad_norm_(
            model.parameters(), 
            self.max_norm, 
            norm_type=self.norm_type
        )


class ModelEMA:
    """Exponential Moving Average for model parameters."""
    
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update EMA parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Apply EMA parameters to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def create_optimizer(model: torch.nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """Create optimizer from configuration."""
    
    optimizer_config = config.get("optimizer", {})
    optimizer_type = optimizer_config.get("type", "adamw").lower()
    learning_rate = optimizer_config.get("learning_rate", 1e-4)
    weight_decay = optimizer_config.get("weight_decay", 1e-2)
    
    if optimizer_type == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=optimizer_config.get("betas", (0.9, 0.999))
        )
    elif optimizer_type == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=optimizer_config.get("betas", (0.9, 0.999))
        )
    elif optimizer_type == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=optimizer_config.get("momentum", 0.9)
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")


def create_scheduler(optimizer: torch.optim.Optimizer, 
                    config: Dict[str, Any]) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Create learning rate scheduler from configuration."""
    
    scheduler_config = config.get("scheduler", {})
    
    if not scheduler_config:
        return None
    
    scheduler_type = scheduler_config.get("type", "cosine").lower()
    
    if scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config.get("T_max", 100),
            eta_min=scheduler_config.get("eta_min", 1e-6)
        )
    elif scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config.get("step_size", 30),
            gamma=scheduler_config.get("gamma", 0.1)
        )
    elif scheduler_type == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=scheduler_config.get("milestones", [60, 80]),
            gamma=scheduler_config.get("gamma", 0.1)
        )
    elif scheduler_type == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=scheduler_config.get("gamma", 0.95)
        )
    elif scheduler_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_config.get("mode", "max"),
            factor=scheduler_config.get("factor", 0.5),
            patience=scheduler_config.get("patience", 10),
            verbose=True
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


def format_time(seconds: float) -> str:
    """Format time in seconds to human readable format."""
    
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def get_model_size(model: torch.nn.Module) -> Tuple[int, float]:
    """Get model size in parameters and MB."""
    
    param_count = sum(p.numel() for p in model.parameters())
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    
    return param_count, size_mb


def log_model_info(model: torch.nn.Module, logger: logging.Logger):
    """Log model information."""
    
    param_count, size_mb = get_model_size(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model parameters: {param_count:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Model size: {size_mb:.2f} MB")


class Timer:
    """Simple timer utility."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        self.start_time = time.time()
    
    def stop(self):
        self.end_time = time.time()
    
    def elapsed(self) -> float:
        if self.start_time is None:
            return 0.0
        
        end_time = self.end_time if self.end_time else time.time()
        return end_time - self.start_time
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()
