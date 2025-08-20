"""
CurioScan model trainer with advanced training features.

Supports:
- Mixed precision training
- Gradient accumulation
- Learning rate scheduling
- Early stopping
- Model checkpointing
- TensorBoard logging
- Distributed training
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from training.metrics import MetricsCalculator
from training.utils import AverageMeter, save_checkpoint, load_checkpoint

logger = logging.getLogger(__name__)


class CurioScanTrainer:
    """Advanced trainer for CurioScan models."""
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: torch.device,
        output_dir: Path,
        is_distributed: bool = False,
        rank: int = 0
    ):
        self.model = model
        self.config = config
        self.device = device
        self.output_dir = output_dir
        self.is_distributed = is_distributed
        self.rank = rank
        
        # Training configuration
        self.training_config = config.get("training", {})
        self.epochs = self.training_config.get("epochs", 100)
        self.save_every = self.training_config.get("save_every", 10)
        self.eval_every = self.training_config.get("eval_every", 5)
        self.gradient_accumulation_steps = self.training_config.get("gradient_accumulation_steps", 1)
        self.max_grad_norm = self.training_config.get("max_grad_norm", 1.0)
        
        # Mixed precision training
        self.use_amp = self.training_config.get("use_amp", True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Setup loss function
        self.criterion = self._create_criterion()
        
        # Setup metrics calculator
        self.metrics_calculator = MetricsCalculator(config)
        
        # Setup logging
        if self.rank == 0:
            self.writer = SummaryWriter(log_dir=self.output_dir / "tensorboard")
        else:
            self.writer = None
        
        # Training state
        self.best_metric = 0.0
        self.patience_counter = 0
        self.early_stopping_patience = self.training_config.get("early_stopping_patience", 20)
        
        logger.info(f"Trainer initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        optimizer_config = self.training_config.get("optimizer", {})
        optimizer_type = optimizer_config.get("type", "adamw")
        lr = optimizer_config.get("learning_rate", 1e-4)
        weight_decay = optimizer_config.get("weight_decay", 1e-2)
        
        if optimizer_type.lower() == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=optimizer_config.get("betas", (0.9, 0.999))
            )
        elif optimizer_type.lower() == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=optimizer_config.get("momentum", 0.9)
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        scheduler_config = self.training_config.get("scheduler", {})
        
        if not scheduler_config:
            return None
        
        scheduler_type = scheduler_config.get("type", "cosine")
        
        if scheduler_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs,
                eta_min=scheduler_config.get("min_lr", 1e-6)
            )
        elif scheduler_type == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get("step_size", 30),
                gamma=scheduler_config.get("gamma", 0.1)
            )
        elif scheduler_type == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=scheduler_config.get("factor", 0.5),
                patience=scheduler_config.get("patience", 10),
                verbose=True
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    
    def _create_criterion(self) -> nn.Module:
        """Create loss function based on model type."""
        loss_config = self.training_config.get("loss", {})
        loss_type = loss_config.get("type", "cross_entropy")
        
        if loss_type == "cross_entropy":
            return nn.CrossEntropyLoss(
                label_smoothing=loss_config.get("label_smoothing", 0.0)
            )
        elif loss_type == "focal":
            from training.losses import FocalLoss
            return FocalLoss(
                alpha=loss_config.get("alpha", 1.0),
                gamma=loss_config.get("gamma", 2.0)
            )
        elif loss_type == "mse":
            return nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def train(self, train_loader, val_loader, start_epoch: int = 0):
        """Main training loop."""
        logger.info(f"Starting training from epoch {start_epoch}")
        
        for epoch in range(start_epoch, self.epochs):
            # Training phase
            train_metrics = self._train_epoch(train_loader, epoch)
            
            # Validation phase
            if epoch % self.eval_every == 0:
                val_metrics = self._validate_epoch(val_loader, epoch)
                
                # Check for improvement
                current_metric = val_metrics.get("f1_score", val_metrics.get("accuracy", 0.0))
                is_best = current_metric > self.best_metric
                
                if is_best:
                    self.best_metric = current_metric
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                
                # Learning rate scheduling
                if self.scheduler and isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(current_metric)
                
                # Early stopping
                if self.patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
                
                # Save checkpoint
                if self.rank == 0 and (epoch % self.save_every == 0 or is_best):
                    self.save_checkpoint(epoch, is_best)
            
            # Learning rate scheduling (non-plateau schedulers)
            if self.scheduler and not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()
            
            # Log metrics
            if self.rank == 0:
                self._log_metrics(train_metrics, val_metrics if epoch % self.eval_every == 0 else None, epoch)
        
        logger.info("Training completed!")
    
    def _train_epoch(self, train_loader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        losses = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        
        end = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            # Measure data loading time
            data_time.update(time.time() - end)
            
            # Move data to device
            batch = self._move_to_device(batch)
            
            # Forward pass
            with autocast(enabled=self.use_amp):
                outputs = self.model(batch)
                loss = self.criterion(outputs, batch["targets"])
                loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # Update metrics
            losses.update(loss.item() * self.gradient_accumulation_steps, batch["targets"].size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            
            # Log progress
            if batch_idx % 100 == 0 and self.rank == 0:
                logger.info(
                    f"Epoch: [{epoch}][{batch_idx}/{len(train_loader)}] "
                    f"Time {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                    f"Data {data_time.val:.3f} ({data_time.avg:.3f}) "
                    f"Loss {losses.val:.4f} ({losses.avg:.4f})"
                )
        
        return {"loss": losses.avg, "batch_time": batch_time.avg}
    
    def _validate_epoch(self, val_loader, epoch: int) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        losses = AverageMeter()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = self._move_to_device(batch)
                
                with autocast(enabled=self.use_amp):
                    outputs = self.model(batch)
                    loss = self.criterion(outputs, batch["targets"])
                
                losses.update(loss.item(), batch["targets"].size(0))
                
                # Collect predictions and targets for metrics
                predictions = torch.argmax(outputs, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(batch["targets"].cpu().numpy())
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_metrics(all_predictions, all_targets)
        metrics["loss"] = losses.avg
        
        if self.rank == 0:
            logger.info(f"Validation - Epoch: [{epoch}] Loss: {losses.avg:.4f} Metrics: {metrics}")
        
        return metrics
    
    def evaluate(self, test_loader) -> Dict[str, float]:
        """Evaluate model on test set."""
        logger.info("Evaluating on test set...")
        return self._validate_epoch(test_loader, -1)
    
    def save_checkpoint(self, epoch: int, is_best: bool = False, filename: str = None):
        """Save model checkpoint."""
        if filename is None:
            filename = f"checkpoint_epoch_{epoch}.pth"
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_metric": self.best_metric,
            "config": self.config,
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
        }
        
        filepath = self.output_dir / filename
        torch.save(checkpoint, filepath)
        
        if is_best:
            best_filepath = self.output_dir / "best_model.pth"
            torch.save(checkpoint, best_filepath)
            logger.info(f"New best model saved to {best_filepath}")
        
        logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_metric = checkpoint.get("best_metric", 0.0)
        
        if self.scaler and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        epoch = checkpoint["epoch"]
        logger.info(f"Loaded checkpoint from {checkpoint_path}, epoch {epoch}")
        
        return epoch + 1
    
    def _move_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move batch to device."""
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    def _log_metrics(self, train_metrics: Dict[str, float], 
                    val_metrics: Optional[Dict[str, float]], epoch: int):
        """Log metrics to TensorBoard."""
        if self.writer is None:
            return
        
        # Log training metrics
        for key, value in train_metrics.items():
            self.writer.add_scalar(f"train/{key}", value, epoch)
        
        # Log validation metrics
        if val_metrics:
            for key, value in val_metrics.items():
                self.writer.add_scalar(f"val/{key}", value, epoch)
        
        # Log learning rate
        current_lr = self.optimizer.param_groups[0]["lr"]
        self.writer.add_scalar("learning_rate", current_lr, epoch)
        
        self.writer.flush()
