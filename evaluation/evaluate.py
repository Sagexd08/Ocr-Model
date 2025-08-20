"""
Main evaluation script for CurioScan models.

Provides comprehensive evaluation across different metrics and datasets.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import time

import torch
import numpy as np
import pandas as pd
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from evaluation.evaluator import CurioScanEvaluator
from evaluation.report_generator import ReportGenerator
from training.data_loader import create_data_loaders
from models import create_model
from training.utils import setup_logging, load_checkpoint

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="CurioScan Model Evaluation")
    
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to evaluation configuration file"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="test",
        help="Dataset split to evaluate on (test, val, demo_holdout)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cpu, cuda)"
    )
    
    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="Generate detailed evaluation report"
    )
    
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save model predictions for analysis"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load evaluation configuration."""
    with open(config_path, 'r') as f:
        if config_path.endswith('.json'):
            config = json.load(f)
        elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
            import yaml
            config = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path}")
    
    return config


def setup_device(device_str: str) -> torch.device:
    """Setup computation device."""
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)
    
    logger.info(f"Using device: {device}")
    return device


def load_model(config: Dict[str, Any], model_path: str, device: torch.device) -> torch.nn.Module:
    """Load trained model."""
    logger.info(f"Loading model from {model_path}")
    
    # Create model
    model = create_model(config)
    
    # Load checkpoint
    checkpoint = load_checkpoint(model_path, model)
    
    # Move to device and set to eval mode
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded successfully (epoch {checkpoint.get('epoch', 'unknown')})")
    
    return model


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Setup logging
    setup_logging(args.debug)
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup device
    device = setup_device(args.device)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = load_model(config, args.model_path, device)
    
    # Create data loaders
    logger.info("Creating data loaders...")
    
    # Override batch size in config
    config["data"]["batch_size"] = args.batch_size
    
    train_loader, val_loader, test_loader = create_data_loaders(config)
    
    # Select dataset
    if args.dataset == "train":
        eval_loader = train_loader
    elif args.dataset == "val":
        eval_loader = val_loader
    elif args.dataset == "test":
        eval_loader = test_loader
    elif args.dataset == "demo_holdout":
        # Load demo holdout dataset
        eval_loader = load_demo_holdout_dataset(config)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    if eval_loader is None:
        raise ValueError(f"Dataset {args.dataset} not available")
    
    # Create evaluator
    evaluator = CurioScanEvaluator(config, device)
    
    # Run evaluation
    logger.info(f"Starting evaluation on {args.dataset} dataset...")
    start_time = time.time()
    
    results = evaluator.evaluate(
        model, 
        eval_loader,
        save_predictions=args.save_predictions,
        output_dir=output_dir
    )
    
    eval_time = time.time() - start_time
    logger.info(f"Evaluation completed in {eval_time:.2f} seconds")
    
    # Save results
    results_file = output_dir / f"evaluation_results_{args.dataset}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Results saved to {results_file}")
    
    # Print summary
    print_evaluation_summary(results)
    
    # Generate detailed report
    if args.generate_report:
        logger.info("Generating detailed evaluation report...")
        
        report_generator = ReportGenerator(config)
        report_path = report_generator.generate_report(
            results, 
            output_dir,
            dataset_name=args.dataset,
            model_path=args.model_path
        )
        
        logger.info(f"Detailed report saved to {report_path}")
    
    logger.info("Evaluation completed successfully!")


def load_demo_holdout_dataset(config: Dict[str, Any]):
    """Load demo holdout dataset."""
    # This would load a special holdout dataset for demo purposes
    # For now, return None to indicate it's not implemented
    logger.warning("Demo holdout dataset not implemented")
    return None


def print_evaluation_summary(results: Dict[str, Any]):
    """Print evaluation summary to console."""
    
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    # Overall metrics
    overall_metrics = results.get("overall_metrics", {})
    
    print(f"\nOverall Performance:")
    print(f"  Accuracy:     {overall_metrics.get('accuracy', 0.0):.3f}")
    print(f"  Precision:    {overall_metrics.get('precision', 0.0):.3f}")
    print(f"  Recall:       {overall_metrics.get('recall', 0.0):.3f}")
    print(f"  F1 Score:     {overall_metrics.get('f1_score', 0.0):.3f}")
    
    # Confidence metrics
    confidence_metrics = results.get("confidence_metrics", {})
    if confidence_metrics:
        print(f"\nConfidence Analysis:")
        print(f"  Avg Confidence:     {confidence_metrics.get('avg_confidence', 0.0):.3f}")
        print(f"  Calibration Error:  {confidence_metrics.get('calibration_error', 0.0):.3f}")
        print(f"  Accuracy @ 0.8:     {confidence_metrics.get('accuracy_at_0.8', 0.0):.3f}")
        print(f"  Coverage @ 0.8:     {confidence_metrics.get('coverage_at_0.8', 0.0):.3f}")
    
    # Per-class metrics
    per_class_metrics = results.get("per_class_metrics", {})
    if per_class_metrics:
        print(f"\nPer-Class Performance:")
        for class_name, metrics in per_class_metrics.items():
            print(f"  {class_name:15} - F1: {metrics.get('f1', 0.0):.3f}, "
                  f"Precision: {metrics.get('precision', 0.0):.3f}, "
                  f"Recall: {metrics.get('recall', 0.0):.3f}")
    
    # Processing time
    processing_time = results.get("processing_time", {})
    if processing_time:
        print(f"\nProcessing Time:")
        print(f"  Total Time:         {processing_time.get('total_time', 0.0):.2f}s")
        print(f"  Avg Time per Sample: {processing_time.get('avg_time_per_sample', 0.0):.3f}s")
        print(f"  Throughput:         {processing_time.get('throughput', 0.0):.1f} samples/s")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
