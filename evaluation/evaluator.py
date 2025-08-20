"""
CurioScan model evaluator.

Comprehensive evaluation framework for different model types.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from training.metrics import MetricsCalculator, TokenLevelMetrics

logger = logging.getLogger(__name__)


class CurioScanEvaluator:
    """Comprehensive evaluator for CurioScan models."""
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.config = config
        self.device = device
        self.model_type = config.get("model", {}).get("type", "renderer_classifier")
        
        # Initialize metrics calculator
        self.metrics_calculator = MetricsCalculator(config)
        
        # Evaluation settings
        eval_config = config.get("evaluation", {})
        self.confidence_thresholds = eval_config.get("confidence_thresholds", [0.5, 0.7, 0.8, 0.9, 0.95])
        self.save_visualizations = eval_config.get("save_visualizations", True)
        self.detailed_analysis = eval_config.get("detailed_analysis", True)
    
    def evaluate(self, model: torch.nn.Module, data_loader: torch.utils.data.DataLoader,
                save_predictions: bool = False, output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Comprehensive model evaluation.
        
        Args:
            model: Model to evaluate
            data_loader: Data loader for evaluation
            save_predictions: Whether to save predictions
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary containing evaluation results
        """
        
        logger.info(f"Starting evaluation with {len(data_loader)} batches")
        
        model.eval()
        
        # Collect predictions and targets
        all_predictions = []
        all_targets = []
        all_confidences = []
        all_outputs = []
        prediction_details = []
        
        # Timing
        total_time = 0.0
        sample_count = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader, desc="Evaluating")):
                start_time = time.time()
                
                # Move batch to device
                batch = self._move_to_device(batch)
                
                # Forward pass
                outputs = model(batch)
                
                # Process outputs based on model type
                predictions, confidences = self._process_outputs(outputs, batch)
                
                # Collect results
                targets = batch["targets"].cpu().numpy()
                
                all_predictions.extend(predictions)
                all_targets.extend(targets)
                all_confidences.extend(confidences)
                all_outputs.append(outputs.cpu())
                
                # Save detailed predictions if requested
                if save_predictions:
                    batch_details = self._extract_prediction_details(
                        batch, outputs, predictions, confidences, targets
                    )
                    prediction_details.extend(batch_details)
                
                # Update timing
                batch_time = time.time() - start_time
                total_time += batch_time
                sample_count += len(targets)
                
                if batch_idx % 100 == 0:
                    logger.info(f"Processed {batch_idx}/{len(data_loader)} batches")
        
        logger.info("Evaluation forward pass completed")
        
        # Calculate metrics
        results = self._calculate_comprehensive_metrics(
            all_predictions, all_targets, all_confidences
        )
        
        # Add timing information
        results["processing_time"] = {
            "total_time": total_time,
            "avg_time_per_sample": total_time / max(sample_count, 1),
            "throughput": sample_count / max(total_time, 1e-6),
            "sample_count": sample_count
        }
        
        # Save predictions if requested
        if save_predictions and output_dir:
            self._save_predictions(prediction_details, output_dir)
        
        # Generate visualizations
        if self.save_visualizations and output_dir:
            self._generate_visualizations(results, output_dir)
        
        # Detailed analysis
        if self.detailed_analysis:
            detailed_results = self._detailed_analysis(
                all_predictions, all_targets, all_confidences, all_outputs
            )
            results.update(detailed_results)
        
        logger.info("Evaluation completed")
        
        return results
    
    def _move_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move batch to device."""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
            for k, v in batch.items()
        }
    
    def _process_outputs(self, outputs: torch.Tensor, batch: Dict[str, Any]) -> Tuple[List[int], List[float]]:
        """Process model outputs to get predictions and confidences."""
        
        if self.model_type == "renderer_classifier":
            # Classification outputs
            probabilities = F.softmax(outputs, dim=1)
            confidences = torch.max(probabilities, dim=1)[0].cpu().numpy()
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            
        elif self.model_type == "ocr":
            # OCR outputs (simplified)
            # This would depend on the specific OCR model architecture
            predictions = []
            confidences = []
            
            # Placeholder implementation
            batch_size = outputs.size(0)
            for i in range(batch_size):
                predictions.append(0)  # Placeholder
                confidences.append(0.5)  # Placeholder
                
        else:
            # Default to classification
            probabilities = F.softmax(outputs, dim=1)
            confidences = torch.max(probabilities, dim=1)[0].cpu().numpy()
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        
        return predictions.tolist(), confidences.tolist()
    
    def _extract_prediction_details(self, batch: Dict[str, Any], outputs: torch.Tensor,
                                   predictions: List[int], confidences: List[float],
                                   targets: np.ndarray) -> List[Dict[str, Any]]:
        """Extract detailed prediction information."""
        
        details = []
        
        for i in range(len(predictions)):
            detail = {
                "prediction": predictions[i],
                "target": int(targets[i]),
                "confidence": confidences[i],
                "correct": predictions[i] == targets[i],
                "metadata": batch.get("metadata", [{}])[i] if "metadata" in batch else {}
            }
            
            # Add model-specific details
            if self.model_type == "renderer_classifier":
                probabilities = F.softmax(outputs[i], dim=0).cpu().numpy()
                detail["class_probabilities"] = probabilities.tolist()
            
            details.append(detail)
        
        return details
    
    def _calculate_comprehensive_metrics(self, predictions: List[int], targets: List[int],
                                       confidences: List[float]) -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics."""
        
        # Basic metrics
        overall_metrics = self.metrics_calculator.calculate_metrics(predictions, targets)
        
        # Confidence-based metrics
        confidence_metrics = self.metrics_calculator.calculate_confidence_metrics(
            confidences, predictions, targets
        )
        
        # Per-class metrics
        per_class_metrics = self._calculate_per_class_metrics(predictions, targets)
        
        # Error analysis
        error_analysis = self._analyze_errors(predictions, targets, confidences)
        
        return {
            "overall_metrics": overall_metrics,
            "confidence_metrics": confidence_metrics,
            "per_class_metrics": per_class_metrics,
            "error_analysis": error_analysis
        }
    
    def _calculate_per_class_metrics(self, predictions: List[int], targets: List[int]) -> Dict[str, Dict[str, float]]:
        """Calculate per-class metrics."""
        
        from sklearn.metrics import precision_recall_fscore_support
        
        # Get unique classes
        unique_classes = sorted(set(targets + predictions))
        
        # Calculate per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, predictions, labels=unique_classes, zero_division=0
        )
        
        per_class_metrics = {}
        class_names = self._get_class_names()
        
        for i, class_id in enumerate(unique_classes):
            class_name = class_names[class_id] if class_names and class_id < len(class_names) else f"class_{class_id}"
            
            per_class_metrics[class_name] = {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i])
            }
        
        return per_class_metrics
    
    def _analyze_errors(self, predictions: List[int], targets: List[int],
                       confidences: List[float]) -> Dict[str, Any]:
        """Analyze prediction errors."""
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        confidences = np.array(confidences)
        
        # Find errors
        errors = predictions != targets
        error_count = np.sum(errors)
        
        if error_count == 0:
            return {"error_count": 0, "error_rate": 0.0}
        
        # Error statistics
        error_confidences = confidences[errors]
        correct_confidences = confidences[~errors]
        
        error_analysis = {
            "error_count": int(error_count),
            "error_rate": float(error_count / len(predictions)),
            "avg_error_confidence": float(np.mean(error_confidences)),
            "avg_correct_confidence": float(np.mean(correct_confidences)),
            "high_confidence_errors": int(np.sum(error_confidences > 0.8)),
            "low_confidence_correct": int(np.sum(correct_confidences < 0.5))
        }
        
        # Most common errors
        error_pairs = list(zip(targets[errors], predictions[errors]))
        from collections import Counter
        common_errors = Counter(error_pairs).most_common(5)
        
        error_analysis["most_common_errors"] = [
            {"true_class": int(true_cls), "predicted_class": int(pred_cls), "count": count}
            for (true_cls, pred_cls), count in common_errors
        ]
        
        return error_analysis
    
    def _detailed_analysis(self, predictions: List[int], targets: List[int],
                          confidences: List[float], outputs: List[torch.Tensor]) -> Dict[str, Any]:
        """Perform detailed analysis of model behavior."""
        
        detailed_results = {}
        
        # Confidence distribution analysis
        confidences = np.array(confidences)
        detailed_results["confidence_distribution"] = {
            "mean": float(np.mean(confidences)),
            "std": float(np.std(confidences)),
            "min": float(np.min(confidences)),
            "max": float(np.max(confidences)),
            "percentiles": {
                "25": float(np.percentile(confidences, 25)),
                "50": float(np.percentile(confidences, 50)),
                "75": float(np.percentile(confidences, 75)),
                "90": float(np.percentile(confidences, 90)),
                "95": float(np.percentile(confidences, 95))
            }
        }
        
        # Prediction distribution
        from collections import Counter
        pred_distribution = Counter(predictions)
        target_distribution = Counter(targets)
        
        detailed_results["prediction_distribution"] = dict(pred_distribution)
        detailed_results["target_distribution"] = dict(target_distribution)
        
        return detailed_results
    
    def _save_predictions(self, prediction_details: List[Dict[str, Any]], output_dir: Path):
        """Save detailed predictions to file."""
        
        predictions_file = output_dir / "predictions.json"
        
        with open(predictions_file, 'w') as f:
            json.dump(prediction_details, f, indent=2, default=str)
        
        logger.info(f"Predictions saved to {predictions_file}")
    
    def _generate_visualizations(self, results: Dict[str, Any], output_dir: Path):
        """Generate evaluation visualizations."""
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Confusion matrix
            if "confusion_matrix" in results:
                plt.figure(figsize=(10, 8))
                sns.heatmap(results["confusion_matrix"], annot=True, fmt='d', cmap='Blues')
                plt.title("Confusion Matrix")
                plt.ylabel("True Label")
                plt.xlabel("Predicted Label")
                plt.savefig(output_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            # Confidence distribution
            if "confidence_distribution" in results:
                plt.figure(figsize=(10, 6))
                # This would plot confidence distribution
                plt.title("Confidence Distribution")
                plt.savefig(output_dir / "confidence_distribution.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            logger.info("Visualizations saved to output directory")
            
        except ImportError:
            logger.warning("Matplotlib/Seaborn not available, skipping visualizations")
        except Exception as e:
            logger.error(f"Failed to generate visualizations: {e}")
    
    def _get_class_names(self) -> Optional[List[str]]:
        """Get class names for the current model type."""
        
        if self.model_type == "renderer_classifier":
            return [
                "digital_pdf", "scanned_image", "photograph", "docx",
                "form", "table_heavy", "invoice", "handwritten"
            ]
        else:
            return None
