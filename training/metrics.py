"""
Metrics calculation for CurioScan model evaluation.

Provides comprehensive metrics for different model types.
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import torch


class MetricsCalculator:
    """Calculate various metrics for model evaluation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_type = config.get("model", {}).get("type", "renderer_classifier")
    
    def calculate_metrics(self, predictions: List[int], targets: List[int]) -> Dict[str, float]:
        """Calculate metrics based on model type."""
        
        if self.model_type == "renderer_classifier":
            return self._calculate_classification_metrics(predictions, targets)
        elif self.model_type == "ocr":
            return self._calculate_ocr_metrics(predictions, targets)
        elif self.model_type == "table_detector":
            return self._calculate_detection_metrics(predictions, targets)
        else:
            return self._calculate_classification_metrics(predictions, targets)
    
    def _calculate_classification_metrics(self, predictions: List[int], targets: List[int]) -> Dict[str, float]:
        """Calculate classification metrics."""
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # Basic metrics
        accuracy = accuracy_score(targets, predictions)
        precision = precision_score(targets, predictions, average='weighted', zero_division=0)
        recall = recall_score(targets, predictions, average='weighted', zero_division=0)
        f1 = f1_score(targets, predictions, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_macro = precision_score(targets, predictions, average='macro', zero_division=0)
        recall_macro = recall_score(targets, predictions, average='macro', zero_division=0)
        f1_macro = f1_score(targets, predictions, average='macro', zero_division=0)
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro
        }
        
        # Add per-class metrics if we have class names
        class_names = self._get_class_names()
        if class_names:
            report = classification_report(
                targets, predictions, 
                target_names=class_names, 
                output_dict=True,
                zero_division=0
            )
            
            for class_name in class_names:
                if class_name in report:
                    metrics[f"{class_name}_precision"] = report[class_name]["precision"]
                    metrics[f"{class_name}_recall"] = report[class_name]["recall"]
                    metrics[f"{class_name}_f1"] = report[class_name]["f1-score"]
        
        return metrics
    
    def _calculate_ocr_metrics(self, predictions: List[Any], targets: List[Any]) -> Dict[str, float]:
        """Calculate OCR-specific metrics."""
        
        # For OCR, we need different metrics like character accuracy, word accuracy, etc.
        # This is a simplified implementation
        
        if not predictions or not targets:
            return {"character_accuracy": 0.0, "word_accuracy": 0.0, "bleu_score": 0.0}
        
        # Character-level accuracy
        char_correct = 0
        char_total = 0
        
        # Word-level accuracy
        word_correct = 0
        word_total = 0
        
        for pred, target in zip(predictions, targets):
            if isinstance(pred, str) and isinstance(target, str):
                # Character accuracy
                for p_char, t_char in zip(pred, target):
                    if p_char == t_char:
                        char_correct += 1
                    char_total += 1
                
                # Word accuracy
                pred_words = pred.split()
                target_words = target.split()
                
                for p_word, t_word in zip(pred_words, target_words):
                    if p_word == t_word:
                        word_correct += 1
                    word_total += 1
        
        char_accuracy = char_correct / max(char_total, 1)
        word_accuracy = word_correct / max(word_total, 1)
        
        # BLEU score (simplified)
        bleu_score = self._calculate_bleu_score(predictions, targets)
        
        return {
            "character_accuracy": char_accuracy,
            "word_accuracy": word_accuracy,
            "bleu_score": bleu_score
        }
    
    def _calculate_detection_metrics(self, predictions: List[Any], targets: List[Any]) -> Dict[str, float]:
        """Calculate object detection metrics (mAP, etc.)."""
        
        # This would implement proper detection metrics like mAP
        # For now, return placeholder metrics
        
        return {
            "map_50": 0.0,  # mAP at IoU 0.5
            "map_75": 0.0,  # mAP at IoU 0.75
            "map_50_95": 0.0,  # mAP averaged over IoU 0.5-0.95
            "precision": 0.0,
            "recall": 0.0
        }
    
    def _calculate_bleu_score(self, predictions: List[str], targets: List[str]) -> float:
        """Calculate BLEU score for text predictions."""
        
        # Simplified BLEU calculation
        # In practice, you'd use nltk.translate.bleu_score
        
        if not predictions or not targets:
            return 0.0
        
        total_score = 0.0
        count = 0
        
        for pred, target in zip(predictions, targets):
            if isinstance(pred, str) and isinstance(target, str):
                pred_words = set(pred.split())
                target_words = set(target.split())
                
                if target_words:
                    overlap = len(pred_words & target_words)
                    score = overlap / len(target_words)
                    total_score += score
                    count += 1
        
        return total_score / max(count, 1)
    
    def _get_class_names(self) -> Optional[List[str]]:
        """Get class names for the current model type."""
        
        if self.model_type == "renderer_classifier":
            return [
                "digital_pdf", "scanned_image", "photograph", "docx",
                "form", "table_heavy", "invoice", "handwritten"
            ]
        else:
            return None
    
    def calculate_confidence_metrics(self, confidences: List[float], 
                                   predictions: List[int], targets: List[int]) -> Dict[str, float]:
        """Calculate confidence-related metrics."""
        
        confidences = np.array(confidences)
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # Accuracy vs confidence correlation
        correct = (predictions == targets).astype(float)
        
        # Calibration metrics
        calibration_error = self._calculate_calibration_error(confidences, correct)
        
        # Confidence statistics
        avg_confidence = np.mean(confidences)
        confidence_std = np.std(confidences)
        
        # Accuracy at different confidence thresholds
        thresholds = [0.5, 0.7, 0.8, 0.9, 0.95]
        threshold_metrics = {}
        
        for threshold in thresholds:
            mask = confidences >= threshold
            if np.sum(mask) > 0:
                threshold_acc = np.mean(correct[mask])
                threshold_coverage = np.mean(mask)
            else:
                threshold_acc = 0.0
                threshold_coverage = 0.0
            
            threshold_metrics[f"accuracy_at_{threshold}"] = threshold_acc
            threshold_metrics[f"coverage_at_{threshold}"] = threshold_coverage
        
        return {
            "calibration_error": calibration_error,
            "avg_confidence": avg_confidence,
            "confidence_std": confidence_std,
            **threshold_metrics
        }
    
    def _calculate_calibration_error(self, confidences: np.ndarray, 
                                   correct: np.ndarray, num_bins: int = 10) -> float:
        """Calculate Expected Calibration Error (ECE)."""
        
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = correct[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece


class TokenLevelMetrics:
    """Metrics for token-level OCR evaluation."""
    
    @staticmethod
    def calculate_token_metrics(pred_tokens: List[Dict], 
                              target_tokens: List[Dict],
                              iou_threshold: float = 0.5) -> Dict[str, float]:
        """Calculate token-level precision, recall, and F1."""
        
        # Match tokens based on IoU
        matches = TokenLevelMetrics._match_tokens(pred_tokens, target_tokens, iou_threshold)
        
        true_positives = len(matches)
        false_positives = len(pred_tokens) - true_positives
        false_negatives = len(target_tokens) - true_positives
        
        precision = true_positives / max(true_positives + false_positives, 1)
        recall = true_positives / max(true_positives + false_negatives, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        
        return {
            "token_precision": precision,
            "token_recall": recall,
            "token_f1": f1,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives
        }
    
    @staticmethod
    def _match_tokens(pred_tokens: List[Dict], target_tokens: List[Dict],
                     iou_threshold: float) -> List[Tuple[int, int]]:
        """Match predicted and target tokens based on IoU."""
        
        matches = []
        used_targets = set()
        
        for i, pred_token in enumerate(pred_tokens):
            best_iou = 0.0
            best_target_idx = -1
            
            for j, target_token in enumerate(target_tokens):
                if j in used_targets:
                    continue
                
                iou = TokenLevelMetrics._calculate_bbox_iou(
                    pred_token.get("bbox", [0, 0, 0, 0]),
                    target_token.get("bbox", [0, 0, 0, 0])
                )
                
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_target_idx = j
            
            if best_target_idx >= 0:
                matches.append((i, best_target_idx))
                used_targets.add(best_target_idx)
        
        return matches
    
    @staticmethod
    def _calculate_bbox_iou(bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate IoU between two bounding boxes."""
        
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / max(union, 1e-8)
