"""
Evaluation metrics for semantic segmentation.
Includes IoU, Dice, pixel accuracy, mAP approximations, and frequency-weighted IoU.
"""

import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics import confusion_matrix, classification_report


def pixel_accuracy(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Calculate overall pixel accuracy.
    
    Args:
        pred_mask (np.ndarray): Predicted segmentation mask (H, W).
        gt_mask (np.ndarray): Ground truth segmentation mask (H, W).
    
    Returns:
        float: Pixel accuracy (0.0 to 1.0).
    """
    correct_pixels = np.sum(pred_mask == gt_mask)
    total_pixels = gt_mask.size
    return float(correct_pixels / total_pixels) if total_pixels > 0 else 0.0


def mean_pixel_accuracy(pred_mask: np.ndarray, gt_mask: np.ndarray, num_classes: int) -> float:
    """
    Calculate mean pixel accuracy across all classes.
    
    Args:
        pred_mask (np.ndarray): Predicted segmentation mask (H, W).
        gt_mask (np.ndarray): Ground truth segmentation mask (H, W).
        num_classes (int): Number of classes.
    
    Returns:
        float: Mean pixel accuracy across classes.
    """
    class_accuracies = []
    for class_id in range(num_classes):
        true_mask = (gt_mask == class_id)
        pred_mask_class = (pred_mask == class_id)
        
        correct = np.logical_and(true_mask, pred_mask_class).sum()
        total = true_mask.sum()
        
        if total > 0:
            accuracy = correct / total
        else:
            accuracy = 1.0  # Perfect accuracy if class doesn't exist in GT
        class_accuracies.append(accuracy)
    
    return float(np.mean(class_accuracies)) if class_accuracies else 0.0


def per_class_iou(pred_mask: np.ndarray, gt_mask: np.ndarray, num_classes: int) -> List[float]:
    """
    Calculate Intersection over Union (IoU) for each class.
    
    Args:
        pred_mask (np.ndarray): Predicted segmentation mask (H, W).
        gt_mask (np.ndarray): Ground truth segmentation mask (H, W).
        num_classes (int): Number of classes.
    
    Returns:
        List[float]: List of IoU values for each class.
    """
    ious = []
    for class_id in range(num_classes):
        true_mask = (gt_mask == class_id)
        pred_mask_class = (pred_mask == class_id)
        
        intersection = np.logical_and(true_mask, pred_mask_class).sum()
        union = np.logical_or(true_mask, pred_mask_class).sum()
        
        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = intersection / union
        ious.append(float(iou))
    
    return ious


def mean_iou(per_class_ious: List[float]) -> float:
    """
    Calculate mean IoU from per-class IoU values.
    
    Args:
        per_class_ious (List[float]): List of per-class IoU values.
    
    Returns:
        float: Mean IoU.
    """
    return float(np.mean(per_class_ious)) if per_class_ious else 0.0


def frequency_weighted_iou(pred_mask: np.ndarray, gt_mask: np.ndarray, num_classes: int) -> float:
    """
    Calculate Frequency Weighted IoU.
    
    Args:
        pred_mask (np.ndarray): Predicted segmentation mask (H, W).
        gt_mask (np.ndarray): Ground truth segmentation mask (H, W).
        num_classes (int): Number of classes.
    
    Returns:
        float: Frequency weighted IoU.
    """
    ious = per_class_iou(pred_mask, gt_mask, num_classes)
    
    # Calculate class frequencies in ground truth
    class_frequencies = []
    total_pixels = gt_mask.size
    
    for class_id in range(num_classes):
        class_pixels = np.sum(gt_mask == class_id)
        frequency = class_pixels / total_pixels if total_pixels > 0 else 0.0
        class_frequencies.append(frequency)
    
    # Calculate weighted IoU
    fw_iou = sum(freq * iou for freq, iou in zip(class_frequencies, ious))
    return float(fw_iou)


def dice_coefficient(pred_mask: np.ndarray, gt_mask: np.ndarray, class_id: int) -> float:
    """
    Calculate Dice coefficient for a specific class.
    
    Args:
        pred_mask (np.ndarray): Predicted segmentation mask (H, W).
        gt_mask (np.ndarray): Ground truth segmentation mask (H, W).
        class_id (int): Class index.
    
    Returns:
        float: Dice coefficient for the class.
    """
    true_mask = (gt_mask == class_id)
    pred_mask_class = (pred_mask == class_id)
    
    intersection = np.logical_and(true_mask, pred_mask_class).sum()
    
    # Dice coefficient formula: 2 * |A ∩ B| / (|A| + |B|)
    dice = 2.0 * intersection / (true_mask.sum() + pred_mask_class.sum()) \
        if (true_mask.sum() + pred_mask_class.sum()) > 0 else 1.0
    
    return float(dice)


def per_class_dice(pred_mask: np.ndarray, gt_mask: np.ndarray, num_classes: int) -> List[float]:
    """
    Calculate Dice coefficient for each class.
    
    Args:
        pred_mask (np.ndarray): Predicted segmentation mask (H, W).
        gt_mask (np.ndarray): Ground truth segmentation mask (H, W).
        num_classes (int): Number of classes.
    
    Returns:
        List[float]: List of Dice coefficients for each class.
    """
    dice_scores = []
    for class_id in range(num_classes):
        dice = dice_coefficient(pred_mask, gt_mask, class_id)
        dice_scores.append(dice)
    return dice_scores


def mean_dice(per_class_dices: List[float]) -> float:
    """
    Calculate mean Dice coefficient.
    
    Args:
        per_class_dices (List[float]): List of per-class Dice values.
    
    Returns:
        float: Mean Dice coefficient.
    """
    return float(np.mean(per_class_dices)) if per_class_dices else 0.0


def average_precision_at_threshold(pred_mask: np.ndarray, gt_mask: np.ndarray, 
                                   class_id: int, threshold: float = 0.5) -> float:
    """
    Approximate Average Precision at a specific IoU threshold.
    For semantic segmentation, we approximate AP using precision-recall.
    
    Args:
        pred_mask (np.ndarray): Predicted segmentation mask (H, W).
        gt_mask (np.ndarray): Ground truth segmentation mask (H, W).
        class_id (int): Class index.
        threshold (float): IoU threshold (unused for simplicity, kept for compatibility).
    
    Returns:
        float: Approximate AP (precision-based).
    """
    true_mask = (gt_mask == class_id).astype(int)
    pred_mask_class = (pred_mask == class_id).astype(int)
    
    # Calculate True Positives, False Positives, and False Negatives
    tp = np.logical_and(true_mask, pred_mask_class).sum()
    fp = np.logical_and(1 - true_mask, pred_mask_class).sum()
    fn = np.logical_and(true_mask, 1 - pred_mask_class).sum()
    
    # Calculate precision and recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # For semantic segmentation, AP is approximated as precision
    # since we're dealing with dense predictions (not sparse detections)
    return float(precision)


def map_at_thresholds(pred_mask: np.ndarray, gt_mask: np.ndarray, 
                      num_classes: int, thresholds: List[float] = None) -> Dict[str, float]:
    """
    Calculate mean Average Precision (mAP) at specified IoU thresholds.
    
    Args:
        pred_mask (np.ndarray): Predicted segmentation mask (H, W).
        gt_mask (np.ndarray): Ground truth segmentation mask (H, W).
        num_classes (int): Number of classes.
        thresholds (List[float]): List of IoU thresholds (default: [0.5, 0.75]).
    
    Returns:
        Dict[str, float]: Dictionary with 'mAP@50', 'mAP@75', etc.
    """
    if thresholds is None:
        thresholds = [0.5, 0.75]
    
    results = {}
    for threshold in thresholds:
        ap_scores = []
        for class_id in range(num_classes):
            ap = average_precision_at_threshold(pred_mask, gt_mask, class_id, threshold)
            ap_scores.append(ap)
        map_value = np.mean(ap_scores) if ap_scores else 0.0
        results[f'mAP@{int(threshold*100)}'] = float(map_value)
    
    return results


def compute_confusion_matrix(pred_mask: np.ndarray, gt_mask: np.ndarray, 
                             num_classes: int) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        pred_mask (np.ndarray): Predicted segmentation mask (H, W).
        gt_mask (np.ndarray): Ground truth segmentation mask (H, W).
        num_classes (int): Number of classes.
    
    Returns:
        np.ndarray: Confusion matrix (num_classes, num_classes).
    """
    return confusion_matrix(gt_mask.flatten(), pred_mask.flatten(), 
                          labels=list(range(num_classes)))


def compute_all_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray, 
                        num_classes: int, class_names: List[str] = None) -> Dict:
    """
    Compute all metrics in a single function.
    
    Args:
        pred_mask (np.ndarray): Predicted segmentation mask (H, W).
        gt_mask (np.ndarray): Ground truth segmentation mask (H, W).
        num_classes (int): Number of classes.
        class_names (List[str]): List of class names (optional).
    
    Returns:
        Dict: Dictionary containing all computed metrics.
    """
    if class_names is None:
        class_names = [f'class_{i}' for i in range(num_classes)]
    
    # Compute basic metrics
    pixel_acc = pixel_accuracy(pred_mask, gt_mask)
    mean_pixel_acc = mean_pixel_accuracy(pred_mask, gt_mask, num_classes)
    per_iou = per_class_iou(pred_mask, gt_mask, num_classes)
    mean_iou_val = mean_iou(per_iou)
    fw_iou = frequency_weighted_iou(pred_mask, gt_mask, num_classes)
    per_dice = per_class_dice(pred_mask, gt_mask, num_classes)
    mean_dice_val = mean_dice(per_dice)
    map_scores = map_at_thresholds(pred_mask, gt_mask, num_classes)
    
    # Build results dictionary
    metrics = {
        'pixel_accuracy': pixel_acc,
        'mean_pixel_accuracy': mean_pixel_acc,
        'mean_iou': mean_iou_val,
        'frequency_weighted_iou': fw_iou,
        'mean_dice': mean_dice_val,
        'per_class_iou': {class_names[i]: per_iou[i] for i in range(num_classes)},
        'per_class_dice': {class_names[i]: per_dice[i] for i in range(num_classes)},
    }
    
    # Add mAP scores
    metrics.update(map_scores)
    
    # Add per-class mAP scores
    for threshold in [0.5, 0.75]:
        per_class_ap = {}
        for class_id in range(num_classes):
            ap = average_precision_at_threshold(pred_mask, gt_mask, class_id, threshold)
            per_class_ap[class_names[class_id]] = ap
        metrics[f'per_class_ap_{int(threshold*100)}'] = per_class_ap
    
    return metrics
