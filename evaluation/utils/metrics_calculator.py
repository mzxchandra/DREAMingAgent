from typing import Dict, Set, Any

def compute_classification_metrics(tp: int, fp: int, tn: int, fn: int) -> Dict[str, float]:
    """Compute standard classification metrics from confusion matrix counts."""
    
    # Avoid division by zero
    total = tp + fp + tn + fn
    p_denom = tp + fp
    r_denom = tp + fn
    
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / p_denom if p_denom > 0 else 0.0
    recall = tp / r_denom if r_denom > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

def compute_detection_rate(detected_set: Set[str], ground_truth_set: Set[str]) -> float:
    """Compute the fraction of ground truth items that were detected."""
    if not ground_truth_set:
        return 0.0
    
    intersection = detected_set.intersection(ground_truth_set)
    return len(intersection) / len(ground_truth_set)
