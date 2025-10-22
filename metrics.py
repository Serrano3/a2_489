"""
    This module calculates relevant metrics.
"""

from typing import Sequence, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

def calculate_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: Sequence[float],
) -> Tuple[float, float, float, float, float, float, float]:
    """
    This function computes the following metrics based on ground truth labels,
    predicted labels, and predicted probabilities:
      - Accuracy
      - Precision
      - Recall
      - Specificity
      - F1 Score
      - ROC AUC 
      - PR AUC (Average Precision)

    Args:
        y_true (Sequence[int]): Ground Truth binary labels.
        y_pred (Sequence[int]): Predicted binary labels.
        y_prob (Sequence[float]): Predicted probabilities for the positive class.

    Returns:
        A tuple containing:
        (accuracy, precision, recall, specificity, f1_score, roc_auc, pr_auc)
    """

    """
    
    
    
    To Do
    
    
    
    """

    return acc, prec, rec, spec, f1, roc_auc, pr_auc