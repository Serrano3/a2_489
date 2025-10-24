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
    
    # Accuracy
    acc  = accuracy_score(y_true, y_pred)
    
    # Precision
    prec = precision_score(y_true, y_pred, zero_division=0)
    
    # Recall
    rec  = recall_score(y_true, y_pred,zero_division=0)

    # count all the times when the condtion is met:
    # tn - when class is false and the prediction is false
    # fp - when the class is false but the prediction is true
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)

    # Specificity 
    spec = tn / (tn + fp + 0) if (tn + fp) > 0 else float("nan")
    
    # F1 Score
    f1 = f1_score(y_true, y_pred, zero_division=0)

    #ROC AUC and PR AUC
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc  = average_precision_score(y_true, y_prob)


    return acc, prec, rec, spec, f1, roc_auc, pr_auc