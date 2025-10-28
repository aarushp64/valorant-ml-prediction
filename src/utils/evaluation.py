import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

"""
Comprehensive evaluation metrics for League of Legends match prediction.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def calculate_rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error (for regression tasks)."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def calculate_mae(y_true, y_pred):
    """Calculate Mean Absolute Error (for regression tasks)."""
    return np.mean(np.abs(y_true - y_pred))


def evaluate_model(y_true, y_pred):
    """Basic model evaluation with RMSE and MAE."""
    rmse = calculate_rmse(y_true, y_pred)
    mae = calculate_mae(y_true, y_pred)
    return {'rmse': rmse, 'mae': mae}


def calculate_classification_metrics(y_true: np.ndarray, 
                                     y_pred: np.ndarray,
                                     y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Calculate comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional, for probability-based metrics)
        
    Returns:
        Dictionary with classification metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='binary', zero_division=0),
    }
    
    # Add probability-based metrics if probabilities are provided
    if y_proba is not None:
        # Extract positive class probabilities if 2D
        if y_proba.ndim == 2:
            y_proba = y_proba[:, 1]
        
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        metrics['avg_precision'] = average_precision_score(y_true, y_proba)
    
    return metrics


def get_confusion_matrix(y_true: np.ndarray, 
                        y_pred: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
    """Calculate confusion matrix and derived counts.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Tuple of (confusion matrix, dictionary with TP/TN/FP/FN)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Extract values (assuming binary classification)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    counts = {
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
        'total': int(len(y_true))
    }
    
    return cm, counts


def get_classification_report(y_true: np.ndarray, 
                              y_pred: np.ndarray,
                              target_names: Optional[list] = None) -> Dict:
    """Get detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Optional names for classes
        
    Returns:
        Dictionary with classification report
    """
    if target_names is None:
        target_names = ['Loss', 'Win']
    
    report_dict = classification_report(
        y_true, y_pred, 
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )
    
    return report_dict


def calculate_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate specificity (true negative rate).
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Specificity score
    """
    cm, counts = get_confusion_matrix(y_true, y_pred)
    tn = counts['true_negatives']
    fp = counts['false_positives']
    
    if (tn + fp) == 0:
        return 0.0
    
    return tn / (tn + fp)


def calculate_sensitivity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate sensitivity (same as recall/true positive rate).
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Sensitivity score
    """
    return recall_score(y_true, y_pred, average='binary', zero_division=0)


def evaluate_model_comprehensive(y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 y_proba: Optional[np.ndarray] = None,
                                 model_name: str = "Model") -> Dict:
    """Perform comprehensive model evaluation.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)
        model_name: Name of the model for logging
        
    Returns:
        Dictionary with comprehensive evaluation metrics
    """
    logger.info(f"Evaluating {model_name}...")
    
    # Basic classification metrics
    metrics = calculate_classification_metrics(y_true, y_pred, y_proba)
    
    # Confusion matrix
    cm, counts = get_confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    metrics['confusion_matrix_counts'] = counts
    
    # Additional metrics
    metrics['specificity'] = calculate_specificity(y_true, y_pred)
    metrics['sensitivity'] = calculate_sensitivity(y_true, y_pred)
    
    # Classification report
    metrics['classification_report'] = get_classification_report(y_true, y_pred)
    
    logger.info(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, "
                f"F1: {metrics['f1_score']:.4f}, "
                f"ROC-AUC: {metrics.get('roc_auc', 'N/A')}")
    
    return metrics


def get_roc_curve_data(y_true: np.ndarray, 
                       y_proba: np.ndarray) -> Dict[str, np.ndarray]:
    """Calculate ROC curve data.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        
    Returns:
        Dictionary with FPR, TPR, and thresholds
    """
    # Extract positive class probabilities if 2D
    if y_proba.ndim == 2:
        y_proba = y_proba[:, 1]
    
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    
    return {
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'auc': roc_auc_score(y_true, y_proba)
    }


def get_precision_recall_curve_data(y_true: np.ndarray, 
                                    y_proba: np.ndarray) -> Dict[str, np.ndarray]:
    """Calculate precision-recall curve data.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        
    Returns:
        Dictionary with precision, recall, and thresholds
    """
    # Extract positive class probabilities if 2D
    if y_proba.ndim == 2:
        y_proba = y_proba[:, 1]
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    
    return {
        'precision': precision,
        'recall': recall,
        'thresholds': thresholds,
        'avg_precision': average_precision_score(y_true, y_proba)
    }


def find_optimal_threshold(y_true: np.ndarray,
                           y_proba: np.ndarray,
                           metric: str = 'f1') -> Tuple[float, float]:
    """Find optimal classification threshold.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        metric: Metric to optimize ('f1', 'accuracy', 'balanced')
        
    Returns:
        Tuple of (optimal threshold, metric value)
    """
    # Extract positive class probabilities if 2D
    if y_proba.ndim == 2:
        y_proba = y_proba[:, 1]
    
    thresholds = np.arange(0.1, 0.9, 0.01)
    scores = []
    
    for threshold in thresholds:
        y_pred_threshold = (y_proba >= threshold).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred_threshold, zero_division=0)
        elif metric == 'accuracy':
            score = accuracy_score(y_true, y_pred_threshold)
        elif metric == 'balanced':
            # Balance between precision and recall
            prec = precision_score(y_true, y_pred_threshold, zero_division=0)
            rec = recall_score(y_true, y_pred_threshold, zero_division=0)
            score = 2 * (prec * rec) / (prec + rec + 1e-8)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        scores.append(score)
    
    optimal_idx = np.argmax(scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_score = scores[optimal_idx]
    
    logger.info(f"Optimal threshold: {optimal_threshold:.3f} ({metric}: {optimal_score:.4f})")
    
    return optimal_threshold, optimal_score


def compare_models(results: Dict[str, Dict]) -> pd.DataFrame:
    """Compare multiple models' performance.
    
    Args:
        results: Dictionary with model names as keys and metrics as values
        
    Returns:
        DataFrame with comparison of all models
    """
    comparison_data = []
    
    for model_name, metrics in results.items():
        row = {
            'Model': model_name,
            'Accuracy': metrics.get('accuracy', np.nan),
            'Precision': metrics.get('precision', np.nan),
            'Recall': metrics.get('recall', np.nan),
            'F1-Score': metrics.get('f1_score', np.nan),
            'ROC-AUC': metrics.get('roc_auc', np.nan),
            'Specificity': metrics.get('specificity', np.nan)
        }
        comparison_data.append(row)
    
    df_comparison = pd.DataFrame(comparison_data)
    df_comparison = df_comparison.sort_values('F1-Score', ascending=False)
    
    return df_comparison
