"""
Logistic Regression model for match prediction.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from typing import Dict, Any, Optional
import logging

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class LogisticRegressionModel(BaseModel):
    """Logistic Regression classifier for match outcome prediction."""
    
    def __init__(self, **kwargs):
        """Initialize Logistic Regression model.
        
        Args:
            **kwargs: Model parameters (C, penalty, solver, etc.)
        """
        super().__init__(model_name="LogisticRegression", **kwargs)
        self.build_model()
    
    def build_model(self):
        """Build Logistic Regression model."""
        default_params = {
            'C': 1.0,
            'penalty': 'l2',
            'solver': 'lbfgs',
            'max_iter': 1000,
            'class_weight': 'balanced',
            'random_state': 42
        }
        
        # Update with provided parameters
        default_params.update(self.params)
        self.params = default_params
        
        self.model = LogisticRegression(**self.params)
        logger.info(f"Built {self.model_name} with params: {self.params}")
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Train the Logistic Regression model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (not used for LR)
            y_val: Validation labels (not used for LR)
            
        Returns:
            Dictionary with training information
        """
        logger.info(f"Training {self.model_name}...")
        
        self.feature_names = X_train.columns.tolist()
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Get feature importance (coefficients)
        self.feature_importance_ = np.abs(self.model.coef_[0])
        
        # Training metrics
        train_score = self.model.score(X_train, y_train)
        
        metrics = {
            'train_accuracy': train_score,
            'n_iterations': self.model.n_iter_[0] if hasattr(self.model, 'n_iter_') else None,
            'converged': True
        }
        
        logger.info(f"Training completed - Train Accuracy: {train_score:.4f}")
        
        return metrics
