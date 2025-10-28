"""
Random Forest model for match prediction.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Any, Optional
import logging

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class RandomForestModel(BaseModel):
    """Random Forest classifier for match outcome prediction."""
    
    def __init__(self, **kwargs):
        """Initialize Random Forest model.
        
        Args:
            **kwargs: Model parameters (n_estimators, max_depth, etc.)
        """
        super().__init__(model_name="RandomForest", **kwargs)
        self.build_model()
    
    def build_model(self):
        """Build Random Forest model."""
        default_params = {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1,
            'verbose': 0
        }
        
        # Update with provided parameters
        default_params.update(self.params)
        self.params = default_params
        
        self.model = RandomForestClassifier(**self.params)
        logger.info(f"Built {self.model_name} with params: {self.params}")
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Train the Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dictionary with training information
        """
        logger.info(f"Training {self.model_name}...")
        
        self.feature_names = X_train.columns.tolist()
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Get feature importance
        self.feature_importance_ = self.model.feature_importances_
        
        # Training metrics
        train_score = self.model.score(X_train, y_train)
        
        metrics = {
            'train_accuracy': train_score,
            'n_estimators': self.model.n_estimators,
            'n_features': self.model.n_features_in_,
            'oob_score': self.model.oob_score_ if hasattr(self.model, 'oob_score_') else None
        }
        
        # Validation score if provided
        if X_val is not None and y_val is not None:
            val_score = self.model.score(X_val, y_val)
            metrics['val_accuracy'] = val_score
            logger.info(f"Training completed - Train Acc: {train_score:.4f}, Val Acc: {val_score:.4f}")
        else:
            logger.info(f"Training completed - Train Accuracy: {train_score:.4f}")
        
        return metrics
