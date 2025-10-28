"""
Gradient Boosting models (XGBoost and LightGBM) for match prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging

from .base_model import BaseModel

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

logger = logging.getLogger(__name__)


class XGBoostModel(BaseModel):
    """XGBoost classifier for match outcome prediction."""
    
    def __init__(self, **kwargs):
        """Initialize XGBoost model.
        
        Args:
            **kwargs: Model parameters
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed. Install it with: pip install xgboost")
        
        super().__init__(model_name="XGBoost", **kwargs)
        self.build_model()
    
    def build_model(self):
        """Build XGBoost model."""
        default_params = {
            'learning_rate': 0.1,
            'n_estimators': 200,
            'max_depth': 7,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'logloss'
        }
        
        # Update with provided parameters
        default_params.update(self.params)
        self.params = default_params
        
        self.model = xgb.XGBClassifier(**self.params)
        logger.info(f"Built {self.model_name} with params: {self.params}")
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Train the XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional, for early stopping)
            y_val: Validation labels (optional, for early stopping)
            
        Returns:
            Dictionary with training information
        """
        logger.info(f"Training {self.model_name}...")
        
        self.feature_names = X_train.columns.tolist()
        
        # Prepare evaluation set for early stopping
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
        
        # Train model
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        self.is_fitted = True
        
        # Get feature importance
        self.feature_importance_ = self.model.feature_importances_
        
        # Training metrics
        train_score = self.model.score(X_train, y_train)
        
        metrics = {
            'train_accuracy': train_score,
            'n_estimators': self.model.n_estimators,
            'best_iteration': self.model.best_iteration if hasattr(self.model, 'best_iteration') else None
        }
        
        # Validation score if provided
        if X_val is not None and y_val is not None:
            val_score = self.model.score(X_val, y_val)
            metrics['val_accuracy'] = val_score
            logger.info(f"Training completed - Train Acc: {train_score:.4f}, Val Acc: {val_score:.4f}")
        else:
            logger.info(f"Training completed - Train Accuracy: {train_score:.4f}")
        
        return metrics


class LightGBMModel(BaseModel):
    """LightGBM classifier for match outcome prediction."""
    
    def __init__(self, **kwargs):
        """Initialize LightGBM model.
        
        Args:
            **kwargs: Model parameters
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not installed. Install it with: pip install lightgbm")
        
        super().__init__(model_name="LightGBM", **kwargs)
        self.build_model()
    
    def build_model(self):
        """Build LightGBM model."""
        default_params = {
            'learning_rate': 0.1,
            'n_estimators': 200,
            'max_depth': 7,
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        
        # Update with provided parameters
        default_params.update(self.params)
        self.params = default_params
        
        self.model = lgb.LGBMClassifier(**self.params)
        logger.info(f"Built {self.model_name} with params: {self.params}")
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Train the LightGBM model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional, for early stopping)
            y_val: Validation labels (optional, for early stopping)
            
        Returns:
            Dictionary with training information
        """
        logger.info(f"Training {self.model_name}...")
        
        self.feature_names = X_train.columns.tolist()
        
        # Prepare evaluation set for early stopping
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
        
        # Train model
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            eval_metric='logloss'
        )
        self.is_fitted = True
        
        # Get feature importance
        self.feature_importance_ = self.model.feature_importances_
        
        # Training metrics
        train_score = self.model.score(X_train, y_train)
        
        metrics = {
            'train_accuracy': train_score,
            'n_estimators': self.model.n_estimators,
            'best_iteration': self.model.best_iteration_ if hasattr(self.model, 'best_iteration_') else None
        }
        
        # Validation score if provided
        if X_val is not None and y_val is not None:
            val_score = self.model.score(X_val, y_val)
            metrics['val_accuracy'] = val_score
            logger.info(f"Training completed - Train Acc: {train_score:.4f}, Val Acc: {val_score:.4f}")
        else:
            logger.info(f"Training completed - Train Accuracy: {train_score:.4f}")
        
        return metrics
