"""
Base model class for League of Legends match prediction.
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional
import joblib
import json
import os
import logging

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base class for all prediction models."""
    
    def __init__(self, model_name: str, **kwargs):
        """Initialize base model.
        
        Args:
            model_name: Name of the model
            **kwargs: Additional model-specific parameters
        """
        self.model_name = model_name
        self.model = None
        self.params = kwargs
        self.is_fitted = False
        self.feature_names = None
        self.feature_importance_ = None
        
    @abstractmethod
    def build_model(self):
        """Build the model architecture."""
        pass
    
    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: Optional[pd.DataFrame] = None, 
              y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Train the model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            
        Returns:
            Dictionary with training metrics
        """
        pass
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Features for prediction
            
        Returns:
            Array of predictions (0 or 1)
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Features for prediction
            
        Returns:
            Array of probabilities for each class
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise NotImplementedError(f"{self.model_name} does not support probability prediction")
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance scores.
        
        Returns:
            DataFrame with feature names and importance scores
        """
        if self.feature_importance_ is None:
            return None
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.feature_importance_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, file_path: str):
        """Save model to disk.
        
        Args:
            file_path: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save untrained model")
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save model
        model_file = file_path
        joblib.dump(self.model, model_file)
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'params': self.params,
            'feature_names': self.feature_names if self.feature_names is not None else [],
            'is_fitted': self.is_fitted
        }
        
        metadata_file = file_path.replace('.pkl', '_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {model_file}")
        logger.info(f"Metadata saved to {metadata_file}")
    
    def load_model(self, file_path: str):
        """Load model from disk.
        
        Args:
            file_path: Path to the saved model
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        # Load model
        self.model = joblib.load(file_path)
        
        # Load metadata
        metadata_file = file_path.replace('.pkl', '_metadata.json')
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            self.model_name = metadata.get('model_name', self.model_name)
            self.params = metadata.get('params', {})
            self.feature_names = metadata.get('feature_names', None)
            self.is_fitted = metadata.get('is_fitted', True)
        
        logger.info(f"Model loaded from {file_path}")
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters.
        
        Returns:
            Dictionary of model parameters
        """
        return self.params.copy()
    
    def set_params(self, **params):
        """Set model parameters.
        
        Args:
            **params: Parameters to update
        """
        self.params.update(params)
        
    def __repr__(self) -> str:
        """String representation of the model."""
        status = "fitted" if self.is_fitted else "not fitted"
        return f"{self.model_name} ({status})"
    
    def summary(self) -> Dict[str, Any]:
        """Get model summary.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'is_fitted': self.is_fitted,
            'n_features': len(self.feature_names) if self.feature_names else None,
            'params': self.params
        }
