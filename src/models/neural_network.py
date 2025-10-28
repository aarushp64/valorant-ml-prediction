"""
Neural Network model for match prediction using TensorFlow/Keras.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging

from .base_model import BaseModel

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)


class NeuralNetworkModel(BaseModel):
    """Neural Network classifier for match outcome prediction."""
    
    def __init__(self, **kwargs):
        """Initialize Neural Network model.
        
        Args:
            **kwargs: Model parameters
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is not installed. Install it with: pip install tensorflow")
        
        super().__init__(model_name="NeuralNetwork", **kwargs)
        self.history = None
        self.build_model()
    
    def build_model(self):
        """Build Neural Network model."""
        default_params = {
            'layers': [128, 64, 32, 16],
            'activation': 'relu',
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'early_stopping_patience': 10,
            'validation_split': 0.1
        }
        
        # Update with provided parameters
        default_params.update(self.params)
        self.params = default_params
        
        logger.info(f"Built {self.model_name} architecture ready for training")
    
    def _create_model(self, input_dim: int):
        """Create the Keras model architecture.
        
        Args:
            input_dim: Number of input features
        """
        model = keras.Sequential(name=self.model_name)
        
        # Input layer
        model.add(layers.Input(shape=(input_dim,)))
        
        # Hidden layers
        for i, units in enumerate(self.params['layers']):
            model.add(layers.Dense(
                units, 
                activation=self.params['activation'],
                name=f'dense_{i+1}'
            ))
            model.add(layers.Dropout(
                self.params['dropout_rate'],
                name=f'dropout_{i+1}'
            ))
        
        # Output layer
        model.add(layers.Dense(1, activation='sigmoid', name='output'))
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=self.params['learning_rate'])
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        return model
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Train the Neural Network model.
        
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
        input_dim = X_train.shape[1]
        
        # Create model
        self.model = self._create_model(input_dim)
        
        # Callbacks
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=self.params['early_stopping_patience'],
                restore_best_weights=True,
                verbose=0
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=0
            )
        ]
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
            validation_split = 0
        else:
            validation_split = self.params['validation_split']
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=self.params['batch_size'],
            epochs=self.params['epochs'],
            validation_data=validation_data,
            validation_split=validation_split,
            callbacks=callback_list,
            verbose=0
        )
        
        self.is_fitted = True
        
        # Get training metrics
        final_epoch = len(self.history.history['loss'])
        train_loss = self.history.history['loss'][-1]
        train_acc = self.history.history['accuracy'][-1]
        
        metrics = {
            'train_accuracy': train_acc,
            'train_loss': train_loss,
            'epochs_trained': final_epoch
        }
        
        if 'val_loss' in self.history.history:
            val_loss = self.history.history['val_loss'][-1]
            val_acc = self.history.history['val_accuracy'][-1]
            metrics['val_accuracy'] = val_acc
            metrics['val_loss'] = val_loss
            logger.info(f"Training completed - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        else:
            logger.info(f"Training completed - Train Accuracy: {train_acc:.4f}")
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Features for prediction
            
        Returns:
            Array of predictions (0 or 1)
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        probabilities = self.model.predict(X, verbose=0)
        predictions = (probabilities > 0.5).astype(int).flatten()
        
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Features for prediction
            
        Returns:
            Array of probabilities for each class
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        probabilities = self.model.predict(X, verbose=0).flatten()
        
        # Return probabilities for both classes
        proba_both_classes = np.column_stack([1 - probabilities, probabilities])
        
        return proba_both_classes
    
    def get_training_history(self) -> Dict[str, list]:
        """Get training history.
        
        Returns:
            Dictionary with training history metrics
        """
        if self.history is None:
            return {}
        
        return self.history.history
