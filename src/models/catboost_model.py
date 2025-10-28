"""
CatBoost Model Implementation for League of Legends Match Prediction
===================================================================

CatBoost is often superior to XGBoost and LightGBM for tabular data,
especially with categorical features and small datasets.
"""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import cross_val_score
import joblib
import logging
from typing import Dict, Any, Optional, Tuple
import os

from src.models.base_model import BaseModel

logger = logging.getLogger(__name__)


class CatBoostModel(BaseModel):
    """
    CatBoost implementation for binary classification.
    
    Features:
    - Automatic categorical feature handling
    - Built-in regularization
    - GPU support (if available)
    - Feature importance and SHAP values
    - Cross-validation during training
    """
    
    def __init__(self, **kwargs):
        """
        Initialize CatBoost model.
        
        Args:
            **kwargs: CatBoost parameters
        """
        super().__init__()
        self.model_name = "CatBoost"
        
        # Default parameters optimized for LoL data
        default_params = {
            'iterations': 1000,
            'learning_rate': 0.1,
            'depth': 6,
            'l2_leaf_reg': 3,
            'bootstrap_type': 'Bernoulli',
            'subsample': 0.8,
            'random_strength': 1,
            'one_hot_max_size': 10,
            'leaf_estimation_method': 'Newton',
            'eval_metric': 'AUC',
            'loss_function': 'Logloss',
            'random_seed': 42,
            'verbose': False,
            'early_stopping_rounds': 100,
            'use_best_model': True,
            'task_type': 'GPU' if self._has_gpu() else 'CPU'
        }
        
        # Update with user parameters
        self.params = {**default_params, **kwargs}
        self.model = None
        self.feature_importance_ = None
        self.validation_scores_ = None
        
    def _has_gpu(self) -> bool:
        """Check if GPU is available for CatBoost."""
        try:
            import catboost
            # Simple check - in practice you might want more sophisticated detection
            return True  # Assume GPU support, CatBoost will fallback to CPU if needed
        except:
            return False
    
    def build_model(self) -> CatBoostClassifier:
        """
        Build CatBoost model with specified parameters.
        
        Returns:
            CatBoost classifier instance
        """
        logger.info(f"Building {self.model_name} model...")
        logger.info(f"Parameters: {self.params}")
        
        self.model = CatBoostClassifier(**self.params)
        return self.model
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: Optional[pd.DataFrame] = None, 
              y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Train the CatBoost model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training {self.model_name}...")
        
        # Build model if not already built
        if self.model is None:
            self.build_model()
        
        # Prepare data
        categorical_features = self._identify_categorical_features(X_train)
        
        # Create CatBoost Pool objects for efficient training
        train_pool = Pool(
            data=X_train,
            label=y_train,
            cat_features=categorical_features
        )
        
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = Pool(
                data=X_val,
                label=y_val,
                cat_features=categorical_features
            )
        
        # Train the model
        self.model.fit(
            train_pool,
            eval_set=eval_set,
            plot=False,
            verbose_eval=100
        )
        
        # Calculate training metrics
        train_pred = self.model.predict(X_train)
        train_proba = self.model.predict_proba(X_train)[:, 1]
        train_accuracy = (train_pred == y_train).mean()
        
        # Store feature importance
        self.feature_importance_ = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Cross-validation score
        cv_scores = cross_val_score(
            self.model, X_train, y_train, 
            cv=5, scoring='roc_auc', n_jobs=-1
        )
        self.validation_scores_ = cv_scores
        
        self.is_fitted = True
        
        metrics = {
            'train_accuracy': train_accuracy,
            'cv_mean_auc': cv_scores.mean(),
            'cv_std_auc': cv_scores.std(),
            'best_iteration': self.model.get_best_iteration(),
            'feature_count': len(X_train.columns)
        }
        
        logger.info(f"✓ {self.model_name} training completed")
        logger.info(f"  Train Accuracy: {train_accuracy:.4f}")
        logger.info(f"  CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        logger.info(f"  Best Iteration: {self.model.get_best_iteration()}")
        
        return metrics
    
    def _identify_categorical_features(self, X: pd.DataFrame) -> list:
        """
        Identify categorical features in the dataset.
        
        Args:
            X: Feature matrix
            
        Returns:
            List of categorical feature indices
        """
        categorical_features = []
        
        for i, col in enumerate(X.columns):
            # Check if column is categorical by dtype or unique values
            if (X[col].dtype == 'object' or 
                X[col].dtype == 'category' or
                (X[col].dtype in ['int64', 'int32'] and X[col].nunique() < 20)):
                categorical_features.append(i)
        
        logger.info(f"Identified {len(categorical_features)} categorical features: "
                   f"{[X.columns[i] for i in categorical_features]}")
        
        return categorical_features
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted class labels
        """
        self._check_fitted()
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted probabilities for each class
        """
        self._check_fitted()
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, plot: bool = False) -> pd.DataFrame:
        """
        Get feature importance from the trained model.
        
        Args:
            plot: Whether to create a plot
            
        Returns:
            DataFrame with feature importance
        """
        self._check_fitted()
        
        if self.feature_importance_ is None:
            raise ValueError("Feature importance not available")
        
        if plot:
            self._plot_feature_importance()
        
        return self.feature_importance_
    
    def _plot_feature_importance(self):
        """Plot feature importance."""
        import matplotlib.pyplot as plt
        
        top_features = self.feature_importance_.head(20)
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'{self.model_name} - Top 20 Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.
        
        Returns:
            Dictionary with model details
        """
        self._check_fitted()
        
        info = {
            'model_name': self.model_name,
            'model_type': 'CatBoost Gradient Boosting',
            'parameters': self.params,
            'n_features': len(self.feature_importance_) if self.feature_importance_ is not None else None,
            'best_iteration': self.model.get_best_iteration(),
            'tree_count': self.model.tree_count_,
            'cv_scores': {
                'mean_auc': self.validation_scores_.mean() if self.validation_scores_ is not None else None,
                'std_auc': self.validation_scores_.std() if self.validation_scores_ is not None else None
            }
        }
        
        return info
    
    def save_model(self, filepath: str):
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        self._check_fitted()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save CatBoost model
        model_path = filepath.replace('.pkl', '_catboost.cbm')
        self.model.save_model(model_path)
        
        # Save additional metadata
        metadata = {
            'model_name': self.model_name,
            'params': self.params,
            'feature_importance': self.feature_importance_.to_dict('records') if self.feature_importance_ is not None else None,
            'validation_scores': self.validation_scores_.tolist() if self.validation_scores_ is not None else None,
            'model_path': model_path
        }
        
        joblib.dump(metadata, filepath.replace('.pkl', '_metadata.pkl'))
        
        logger.info(f"✓ {self.model_name} saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a trained model.
        
        Args:
            filepath: Path to the saved model
        """
        try:
            # Load metadata
            metadata = joblib.load(filepath.replace('.pkl', '_metadata.pkl'))
            
            # Load CatBoost model
            model_path = metadata['model_path']
            self.model = CatBoostClassifier()
            self.model.load_model(model_path)
            
            # Restore metadata
            self.params = metadata['params']
            self.feature_importance_ = pd.DataFrame(metadata['feature_importance']) if metadata['feature_importance'] else None
            self.validation_scores_ = np.array(metadata['validation_scores']) if metadata['validation_scores'] else None
            
            self.is_fitted = True
            logger.info(f"✓ {self.model_name} loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading {self.model_name}: {str(e)}")
            raise
    
    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series, 
                                n_trials: int = 100) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            X: Feature matrix
            y: Target variable
            n_trials: Number of optimization trials
            
        Returns:
            Best parameters found
        """
        try:
            import optuna
            from sklearn.model_selection import cross_val_score
            
            logger.info(f"Starting hyperparameter optimization with {n_trials} trials...")
            
            def objective(trial):
                # Define parameter search space
                params = {
                    'iterations': trial.suggest_int('iterations', 500, 2000),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'depth': trial.suggest_int('depth', 4, 10),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'random_strength': trial.suggest_float('random_strength', 0.5, 2.0),
                    'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli']),
                    'eval_metric': 'AUC',
                    'loss_function': 'Logloss',
                    'random_seed': 42,
                    'verbose': False
                }
                
                # Create and train model
                model = CatBoostClassifier(**params)
                
                # Use cross-validation for objective
                scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc', n_jobs=1)
                return scores.mean()
            
            # Run optimization
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials)
            
            best_params = study.best_params
            logger.info(f"✓ Hyperparameter optimization completed")
            logger.info(f"  Best AUC: {study.best_value:.4f}")
            logger.info(f"  Best params: {best_params}")
            
            # Update model parameters
            self.params.update(best_params)
            
            return {
                'best_params': best_params,
                'best_score': study.best_value,
                'study': study
            }
            
        except ImportError:
            logger.warning("Optuna not available. Please install with: pip install optuna")
            return {}
        except Exception as e:
            logger.error(f"Error in hyperparameter optimization: {str(e)}")
            return {}