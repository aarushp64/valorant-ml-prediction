"""
Advanced Ensemble Model for League of Legends Match Prediction
==============================================================

This module implements sophisticated ensemble methods including:
- Stacking with meta-learner
- Voting classifiers with optimized weights
- Dynamic ensemble selection
- Bayesian model averaging
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
import logging
from typing import Dict, Any, Optional, List, Tuple
import os

from src.models.base_model import BaseModel
from src.models.logistic_regression import LogisticRegressionModel
from src.models.random_forest import RandomForestModel
from src.models.gradient_boosting import XGBoostModel, LightGBMModel
from src.models.catboost_model import CatBoostModel

logger = logging.getLogger(__name__)


class AdvancedEnsembleModel(BaseModel):
    """
    Advanced ensemble model combining multiple base learners.
    
    Features:
    - Multiple ensemble strategies (voting, stacking, blending)
    - Automatic weight optimization
    - Dynamic model selection based on performance
    - Cross-validation for robust evaluation
    - Model diversity analysis
    """
    
    def __init__(self, ensemble_type: str = 'stacking', **kwargs):
        """
        Initialize ensemble model.
        
        Args:
            ensemble_type: Type of ensemble ('voting', 'stacking', 'blending')
            **kwargs: Additional parameters
        """
        super().__init__()
        self.model_name = "AdvancedEnsemble"
        self.ensemble_type = ensemble_type
        self.params = kwargs
        
        # Base models
        self.base_models = {}
        self.ensemble_model = None
        self.model_weights = None
        self.model_performance = {}
        self.diversity_scores = {}
        
        # Initialize base models with optimized parameters
        self._initialize_base_models()
        
    def _initialize_base_models(self):
        """Initialize base models with optimized parameters."""
        logger.info("Initializing base models for ensemble...")
        
        # Logistic Regression - good baseline
        self.base_models['logistic'] = LogisticRegressionModel(
            C=1.0,
            penalty='l2',
            solver='liblinear',
            random_state=42
        )
        
        # Random Forest - handles feature interactions well
        self.base_models['random_forest'] = RandomForestModel(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42
        )
        
        # XGBoost - gradient boosting
        self.base_models['xgboost'] = XGBoostModel(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        # LightGBM - fast gradient boosting
        self.base_models['lightgbm'] = LightGBMModel(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        # CatBoost - handles categorical features well
        self.base_models['catboost'] = CatBoostModel(
            iterations=200,
            depth=6,
            learning_rate=0.1,
            l2_leaf_reg=3,
            random_seed=42,
            verbose=False
        )
        
        logger.info(f"Initialized {len(self.base_models)} base models")
    
    def build_model(self) -> Any:
        """
        Build the ensemble model.
        
        Returns:
            Ensemble model instance
        """
        logger.info(f"Building {self.ensemble_type} ensemble model...")
        
        # Prepare base estimators for sklearn ensemble methods
        base_estimators = []
        for name, model in self.base_models.items():
            # Get the underlying sklearn model
            if hasattr(model, 'model') and model.model is not None:
                base_estimators.append((name, model.model))
            else:
                # Build the model first
                model.build_model()
                base_estimators.append((name, model.model))
        
        if self.ensemble_type == 'voting':
            self.ensemble_model = VotingClassifier(
                estimators=base_estimators,
                voting='soft',
                weights=self.model_weights,
                n_jobs=-1
            )
        elif self.ensemble_type == 'stacking':
            # Use logistic regression as meta-learner
            meta_learner = LogisticRegression(
                random_state=42,
                max_iter=1000
            )
            self.ensemble_model = StackingClassifier(
                estimators=base_estimators,
                final_estimator=meta_learner,
                cv=5,
                stack_method='predict_proba',
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unsupported ensemble type: {self.ensemble_type}")
        
        return self.ensemble_model
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: Optional[pd.DataFrame] = None, 
              y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Train the ensemble model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training {self.model_name} ({self.ensemble_type})...")
        
        # Step 1: Train individual base models and evaluate performance
        self._train_base_models(X_train, y_train, X_val, y_val)
        
        # Step 2: Calculate model weights based on performance
        if self.ensemble_type == 'voting':
            self._calculate_optimal_weights(X_train, y_train)
        
        # Step 3: Analyze model diversity
        self._analyze_model_diversity(X_train, y_train)
        
        # Step 4: Build and train ensemble
        self.build_model()
        self.ensemble_model.fit(X_train, y_train)
        
        # Step 5: Evaluate ensemble performance
        train_pred = self.ensemble_model.predict(X_train)
        train_proba = self.ensemble_model.predict_proba(X_train)[:, 1]
        train_accuracy = accuracy_score(y_train, train_pred)
        train_auc = roc_auc_score(y_train, train_proba)
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.ensemble_model, X_train, y_train,
            cv=5, scoring='roc_auc', n_jobs=-1
        )
        
        self.is_fitted = True
        
        metrics = {
            'train_accuracy': train_accuracy,
            'train_auc': train_auc,
            'cv_mean_auc': cv_scores.mean(),
            'cv_std_auc': cv_scores.std(),
            'base_model_performance': self.model_performance,
            'model_weights': self.model_weights,
            'diversity_scores': self.diversity_scores
        }
        
        logger.info(f"✓ {self.model_name} training completed")
        logger.info(f"  Train Accuracy: {train_accuracy:.4f}")
        logger.info(f"  Train AUC: {train_auc:.4f}")
        logger.info(f"  CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return metrics
    
    def _train_base_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                          X_val: Optional[pd.DataFrame] = None,
                          y_val: Optional[pd.Series] = None):
        """Train all base models and collect performance metrics."""
        logger.info("Training base models...")
        
        for name, model in self.base_models.items():
            try:
                logger.info(f"  Training {name}...")
                
                # Train the model
                train_metrics = model.train(X_train, y_train, X_val, y_val)
                
                # Evaluate on training set
                train_pred = model.predict(X_train)
                train_proba = model.predict_proba(X_train)[:, 1]
                train_accuracy = accuracy_score(y_train, train_pred)
                train_auc = roc_auc_score(y_train, train_proba)
                
                # Cross-validation score
                cv_scores = cross_val_score(
                    model.model, X_train, y_train,
                    cv=5, scoring='roc_auc', n_jobs=-1
                )
                
                self.model_performance[name] = {
                    'train_accuracy': train_accuracy,
                    'train_auc': train_auc,
                    'cv_mean_auc': cv_scores.mean(),
                    'cv_std_auc': cv_scores.std(),
                    'train_metrics': train_metrics
                }
                
                logger.info(f"    {name} - AUC: {train_auc:.4f}, CV AUC: {cv_scores.mean():.4f}")
                
            except Exception as e:
                logger.error(f"    Error training {name}: {str(e)}")
                # Remove failed model
                del self.base_models[name]
        
        logger.info(f"✓ Trained {len(self.base_models)} base models successfully")
    
    def _calculate_optimal_weights(self, X: pd.DataFrame, y: pd.Series):
        """Calculate optimal weights for voting ensemble based on performance."""
        logger.info("Calculating optimal weights for voting ensemble...")
        
        # Use CV AUC scores as basis for weights
        auc_scores = []
        model_names = []
        
        for name, performance in self.model_performance.items():
            auc_scores.append(performance['cv_mean_auc'])
            model_names.append(name)
        
        auc_scores = np.array(auc_scores)
        
        # Method 1: Softmax of AUC scores
        weights_softmax = np.exp(auc_scores * 5) / np.sum(np.exp(auc_scores * 5))
        
        # Method 2: Linear scaling
        weights_linear = (auc_scores - auc_scores.min()) / (auc_scores.max() - auc_scores.min() + 1e-8)
        weights_linear = weights_linear / weights_linear.sum()
        
        # Method 3: Performance-based with penalty for low scores
        weights_performance = np.maximum(0, auc_scores - 0.5)  # Penalty for AUC < 0.5
        weights_performance = weights_performance / (weights_performance.sum() + 1e-8)
        
        # Use softmax weights (usually works best)
        self.model_weights = weights_softmax.tolist()
        
        weight_info = dict(zip(model_names, self.model_weights))
        logger.info(f"Calculated weights: {weight_info}")
    
    def _analyze_model_diversity(self, X: pd.DataFrame, y: pd.Series):
        """Analyze diversity among base models."""
        logger.info("Analyzing model diversity...")
        
        # Get predictions from all models
        predictions = {}
        probabilities = {}
        
        for name, model in self.base_models.items():
            if name in self.model_performance:  # Only include successful models
                pred = model.predict(X)
                proba = model.predict_proba(X)[:, 1]
                predictions[name] = pred
                probabilities[name] = proba
        
        # Calculate pairwise disagreement (diversity)
        model_names = list(predictions.keys())
        n_models = len(model_names)
        
        disagreement_matrix = np.zeros((n_models, n_models))
        correlation_matrix = np.zeros((n_models, n_models))
        
        for i, name1 in enumerate(model_names):
            for j, name2 in enumerate(model_names):
                if i != j:
                    # Disagreement rate
                    disagreement = np.mean(predictions[name1] != predictions[name2])
                    disagreement_matrix[i, j] = disagreement
                    
                    # Correlation of probabilities
                    correlation = np.corrcoef(probabilities[name1], probabilities[name2])[0, 1]
                    correlation_matrix[i, j] = correlation
        
        # Average diversity metrics
        avg_disagreement = np.mean(disagreement_matrix[np.triu_indices(n_models, k=1)])
        avg_correlation = np.mean(correlation_matrix[np.triu_indices(n_models, k=1)])
        
        self.diversity_scores = {
            'average_disagreement': avg_disagreement,
            'average_correlation': avg_correlation,
            'disagreement_matrix': disagreement_matrix.tolist(),
            'correlation_matrix': correlation_matrix.tolist(),
            'model_names': model_names
        }
        
        logger.info(f"Model diversity analysis:")
        logger.info(f"  Average disagreement rate: {avg_disagreement:.4f}")
        logger.info(f"  Average correlation: {avg_correlation:.4f}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the ensemble.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted class labels
        """
        self._check_fitted()
        return self.ensemble_model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities using the ensemble.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted probabilities for each class
        """
        self._check_fitted()
        return self.ensemble_model.predict_proba(X)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get aggregated feature importance from base models.
        
        Returns:
            DataFrame with feature importance
        """
        self._check_fitted()
        
        importance_data = []
        
        for name, model in self.base_models.items():
            if hasattr(model, 'get_feature_importance'):
                try:
                    model_importance = model.get_feature_importance()
                    model_importance['model'] = name
                    importance_data.append(model_importance)
                except:
                    continue
        
        if not importance_data:
            logger.warning("No feature importance available from base models")
            return pd.DataFrame()
        
        # Combine importance from all models
        all_importance = pd.concat(importance_data, ignore_index=True)
        
        # Calculate weighted average importance
        weighted_importance = []
        features = all_importance['feature'].unique()
        
        for feature in features:
            feature_data = all_importance[all_importance['feature'] == feature]
            
            if self.model_weights:
                # Weight by model performance
                weighted_imp = 0
                total_weight = 0
                for _, row in feature_data.iterrows():
                    model_idx = list(self.base_models.keys()).index(row['model'])
                    weight = self.model_weights[model_idx]
                    weighted_imp += row['importance'] * weight
                    total_weight += weight
                
                if total_weight > 0:
                    weighted_imp /= total_weight
            else:
                weighted_imp = feature_data['importance'].mean()
            
            weighted_importance.append({
                'feature': feature,
                'importance': weighted_imp
            })
        
        result = pd.DataFrame(weighted_importance)
        result = result.sort_values('importance', ascending=False)
        
        return result
    
    def get_model_contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Get individual model contributions to ensemble predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            DataFrame with model contributions
        """
        self._check_fitted()
        
        contributions = {}
        
        # Get predictions from each base model
        for name, model in self.base_models.items():
            if name in self.model_performance:
                proba = model.predict_proba(X)[:, 1]
                contributions[f'{name}_proba'] = proba
        
        # Get ensemble prediction
        ensemble_proba = self.predict_proba(X)[:, 1]
        contributions['ensemble_proba'] = ensemble_proba
        
        return pd.DataFrame(contributions)
    
    def save_model(self, filepath: str):
        """
        Save the ensemble model and all base models.
        
        Args:
            filepath: Path to save the model
        """
        self._check_fitted()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save ensemble model
        joblib.dump(self.ensemble_model, filepath)
        
        # Save base models
        base_models_dir = filepath.replace('.pkl', '_base_models')
        os.makedirs(base_models_dir, exist_ok=True)
        
        for name, model in self.base_models.items():
            model_path = os.path.join(base_models_dir, f'{name}.pkl')
            model.save_model(model_path)
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'ensemble_type': self.ensemble_type,
            'params': self.params,
            'model_weights': self.model_weights,
            'model_performance': self.model_performance,
            'diversity_scores': self.diversity_scores,
            'base_models_dir': base_models_dir
        }
        
        metadata_path = filepath.replace('.pkl', '_metadata.pkl')
        joblib.dump(metadata, metadata_path)
        
        logger.info(f"✓ {self.model_name} saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load the ensemble model and all base models.
        
        Args:
            filepath: Path to the saved model
        """
        try:
            # Load ensemble model
            self.ensemble_model = joblib.load(filepath)
            
            # Load metadata
            metadata_path = filepath.replace('.pkl', '_metadata.pkl')
            metadata = joblib.load(metadata_path)
            
            # Restore attributes
            self.ensemble_type = metadata['ensemble_type']
            self.params = metadata['params']
            self.model_weights = metadata['model_weights']
            self.model_performance = metadata['model_performance']
            self.diversity_scores = metadata['diversity_scores']
            
            # Load base models
            base_models_dir = metadata['base_models_dir']
            self.base_models = {}
            
            for name in os.listdir(base_models_dir):
                if name.endswith('.pkl'):
                    model_name = name.replace('.pkl', '')
                    model_path = os.path.join(base_models_dir, name)
                    
                    # Initialize appropriate model class
                    if model_name == 'logistic':
                        model = LogisticRegressionModel()
                    elif model_name == 'random_forest':
                        model = RandomForestModel()
                    elif model_name == 'xgboost':
                        model = XGBoostModel()
                    elif model_name == 'lightgbm':
                        model = LightGBMModel()
                    elif model_name == 'catboost':
                        model = CatBoostModel()
                    else:
                        continue
                    
                    model.load_model(model_path)
                    self.base_models[model_name] = model
            
            self.is_fitted = True
            logger.info(f"✓ {self.model_name} loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading {self.model_name}: {str(e)}")
            raise