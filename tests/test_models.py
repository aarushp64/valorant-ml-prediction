"""
Tests for model implementations.
"""

import pytest
import numpy as np
import pandas as pd
from src.models.logistic_regression import LogisticRegressionModel
from src.models.random_forest import RandomForestModel


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    n_samples = 100
    n_features = 8
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y = pd.Series(np.random.randint(0, 2, n_samples))
    
    return X, y


def test_logistic_regression_initialization():
    """Test Logistic Regression model initialization."""
    model = LogisticRegressionModel()
    assert model.model_name == "LogisticRegression"
    assert model.is_fitted == False


def test_logistic_regression_training(sample_data):
    """Test Logistic Regression model training."""
    X, y = sample_data
    model = LogisticRegressionModel()
    
    metrics = model.train(X, y)
    
    assert model.is_fitted == True
    assert 'train_accuracy' in metrics
    assert metrics['train_accuracy'] >= 0 and metrics['train_accuracy'] <= 1


def test_logistic_regression_prediction(sample_data):
    """Test Logistic Regression model prediction."""
    X, y = sample_data
    model = LogisticRegressionModel()
    model.train(X, y)
    
    predictions = model.predict(X)
    
    assert len(predictions) == len(X)
    assert all(p in [0, 1] for p in predictions)


def test_random_forest_initialization():
    """Test Random Forest model initialization."""
    model = RandomForestModel(n_estimators=50)
    assert model.model_name == "RandomForest"
    assert model.params['n_estimators'] == 50


def test_random_forest_training(sample_data):
    """Test Random Forest model training."""
    X, y = sample_data
    model = RandomForestModel(n_estimators=10)
    
    metrics = model.train(X, y)
    
    assert model.is_fitted == True
    assert 'train_accuracy' in metrics


def test_random_forest_feature_importance(sample_data):
    """Test Random Forest feature importance."""
    X, y = sample_data
    model = RandomForestModel(n_estimators=10)
    model.train(X, y)
    
    importance_df = model.get_feature_importance()
    
    assert importance_df is not None
    assert len(importance_df) == X.shape[1]
    assert 'feature' in importance_df.columns
    assert 'importance' in importance_df.columns


def test_model_save_load(sample_data, tmp_path):
    """Test model saving and loading."""
    X, y = sample_data
    model = LogisticRegressionModel()
    model.train(X, y)
    
    # Save model
    model_path = tmp_path / "test_model.pkl"
    model.save_model(str(model_path))
    
    # Load model
    new_model = LogisticRegressionModel()
    new_model.load_model(str(model_path))
    
    assert new_model.is_fitted == True
    assert new_model.feature_names == model.feature_names
    
    # Test predictions match
    pred1 = model.predict(X)
    pred2 = new_model.predict(X)
    
    assert np.array_equal(pred1, pred2)
