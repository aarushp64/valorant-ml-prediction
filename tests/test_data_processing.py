"""
Tests for data processing modules.
"""

import pytest
import pandas as pd
import numpy as np
from src.data.data_loader import DataLoader
from src.data.preprocessing import DataPreprocessor


@pytest.fixture
def sample_dataframe():
    """Generate sample dataframe for testing."""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'win': np.random.randint(0, 2, n_samples),
        'kills': np.random.randint(0, 20, n_samples),
        'deaths': np.random.randint(0, 15, n_samples),
        'assists': np.random.randint(0, 25, n_samples),
        'gold_earned': np.random.randint(5000, 20000, n_samples),
        'cs': np.random.randint(50, 300, n_samples),
        'wards_placed': np.random.randint(0, 40, n_samples),
        'wards_killed': np.random.randint(0, 20, n_samples),
        'damage_dealt': np.random.randint(10000, 50000, n_samples)
    }
    
    return pd.DataFrame(data)


def test_data_loader_validation(sample_dataframe):
    """Test data validation."""
    loader = DataLoader()
    is_valid, issues = loader.validate_data(sample_dataframe)
    
    assert isinstance(is_valid, bool)
    assert isinstance(issues, list)


def test_data_loader_info(sample_dataframe):
    """Test data info extraction."""
    loader = DataLoader()
    info = loader.get_data_info(sample_dataframe)
    
    assert 'n_samples' in info
    assert 'n_features' in info
    assert info['n_samples'] == len(sample_dataframe)


def test_preprocessor_derived_features(sample_dataframe):
    """Test derived feature creation."""
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.create_derived_features(sample_dataframe)
    
    # Check new features are created
    assert 'kda_ratio' in df_processed.columns
    assert 'gold_per_cs' in df_processed.columns
    assert 'vision_score' in df_processed.columns
    assert 'damage_efficiency' in df_processed.columns
    
    # Check no NaN values
    assert not df_processed[['kda_ratio', 'gold_per_cs']].isnull().any().any()


def test_preprocessor_data_split(sample_dataframe):
    """Test data splitting."""
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.create_derived_features(sample_dataframe)
    
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(df_processed)
    
    # Check splits exist
    assert X_train is not None
    assert X_test is not None
    assert y_train is not None
    assert y_test is not None
    
    # Check no data leakage
    assert len(set(X_train.index) & set(X_test.index)) == 0


def test_preprocessor_scaling(sample_dataframe):
    """Test feature scaling."""
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.create_derived_features(sample_dataframe)
    
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(df_processed)
    
    X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
    
    # Check shapes preserved
    assert X_train_scaled.shape == X_train.shape
    assert X_test_scaled.shape == X_test.shape
    
    # Check scaling worked (mean ~ 0, std ~ 1 for standard scaler)
    assert abs(X_train_scaled.mean().mean()) < 0.1
    assert abs(X_train_scaled.std().mean() - 1.0) < 0.5


def test_preprocessor_pipeline(sample_dataframe):
    """Test full preprocessing pipeline."""
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.preprocess_pipeline(sample_dataframe)
    
    # Check processing completed
    assert df_processed is not None
    assert len(df_processed) == len(sample_dataframe)
    
    # Check target column preserved
    assert 'win' in df_processed.columns
