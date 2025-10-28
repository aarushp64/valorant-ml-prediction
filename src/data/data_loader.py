"""
Data loading utilities for League of Legends match prediction system.
"""

import os
import pandas as pd
import yaml
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Handle data loading and initial validation."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize DataLoader with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_config = self.config['data']
        self.raw_data_path = self.data_config['raw_data_path']
        self.processed_data_path = self.data_config['processed_data_path']
        
    def load_raw_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """Load raw data from CSV file.
        
        Args:
            file_path: Optional custom file path
            
        Returns:
            DataFrame with raw data
        """
        path = file_path or self.raw_data_path
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found at: {path}")
        
        logger.info(f"Loading data from {path}")
        df = pd.read_csv(path)
        
        logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
        return df
    
    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, list]:
        """Validate data quality.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []
        
        # Check for required columns
        required_cols = self.data_config['features'] + [self.data_config['target']]
        missing_cols = set(required_cols) - set(df.columns)
        
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")
        
        # Check for missing values
        missing_values = df[required_cols].isnull().sum()
        if missing_values.any():
            issues.append(f"Missing values detected:\n{missing_values[missing_values > 0]}")
        
        # Check target column values
        target_col = self.data_config['target']
        if target_col in df.columns:
            unique_values = df[target_col].unique()
            if not set(unique_values).issubset({0, 1}):
                issues.append(f"Target column should only contain 0 or 1, found: {unique_values}")
        
        # Check for duplicates
        n_duplicates = df.duplicated().sum()
        if n_duplicates > 0:
            issues.append(f"Found {n_duplicates} duplicate rows")
        
        # Check data types
        for col in self.data_config['features']:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                issues.append(f"Column {col} should be numeric, found {df[col].dtype}")
        
        is_valid = len(issues) == 0
        
        if is_valid:
            logger.info("Data validation passed")
        else:
            logger.warning(f"Data validation found {len(issues)} issues")
            for issue in issues:
                logger.warning(f"  - {issue}")
        
        return is_valid, issues
    
    def get_data_info(self, df: pd.DataFrame) -> dict:
        """Get comprehensive data information.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with data statistics
        """
        info = {
            'n_samples': len(df),
            'n_features': len(df.columns),
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': df.duplicated().sum(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        }
        
        # Target distribution
        target_col = self.data_config.get('target')
        if target_col and target_col in df.columns:
            info['target_distribution'] = df[target_col].value_counts().to_dict()
            info['target_balance'] = df[target_col].value_counts(normalize=True).to_dict()
        
        # Numeric features statistics
        numeric_cols = df.select_dtypes(include=['number']).columns
        info['numeric_summary'] = df[numeric_cols].describe().to_dict()
        
        return info
    
    def save_processed_data(self, df: pd.DataFrame, file_path: Optional[str] = None):
        """Save processed data to CSV.
        
        Args:
            df: DataFrame to save
            file_path: Optional custom file path
        """
        path = file_path or self.processed_data_path
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        df.to_csv(path, index=False)
        logger.info(f"Saved processed data to {path}")
    
    def load_processed_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """Load processed data.
        
        Args:
            file_path: Optional custom file path
            
        Returns:
            DataFrame with processed data
        """
        path = file_path or self.processed_data_path
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Processed data not found at: {path}")
        
        logger.info(f"Loading processed data from {path}")
        df = pd.read_csv(path)
        
        return df


def main():
    """Example usage of DataLoader."""
    # Initialize loader
    loader = DataLoader()
    
    # Load data
    df = loader.load_raw_data()
    
    # Validate data
    is_valid, issues = loader.validate_data(df)
    
    # Get data info
    info = loader.get_data_info(df)
    print("\n=== Data Information ===")
    print(f"Samples: {info['n_samples']}")
    print(f"Features: {info['n_features']}")
    print(f"Target distribution: {info.get('target_distribution', 'N/A')}")
    
    if is_valid:
        print("\n✓ Data validation passed")
    else:
        print(f"\n✗ Data validation failed with {len(issues)} issues")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
