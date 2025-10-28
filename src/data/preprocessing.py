"""
Data preprocessing and feature engineering for LoL match prediction.
"""

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional
import joblib
import logging
import os

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handle data preprocessing and feature engineering."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize preprocessor with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_config = self.config['data']
        self.feature_config = self.config.get('feature_engineering', {})
        self.scaler = None
        
    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features from existing ones.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with additional features
        """
        df = df.copy()
        
        if not self.feature_config.get('create_ratios', False):
            return df
        
        logger.info("Creating derived features...")
        
        # KDA Ratio: (Kills + Assists) / Deaths
        # Add small constant to avoid division by zero
        df['kda_ratio'] = (df['kills'] + df['assists']) / (df['deaths'] + 1)
        
        # Gold per CS (Creep Score)
        df['gold_per_cs'] = df['gold_earned'] / (df['cs'] + 1)
        
        # Vision Score (Ward contribution)
        df['vision_score'] = df['wards_placed'] + df['wards_killed']
        
        # Damage Efficiency (damage per gold)
        df['damage_efficiency'] = df['damage_dealt'] / (df['gold_earned'] + 1)
        
        # Kill Participation (simplified - assumes 5v5)
        df['kill_participation'] = (df['kills'] + df['assists']) / (df['kills'].max() + 1)
        
        # Death Rate
        df['death_rate'] = df['deaths'] / (df['kills'] + df['assists'] + df['deaths'] + 1)
        
        # Gold Efficiency
        df['gold_efficiency'] = df['gold_earned'] / (df['cs'] + df['kills'] * 300 + 1)
        
        # Combat Score (composite metric)
        df['combat_score'] = (
            df['kills'] * 3 + 
            df['assists'] * 1.5 - 
            df['deaths'] * 2 +
            df['damage_dealt'] / 1000
        )
        
        logger.info(f"Created {8} derived features")
        
        return df
    
    def handle_outliers(self, df: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """Handle outliers in the dataset.
        
        Args:
            df: Input DataFrame
            method: Method to use ('iqr' or 'zscore')
            
        Returns:
            DataFrame with outliers handled
        """
        if not self.feature_config.get('handle_outliers', False):
            return df
        
        df = df.copy()
        numeric_cols = df.select_dtypes(include=['number']).columns
        numeric_cols = [col for col in numeric_cols if col != self.data_config['target']]
        
        logger.info(f"Handling outliers using {method} method...")
        
        if method == 'iqr':
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Clip outliers
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        elif method == 'zscore':
            for col in numeric_cols:
                mean = df[col].mean()
                std = df[col].std()
                lower_bound = mean - 3 * std
                upper_bound = mean + 3 * std
                
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        return df
    
    def scale_features(self, 
                      X_train: pd.DataFrame, 
                      X_test: Optional[pd.DataFrame] = None,
                      method: str = 'standard') -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Scale features using specified method.
        
        Args:
            X_train: Training features
            X_test: Optional test features
            method: Scaling method ('standard', 'minmax', 'robust')
            
        Returns:
            Tuple of scaled training and test features
        """
        method = method or self.feature_config.get('scaling_method', 'standard')
        
        logger.info(f"Scaling features using {method} scaler...")
        
        # Initialize scaler
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        # Fit on training data
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        # Transform test data if provided
        X_test_scaled = None
        if X_test is not None:
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
        
        return X_train_scaled, X_test_scaled
    
    def split_data(self, 
                   df: pd.DataFrame,
                   test_size: Optional[float] = None,
                   val_size: Optional[float] = None,
                   random_state: Optional[int] = None) -> Tuple:
        """Split data into train, validation, and test sets.
        
        Args:
            df: Input DataFrame
            test_size: Proportion of test set
            val_size: Proportion of validation set from training data
            random_state: Random state for reproducibility
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        test_size = test_size or self.data_config.get('test_size', 0.2)
        val_size = val_size or self.data_config.get('validation_size', 0.1)
        random_state = random_state or self.data_config.get('random_state', 42)
        
        # Separate features and target
        target_col = self.data_config['target']
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: train and validation
        if val_size > 0:
            val_size_adjusted = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, 
                test_size=val_size_adjusted, 
                random_state=random_state,
                stratify=y_temp
            )
        else:
            X_train, y_train = X_temp, y_temp
            X_val, y_val = None, None
        
        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val) if X_val is not None else 0}, Test: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def preprocess_pipeline(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Run complete preprocessing pipeline.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit the scaler (True for training, False for inference)
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info("Running preprocessing pipeline...")
        
        # Create derived features
        df = self.create_derived_features(df)
        
        # Handle outliers
        df = self.handle_outliers(
            df, 
            method=self.feature_config.get('outlier_method', 'iqr')
        )
        
        logger.info("Preprocessing pipeline completed")
        
        return df
    
    def save_scaler(self, file_path: str = "models/production/scaler.pkl"):
        """Save the fitted scaler.
        
        Args:
            file_path: Path to save the scaler
        """
        if self.scaler is None:
            raise ValueError("Scaler has not been fitted yet")
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(self.scaler, file_path)
        logger.info(f"Scaler saved to {file_path}")
    
    def load_scaler(self, file_path: str = "models/production/scaler.pkl"):
        """Load a saved scaler.
        
        Args:
            file_path: Path to the saved scaler
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Scaler not found at: {file_path}")
        
        self.scaler = joblib.load(file_path)
        logger.info(f"Scaler loaded from {file_path}")


def main():
    """Example usage of DataPreprocessor."""
    from data_loader import DataLoader
    
    # Load data
    loader = DataLoader()
    df = loader.load_raw_data()
    
    # Preprocess
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.preprocess_pipeline(df)
    
    print("\n=== Preprocessing Results ===")
    print(f"Original features: {len(loader.data_config['features'])}")
    print(f"Total features after engineering: {len(df_processed.columns) - 1}")  # -1 for target
    print(f"\nNew columns: {[col for col in df_processed.columns if col not in df.columns]}")
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(df_processed)
    
    # Scale features
    X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
    
    print(f"\n✓ Data preprocessing completed")
    print(f"  Train set: {X_train_scaled.shape}")
    print(f"  Val set: {X_val.shape if X_val is not None else 'N/A'}")
    print(f"  Test set: {X_test_scaled.shape}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
