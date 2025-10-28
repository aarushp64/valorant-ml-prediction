#!/usr/bin/env python3
"""
Simple ML Training Script - Valorant Player Tier Prediction
Build a basic model that we can wire into the UI
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def load_and_prep_valorant_data():
    """Load the Valorant dataset and prep for ML"""
    print("Loading Valorant dataset...")
    
    df = pd.read_csv('data/raw/valorant_dataset_v3.csv')
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    print("Columns:", df.columns.tolist())
    
    # Clean comma-separated numeric columns
    numeric_cols = ['assists', 'damage_received', 'headshots', 'kills', 'matches', 'deaths', 'damage']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '').astype(float)
    
    # Drop rows with missing tier info
    df = df.dropna(subset=['tier'])
    print(f"After removing NaN tiers: {len(df)} rows")
    
    # Create features
    feature_cols = ['kills', 'deaths', 'assists', 'damage', 'headshots', 'matches']
    
    # Filter to rows with all features available
    df = df.dropna(subset=feature_cols)
    print(f"After removing NaN features: {len(df)} rows")
    
    # Create derived features
    df['kd_ratio'] = df['kills'] / (df['deaths'] + 1)
    df['damage_per_match'] = df['damage'] / (df['matches'] + 1)
    df['kills_per_match'] = df['kills'] / (df['matches'] + 1)
    
    # Final feature set
    features = feature_cols + ['kd_ratio', 'damage_per_match', 'kills_per_match']
    
    X = df[features]
    y = df['tier']
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target distribution: {y.value_counts()}")
    
    return X, y, features

def train_model():
    """Train a simple random forest model"""
    print("\n" + "="*50)
    print("TRAINING VALORANT TIER PREDICTION MODEL")
    print("="*50)
    
    # Load data
    X, y, feature_names = load_and_prep_valorant_data()
    
    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nTrain size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")
    
    # Train model
    print("\nTraining Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)
    
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    print(f"\nTrain Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Feature importance
    importance = model.feature_importances_
    feature_importance = list(zip(feature_names, importance))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 5 Feature Importances:")
    for feat, imp in feature_importance[:5]:
        print(f"  {feat}: {imp:.4f}")
    
    # Save model artifacts
    os.makedirs('models/simple', exist_ok=True)
    
    # Save model
    joblib.dump(model, 'models/simple/valorant_tier_model.pkl')
    joblib.dump(scaler, 'models/simple/scaler.pkl')
    joblib.dump(label_encoder, 'models/simple/label_encoder.pkl')
    
    # Save metadata
    metadata = {
        'model_type': 'RandomForestClassifier',
        'target': 'tier',
        'features': feature_names,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'n_classes': len(label_encoder.classes_),
        'classes': label_encoder.classes_.tolist()
    }
    
    import json
    with open('models/simple/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Model saved to models/simple/")
    print(f"   - valorant_tier_model.pkl")
    print(f"   - scaler.pkl") 
    print(f"   - label_encoder.pkl")
    print(f"   - model_metadata.json")
    
    return model, scaler, label_encoder, feature_names, metadata

if __name__ == "__main__":
    try:
        model, scaler, label_encoder, features, metadata = train_model()
        print(f"\n🎉 SUCCESS! Model ready for use in the app.")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()