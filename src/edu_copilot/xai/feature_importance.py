import os
import numpy as np
import pandas as pd
import keras
from sklearn.metrics import log_loss
from typing import Dict
from edu_copilot.data.preprocessing.tabular_preprocessing import TabularPreprocessor

def get_permutation_importance(
    df: pd.DataFrame, 
    artifacts_dir: str = "src/edu_copilot/models/artifacts"
) -> Dict[str, float]:
    """
    Computes global feature importances for the tabular ANN features using Permutation Importance.
    
    Args:
        df (pd.DataFrame): Validation or background dataset.
        artifacts_dir (str): Directory containing model/scaler files.
        
    Returns:
        Dict[str, float]: Standardized feature importance scores (sum to 1.0).
    """
    model_path = os.path.join(artifacts_dir, "student_model.keras")
    preprocessor_path = os.path.join(artifacts_dir, "preprocessor.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
        raise FileNotFoundError("Model or preprocessor artifact missing.")
        
    model = keras.models.load_model(model_path)
    preprocessor = TabularPreprocessor.load(preprocessor_path)
    
    # Handle target label
    if 'at_risk' not in df.columns:
        df['at_risk'] = ((df['gpa'] < 2.5) | (df['attendance_rate'] < 0.85)).astype(int)
        
    X = preprocessor.transform(df)
    y = df['at_risk'].values
    
    # Calculate baseline loss
    baseline_preds = model.predict(X, verbose=0).flatten()
    baseline_loss = log_loss(y, baseline_preds, labels=[0, 1])
    
    importances = {}
    feature_names = preprocessor.feature_columns
    
    # Shuffle columns one by one and record the increase in loss
    for idx, col in enumerate(feature_names):
        X_shuffled = X.copy()
        # Shuffle values within the column
        np.random.shuffle(X_shuffled[:, idx])
        
        shuffled_preds = model.predict(X_shuffled, verbose=0).flatten()
        shuffled_loss = log_loss(y, shuffled_preds, labels=[0, 1])
        
        # Performance drop (higher loss increase = higher importance)
        importances[col] = float(max(0.0, shuffled_loss - baseline_loss))
        
    # Normalize values to sum to 1.0
    sum_importances = sum(importances.values())
    if sum_importances > 0:
        importances = {k: v / sum_importances for k, v in importances.items()}
    else:
        # Equal distribution if all values are 0
        importances = {k: 1.0 / len(feature_names) for k in feature_names}
        
    # Sort descending
    sorted_importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
    
    return sorted_importances
