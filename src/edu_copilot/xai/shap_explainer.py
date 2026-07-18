import os
import shap
import pandas as pd
import numpy as np
import keras
from typing import Dict, Any, List
from edu_copilot.data.preprocessing.tabular_preprocessing import TabularPreprocessor

def get_shap_explanations(
    student_record: pd.DataFrame, 
    background_df: pd.DataFrame, 
    artifacts_dir: str = "src/edu_copilot/models/artifacts"
) -> Dict[str, Any]:
    """
    Generates SHAP feature attributions for a student's risk prediction.
    
    Args:
        student_record (pd.DataFrame): Tabular record of the target student (1 row).
        background_df (pd.DataFrame): Training subset used as reference background.
        artifacts_dir (str): Directory containing model/scaler files.
        
    Returns:
        Dict[str, Any]: Base expected value, feature attributions, and display details.
    """
    model_path = os.path.join(artifacts_dir, "student_model.keras")
    preprocessor_path = os.path.join(artifacts_dir, "preprocessor.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
        raise FileNotFoundError("Model or preprocessor artifact missing.")
        
    # Load model and preprocessor
    model = keras.models.load_model(model_path)
    preprocessor = TabularPreprocessor.load(preprocessor_path)
    
    # Preprocess inputs
    X_bg = preprocessor.transform(background_df)
    X_target = preprocessor.transform(student_record)
    
    # Summarize background data using k-means to speed up KernelExplainer calculations
    # KernelExplainer scales with O(N * background_size)
    bg_summary = shap.kmeans(X_bg, min(10, len(X_bg)))
    
    def predict_fn(x: np.ndarray) -> np.ndarray:
        # Returns probability values [0.0 - 1.0]
        preds = model.predict(x, verbose=0)
        return preds.flatten()

    explainer = shap.KernelExplainer(predict_fn, bg_summary)
    
    # Compute SHAP values for the single target student
    shap_values = explainer.shap_values(X_target)
    
    # Handle single output vs multiple outputs list format from SHAP
    if isinstance(shap_values, list):
        # In binary classification, SHAP may return a list [shap_values_class0, shap_values_class1]
        # or just single array. We extract the array for the positive class (At Risk).
        shap_values_array = shap_values[0]
    else:
        shap_values_array = shap_values
        
    # Map values back to their preprocessed feature names
    feature_names = preprocessor.feature_columns
    attributions = {}
    
    for idx, name in enumerate(feature_names):
        attributions[name] = float(shap_values_array[0, idx])
        
    # Extract the scaled input values for plotting
    scaled_inputs = {}
    for idx, name in enumerate(feature_names):
        scaled_inputs[name] = float(X_target[0, idx])
        
    return {
        "base_value": float(explainer.expected_value),
        "prediction_value": float(predict_fn(X_target)[0]),
        "attributions": attributions,
        "scaled_inputs": scaled_inputs,
        "feature_names": feature_names
    }
