import os
import sys
import numpy as np
import pandas as pd
import keras
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from edu_copilot.data.preprocessing.tabular_preprocessing import TabularPreprocessor

def evaluate_model(csv_path: str, artifacts_dir: str = "src/edu_copilot/models/artifacts") -> dict:
    """
    Evaluates the trained ANN model on a dataset and prints performance metrics.
    
    Args:
        csv_path (str): Path to evaluation CSV.
        artifacts_dir (str): Location of saved model artifacts.
        
    Returns:
        dict: Performance metrics dictionary.
    """
    model_path = os.path.join(artifacts_dir, "student_model.keras")
    preprocessor_path = os.path.join(artifacts_dir, "preprocessor.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
        raise FileNotFoundError("Trained model or preprocessor was not found. Train the model first.")
        
    # Load model and scaler
    model = keras.models.load_model(model_path)
    preprocessor = TabularPreprocessor.load(preprocessor_path)
    
    df = pd.read_csv(csv_path)
    if 'at_risk' not in df.columns:
        df['at_risk'] = ((df['gpa'] < 2.5) | (df['attendance_rate'] < 0.85)).astype(int)
        
    X = preprocessor.transform(df)
    y_true = df['at_risk'].values
    
    # Run predictions
    y_prob = model.predict(X).flatten()
    y_pred = (y_prob >= 0.5).astype(int)
    
    # Calculate metrics
    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    # Handle single class case for ROC AUC (if test set is unbalanced)
    if len(np.unique(y_true)) > 1:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    else:
        roc_auc = 1.0
        
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    }
    
    print("\n--- Model Evaluation Report ---")
    for metric_name, val in metrics.items():
        print(f"{metric_name.upper():<10}: {val:.4f}")
    print("--------------------------------\n")
    
    return metrics

if __name__ == "__main__":
    default_csv = os.path.join("data", "sample", "student_records.csv")
    csv_file = sys.argv[1] if len(sys.argv) > 1 else default_csv
    evaluate_model(csv_file)
