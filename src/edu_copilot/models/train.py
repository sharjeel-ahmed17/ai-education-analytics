import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from edu_copilot.data.preprocessing.tabular_preprocessing import TabularPreprocessor
from edu_copilot.models.ann_model import create_ann_model

def train_model(csv_path: str, artifacts_dir: str = "src/edu_copilot/models/artifacts") -> None:
    """
    Trains the ANN on student tabular records and saves training artifacts.
    
    Args:
        csv_path (str): Path to the tabular CSV data file.
        artifacts_dir (str): Directory where trained model and scaler will be saved.
    """
    print(f"Loading training data from {csv_path}...")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Training CSV not found at: {csv_path}")
        
    df = pd.read_csv(csv_path)
    
    # Ensure the target column exists. If not, generate a synthetic one for development.
    if 'at_risk' not in df.columns:
        print("Target column 'at_risk' not found. Generating labels based on GPA and attendance rate...")
        # At risk if GPA is low or attendance is low
        df['at_risk'] = ((df['gpa'] < 2.5) | (df['attendance_rate'] < 0.85)).astype(int)
        
    # Standardize/Scale features
    preprocessor = TabularPreprocessor()
    X = preprocessor.fit_transform(df)
    y = df['at_risk'].values
    
    # Stratified split to maintain balance of risk/no-risk
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create model
    model = create_ann_model(input_dim=X.shape[1])
    
    # Train
    print("Starting Keras model training...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=40,
        batch_size=8,
        verbose=1
    )
    
    # Save artifacts
    os.makedirs(artifacts_dir, exist_ok=True)
    model_path = os.path.join(artifacts_dir, "student_model.keras")
    preprocessor_path = os.path.join(artifacts_dir, "preprocessor.pkl")
    
    model.save(model_path)
    preprocessor.save(preprocessor_path)
    print(f"Model successfully saved to {model_path}")
    print(f"Preprocessor successfully saved to {preprocessor_path}")

if __name__ == "__main__":
    # Resolve project root relative path
    default_csv = os.path.join("data", "sample", "student_records.csv")
    csv_file = sys.argv[1] if len(sys.argv) > 1 else default_csv
    
    train_model(csv_file)
