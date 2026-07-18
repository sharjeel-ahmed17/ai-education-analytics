import os
import sys
import shutil
import subprocess
import tensorflow as tf
import keras

def export_to_onnx(artifacts_dir: str = "src/edu_copilot/models/artifacts") -> str:
    """
    Loads the trained Keras model, saves it in TensorFlow SavedModel format,
    and converts it to ONNX format.
    
    Args:
        artifacts_dir (str): Directory containing model files.
        
    Returns:
        str: Path to the exported ONNX model file.
    """
    keras_model_path = os.path.join(artifacts_dir, "student_model.keras")
    onnx_model_path = os.path.join(artifacts_dir, "student_model.onnx")
    
    if not os.path.exists(keras_model_path):
        raise FileNotFoundError(f"Keras model not found at: {keras_model_path}. Run training first.")
        
    print(f"Loading Keras model from {keras_model_path}...")
    model = keras.models.load_model(keras_model_path)
    
    temp_saved_model_dir = os.path.join(artifacts_dir, "temp_saved_model")
    if os.path.exists(temp_saved_model_dir):
        shutil.rmtree(temp_saved_model_dir)
        
    print(f"Exporting model to temporary SavedModel format at {temp_saved_model_dir}...")
    # Save in standard TensorFlow SavedModel format
    tf.saved_model.save(model, temp_saved_model_dir)
    
    print("Converting SavedModel to ONNX...")
    # Build the CLI invocation command for tf2onnx
    cmd = [
        sys.executable, "-m", "tf2onnx.convert",
        "--saved-model", temp_saved_model_dir,
        "--output", onnx_model_path,
        "--opset", "13"
    ]
    
    # Execute tf2onnx process
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Cleanup temp directory
    try:
        shutil.rmtree(temp_saved_model_dir)
    except Exception as e:
        print(f"Warning: Failed to remove temporary directory: {e}")
        
    if result.returncode != 0:
        print("tf2onnx conversion failed:")
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"tf2onnx CLI failed with exit code {result.returncode}")
        
    print(f"ONNX model saved successfully to {onnx_model_path}")
    return onnx_model_path

if __name__ == "__main__":
    export_to_onnx()
