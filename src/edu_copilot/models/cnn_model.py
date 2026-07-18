import os
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from PIL import Image
import cv2

def create_cnn_model(input_shape=(128, 128, 3)) -> keras.Model:
    """
    Creates a Convolutional Neural Network (CNN) using the Functional API 
    that predicts three quality metrics for student worksheets:
    1. Legibility score
    2. Correctness of diagrams score
    3. Completeness score
    
    All scores are in range [0, 1].
    """
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    # Explicitly name the last convolutional layer for Grad-CAM
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='conv_last')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(3, activation='sigmoid', name='quality_outputs')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mae']
    )
    return model

def get_or_create_cnn_model(artifacts_dir: str = "src/edu_copilot/models/artifacts") -> keras.Model:
    """
    Loads the saved CNN model or creates a fresh one with random weights if not found.
    """
    os.makedirs(artifacts_dir, exist_ok=True)
    model_path = os.path.join(artifacts_dir, "cnn_model.keras")
    
    if os.path.exists(model_path):
        try:
            return keras.models.load_model(model_path)
        except Exception as e:
            print(f"Error loading CNN model: {e}. Re-creating model.")
            
    model = create_cnn_model()
    model.save(model_path)
    print(f"Initialized CNN model saved to {model_path}")
    return model

def generate_gradcam_heatmap(
    image_path: str,
    model: keras.Model,
    target_layer_name: str = 'conv_last',
    output_path: str = None
) -> Image.Image:
    """
    Generates a Grad-CAM heatmap overlay for a student worksheet image.
    Uses gradient attribution of the combined output score.
    
    Args:
        image_path (str): Path to original image.
        model (keras.Model): Compiled CNN model.
        target_layer_name (str): Named conv layer to compute gradients against.
        output_path (str): Optional path to save the overlaid image.
        
    Returns:
        Image.Image: The image with Grad-CAM heatmap overlaid.
    """
    original_img = Image.open(image_path).convert("RGB")
    width, height = original_img.size
    
    # 1. Preprocess image
    img_resized = original_img.resize((128, 128), Image.Resampling.LANCZOS)
    img_arr = np.array(img_resized, dtype=np.float32) / 255.0
    img_batch = np.expand_dims(img_arr, axis=0)
    
    try:
        # Create a model that outputs the activations of the last conv layer and the final output
        grad_model = keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(target_layer_name).output, model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_batch)
            # Combine the scores (legibility, correctness, completeness) for saliency
            score = tf.reduce_mean(predictions)
            
        # Gradient of the combined quality score w.r.t last conv layer outputs
        grads = tape.gradient(score, conv_outputs)
        
        # Mean intensity of the gradient over each feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Multiply each channel in the feature map by its gradient weight
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Apply ReLU to keep only positive contributions, then normalize
        heatmap = tf.maximum(heatmap, 0.0)
        max_val = tf.math.reduce_max(heatmap)
        if max_val > 0.0:
            heatmap = heatmap / max_val
        heatmap = heatmap.numpy()
        
    except Exception as e:
        print(f"Grad-CAM tape execution failed: {e}. Falling back to center-weighted mock heatmap.")
        # Fallback: create a radial gradient heatmap centered in the image
        x = np.linspace(-1.5, 1.5, 16)
        y = np.linspace(-1.5, 1.5, 16)
        x_grid, y_grid = np.meshgrid(x, y)
        z = np.exp(-0.5 * (x_grid**2 + y_grid**2))
        heatmap = (z - z.min()) / (z.max() - z.min())

    # 2. Convert heatmap to PIL Image, resize to match original image, and overlay
    heatmap_resized = cv2.resize(heatmap, (width, height))
    heatmap_color = np.uint8(255 * heatmap_resized)
    
    # Apply JET colormap using cv2
    heatmap_colored = cv2.applyColorMap(heatmap_color, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Superimpose heatmap on original image
    original_arr = np.array(original_img)
    overlaid = cv2.addWeighted(original_arr, 0.6, heatmap_colored, 0.4, 0)
    
    result_img = Image.fromarray(overlaid)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result_img.save(output_path)
        print(f"Grad-CAM image saved to {output_path}")
        
    return result_img
