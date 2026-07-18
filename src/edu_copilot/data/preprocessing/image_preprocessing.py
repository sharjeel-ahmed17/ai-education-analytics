import numpy as np
from PIL import Image

def preprocess_worksheet(image: Image.Image, target_size=(128, 128)) -> np.ndarray:
    """
    Resizes and normalizes a PIL Image for CNN model consumption.
    
    Args:
        image (Image.Image): Ingested PIL Image.
        target_size (tuple): Target height and width. Default (128, 128).
        
    Returns:
        np.ndarray: Preprocessed image batch of shape (1, height, width, 3), scaled to [0.0, 1.0].
    """
    # Resize the image
    img_resized = image.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convert to float numpy array and normalize to [0, 1]
    img_arr = np.array(img_resized, dtype=np.float32) / 255.0
    
    # Add batch dimension: (1, height, width, 3)
    return np.expand_dims(img_arr, axis=0)
