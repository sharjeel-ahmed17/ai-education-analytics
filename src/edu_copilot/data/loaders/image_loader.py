import os
from PIL import Image

def load_student_worksheet(file_path: str) -> Image.Image:
    """
    Loads a scanned student worksheet image.
    
    Args:
        file_path (str): Path to the image file.
        
    Returns:
        Image.Image: Loaded PIL Image in RGB format.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Worksheet image not found at: {file_path}")
    return Image.open(file_path).convert("RGB")
