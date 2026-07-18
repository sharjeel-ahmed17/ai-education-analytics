"""
Grad-CAM Explainability Hook (Extension Point)

IMPORTANT NOTE:
This is an explicit stub. It is NOT wired to any model in this build,
as image and audio modalities are explicitly out-of-scope for the primary predictor.

In a future version of this application, if computer vision (CNN) is added
to analyze scans of physical report cards or written student exam papers, 
this hook will generate Grad-CAM (Gradient-weighted Class Activation Mapping) 
heatmaps to highlight which physical regions of the document images (e.g., specific
handwritten feedback or grade boxes) the model focused on when making predictions.
"""

from typing import Any

def generate_gradcam_heatmap(
    image_data: Any, 
    model: Any, 
    target_layer_name: str
) -> Any:
    """
    Placeholder/stub function for generating Grad-CAM heatmaps for document scans.
    
    Args:
        image_data (Any): Raw or preprocessed image array.
        model (Any): The CNN model (e.g., ResNet/EfficientNet).
        target_layer_name (str): Name of the final conv layer to extract gradients from.
        
    Raises:
        NotImplementedError: Explaining that visual modalities are not in scope.
    """
    raise NotImplementedError(
        "Grad-CAM is not supported in this build. "
        "Image and CNN-based modalities are out of scope. The primary predictive engine "
        "only uses tabular records analyzed via a feedforward ANN."
    )
