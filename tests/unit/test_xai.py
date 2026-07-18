import pytest
from edu_copilot.xai.confidence import calculate_prediction_confidence
from edu_copilot.xai.gradcam_stub import generate_gradcam_heatmap

def test_calculate_prediction_confidence() -> None:
    """
    Verifies the sigmoid distance confidence calculation logic.
    """
    # Boundary tests
    assert calculate_prediction_confidence(0.5) == 0.0
    assert calculate_prediction_confidence(1.0) == 1.0
    assert calculate_prediction_confidence(0.0) == 1.0
    
    # Intermediate values
    assert pytest.approx(calculate_prediction_confidence(0.75)) == 0.50
    assert pytest.approx(calculate_prediction_confidence(0.20)) == 0.60
    assert pytest.approx(calculate_prediction_confidence(0.85)) == 0.70

def test_gradcam_stub_exception() -> None:
    """
    Verifies that invoking the Grad-CAM computer vision stub raises 
    NotImplementedError as visual models are explicitly out-of-scope.
    """
    with pytest.raises(NotImplementedError) as exc_info:
        generate_gradcam_heatmap(image_data=None, model=None, target_layer_name="conv_final")
        
    assert "Grad-CAM is not supported in this build" in str(exc_info.value)
