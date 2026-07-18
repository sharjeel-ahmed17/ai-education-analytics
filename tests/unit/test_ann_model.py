import keras
from edu_copilot.models.ann_model import create_ann_model

def test_create_ann_model() -> None:
    """
    Verifies that the Keras ANN model instantiates with the correct
    input dimensions, compilation configs, and structural layers.
    """
    input_dim = 9
    model = create_ann_model(input_dim=input_dim)
    
    # 1. Assert correct keras model type
    assert isinstance(model, keras.Model)
    
    # 2. Check input and output tensor dimensions
    assert model.input_shape == (None, input_dim)
    assert model.output_shape == (None, 1)
    
    # 3. Assert compiling configurations
    assert model.optimizer is not None
    assert model.loss == "binary_crossentropy"
