import keras
from keras import layers

def create_ann_model(input_dim: int = 9) -> keras.Model:
    """
    Creates a Keras Sequential Artificial Neural Network (ANN) model 
    for predicting student risk (binary classification: 1=At Risk, 0=On Track).
    
    Args:
        input_dim (int): Number of input features. Default is 9.
        
    Returns:
        keras.Model: Compiled Keras model.
    """
    # Simple feedforward architecture with Batch Normalization and Dropout for regularization
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')  # Output layer: probability of being at risk
    ])
    
    # Compile with Adam optimizer and binary crossentropy loss
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.005),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    
    return model
