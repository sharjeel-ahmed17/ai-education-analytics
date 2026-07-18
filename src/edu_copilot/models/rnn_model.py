import os
import keras
from keras import layers

def create_rnn_model(
    seq_len: int, 
    num_features: int, 
    rnn_type: str = 'lstm', 
    units: int = 32
) -> keras.Model:
    """
    Standard builder function for compiling sequential RNN (LSTM/GRU) architectures.
    
    Args:
        seq_len (int): Input sequence length (number of timesteps).
        num_features (int): Number of features per timestep.
        rnn_type (str): Type of recurrent layer ('lstm' or 'gru'). Default is 'lstm'.
        units (int): Hidden units size. Default is 32.
        
    Returns:
        keras.Model: Compiled Keras model.
    """
    rnn_type = rnn_type.lower()
    if rnn_type == 'gru':
        recurrent_layer = layers.GRU(units, return_sequences=False)
    else:
        recurrent_layer = layers.LSTM(units, return_sequences=False)
        
    model = keras.Sequential([
        layers.Input(shape=(seq_len, num_features)),
        recurrent_layer,
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid') # score / risk probability output
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def get_or_create_rnn_model(
    name: str, 
    seq_len: int, 
    num_features: int, 
    rnn_type: str = 'lstm', 
    artifacts_dir: str = "src/edu_copilot/models/artifacts"
) -> keras.Model:
    """
    Retrieves the saved RNN model or compiles and saves a fresh instance with random weights.
    
    Args:
        name (str): Unique filename (e.g. 'timeseries_model' or 'audio_model').
        seq_len (int): Timestep sequence size.
        num_features (int): Features size.
        rnn_type (str): 'lstm' or 'gru' architecture selection.
        artifacts_dir (str): Location of serialized weights.
        
    Returns:
        keras.Model: Loaded Keras RNN model.
    """
    os.makedirs(artifacts_dir, exist_ok=True)
    model_path = os.path.join(artifacts_dir, f"{name}.keras")
    
    if os.path.exists(model_path):
        try:
            return keras.models.load_model(model_path)
        except Exception as e:
            print(f"Error loading RNN model '{name}': {e}. Re-creating model.")
            
    model = create_rnn_model(seq_len, num_features, rnn_type)
    model.save(model_path)
    print(f"Initialized RNN model saved to {model_path}")
    return model
