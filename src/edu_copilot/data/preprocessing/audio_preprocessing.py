import numpy as np
import librosa

def preprocess_audio(
    y: np.ndarray, 
    sr: int = 16000, 
    n_mfcc: int = 13, 
    seq_len: int = 50
) -> np.ndarray:
    """
    Extracts MFCC features from raw audio signal and pads/truncates 
    the resulting sequence to a fixed length for RNN input.
    
    Args:
        y (np.ndarray): Raw audio signal.
        sr (int): Sample rate. Default is 16000Hz.
        n_mfcc (int): Number of MFCC coefficients to extract. Default is 13.
        seq_len (int): Target sequence length (frames). Default is 50.
        
    Returns:
        np.ndarray: Normalized MFCC sequence of shape (1, seq_len, n_mfcc).
    """
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc) # shape: (n_mfcc, time_steps)
    
    # Transpose to shape: (time_steps, n_mfcc)
    mfccs = mfccs.T
    
    # Pad or truncate sequence
    curr_len = mfccs.shape[0]
    if curr_len < seq_len:
        pad_len = seq_len - curr_len
        mfccs = np.pad(mfccs, ((0, pad_len), (0, 0)), mode='constant')
    else:
        mfccs = mfccs[:seq_len, :]
        
    # Standardize features (z-score normalization) for model stability
    mean = np.mean(mfccs, axis=0, keepdims=True)
    std = np.std(mfccs, axis=0, keepdims=True) + 1e-8
    mfccs_normalized = (mfccs - mean) / std
    
    # Add batch dimension: (1, seq_len, n_mfcc)
    return np.expand_dims(mfccs_normalized, axis=0)
