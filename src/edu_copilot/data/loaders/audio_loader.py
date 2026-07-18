import os
import librosa

def load_student_audio(file_path: str, sr: int = 16000) -> tuple:
    """
    Loads a student oral exam audio recording.
    
    Args:
        file_path (str): Path to the audio file.
        sr (int): Target sample rate. Default is 16000Hz.
        
    Returns:
        tuple: (y, sr) where y is the audio signal (numpy array) and sr is the sample rate.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found at: {file_path}")
    # Load using librosa
    y, sample_rate = librosa.load(file_path, sr=sr)
    return y, sample_rate
