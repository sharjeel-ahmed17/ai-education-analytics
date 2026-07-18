import numpy as np
import pandas as pd

def preprocess_timeseries(df: pd.DataFrame, seq_len: int = 10) -> np.ndarray:
    """
    Preprocesses student weekly timeseries data by scaling features and
    padding/truncating the sequence to a fixed window length.
    
    Args:
        df (pd.DataFrame): Input dataframe with columns [lms_logins, attendance_rate, quiz_average].
        seq_len (int): Target sequence length (time steps). Default is 10.
        
    Returns:
        np.ndarray: Preprocessed batch of shape (1, seq_len, 3) ready for RNN.
    """
    feature_cols = ['lms_logins', 'attendance_rate', 'quiz_average']
    
    # Fill missing columns if any
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
            
    # Extract copy
    data_df = df[feature_cols].copy().astype(np.float32)
    
    # Normalize features
    # LMS logins: assume [0, 30] range
    data_df['lms_logins'] = data_df['lms_logins'].clip(0.0, 30.0) / 30.0
    
    # Attendance: assume [0.0, 1.0] range. If percentage, convert to fraction
    max_attendance = data_df['attendance_rate'].max()
    if max_attendance > 1.0:
        data_df['attendance_rate'] = data_df['attendance_rate'] / 100.0
    data_df['attendance_rate'] = data_df['attendance_rate'].clip(0.0, 1.0)
    
    # Quiz scores: assume [0, 100] range. Convert to [0.0, 1.0]
    max_quiz = data_df['quiz_average'].max()
    if max_quiz > 1.0:
        data_df['quiz_average'] = data_df['quiz_average'] / 100.0
    data_df['quiz_average'] = data_df['quiz_average'].clip(0.0, 1.0)
    
    values = data_df.values
    
    # Pad / Truncate sequence
    curr_len = len(values)
    if curr_len < seq_len:
        # Pre-pad with zeros
        pad_len = seq_len - curr_len
        values = np.pad(values, ((pad_len, 0), (0, 0)), mode='constant')
    else:
        # Truncate to the most recent seq_len steps
        values = values[-seq_len:]
        
    # Return batch: (1, seq_len, 3)
    return np.expand_dims(values, axis=0)
