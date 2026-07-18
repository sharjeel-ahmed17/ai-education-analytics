import os
import pandas as pd

def load_student_timeseries(file_path: str) -> pd.DataFrame:
    """
    Loads weekly student engagement and quiz average sequences.
    
    Args:
        file_path (str): Path to the time-series CSV file.
        
    Returns:
        pd.DataFrame: Weekly metric sequence columns (e.g. week, lms_logins, attendance_rate, quiz_average).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Time-series file not found at: {file_path}")
    return pd.read_csv(file_path)
