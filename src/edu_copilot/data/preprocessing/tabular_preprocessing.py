import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List

class TabularPreprocessor:
    """
    Handles preprocessing of student tabular data, including categorical mapping,
    boolean conversion, and feature scaling.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = [
            'gpa', 
            'attendance_rate', 
            'study_hours_weekly', 
            'sleep_hours', 
            'previous_grade',
            'parental_involvement_encoded', 
            'family_income_encoded',
            'extracurricular_activities_encoded', 
            'internet_access_encoded'
        ]
        self.is_fitted = False

    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        df_encoded = df.copy()
        
        # Ordinal mapping for categorical columns
        parent_map = {'low': 0, 'medium': 1, 'high': 2}
        income_map = {'low': 0, 'medium': 1, 'high': 2}
        
        df_encoded['parental_involvement_encoded'] = df_encoded['parental_involvement'].astype(str).str.lower().map(parent_map).fillna(1)
        df_encoded['family_income_encoded'] = df_encoded['family_income'].astype(str).str.lower().map(income_map).fillna(1)
        
        # Boolean to integer mapping
        df_encoded['extracurricular_activities_encoded'] = df_encoded['extracurricular_activities'].astype(int)
        df_encoded['internet_access_encoded'] = df_encoded['internet_access'].astype(int)
        
        return df_encoded

    def fit(self, df: pd.DataFrame) -> 'TabularPreprocessor':
        """
        Fits the scaler on the preprocessed tabular columns.
        """
        df_encoded = self._encode_categorical(df)
        self.scaler.fit(df_encoded[self.feature_columns].values)
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transforms raw tabular columns into scaled arrays.
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor has not been fitted yet.")
        df_encoded = self._encode_categorical(df)
        return self.scaler.transform(df_encoded[self.feature_columns].values)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Fits on the data and transforms it in one step.
        """
        return self.fit(df).transform(df)

    def save(self, file_path: str) -> None:
        """
        Serializes the preprocessor to a pickle file.
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file_path: str) -> 'TabularPreprocessor':
        """
        Loads the preprocessor from a pickle file.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Preprocessor file not found at: {file_path}")
        with open(file_path, 'rb') as f:
            return pickle.load(f)
