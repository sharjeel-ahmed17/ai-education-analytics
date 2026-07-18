import pandas as pd
from typing import List
from edu_copilot.data.schemas import TabularStudentData

def load_tabular_csv(file_path: str) -> List[TabularStudentData]:
    """
    Loads student performance records from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file.
        
    Returns:
        List[TabularStudentData]: List of validated student tabular records.
    """
    df = pd.read_csv(file_path)
    records = []
    for _, row in df.iterrows():
        # Ensure correct boolean conversion for columns that might be stored as strings or numbers
        extracurricular = str(row['extracurricular_activities']).lower() in ('true', '1', 'yes')
        internet = str(row['internet_access']).lower() in ('true', '1', 'yes')
        
        record = TabularStudentData(
            student_id=str(row['student_id']),
            gpa=float(row['gpa']),
            attendance_rate=float(row['attendance_rate']),
            study_hours_weekly=float(row['study_hours_weekly']),
            parental_involvement=str(row['parental_involvement']),
            extracurricular_activities=extracurricular,
            sleep_hours=float(row['sleep_hours']),
            previous_grade=float(row['previous_grade']),
            family_income=str(row['family_income']),
            internet_access=internet
        )
        records.append(record)
    return records
